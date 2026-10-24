"""Explicit user-authorized wall time failures for fixed unfiltered scenes.

They count as zero-score attempts, never as completed policy trajectories.
The paired fixed-scene policy also counts stalled episode workers. Crashes,
malformed evidence and cleanup failures remain fatal.
"""
import hashlib
import json
import math
import os
from pathlib import Path

TIMEOUT_MODE = 'wall900_zero_continue_v1'
UNFILTERED_TIMEOUT_MODE = 'unfiltered_wall900_stall240_zero_continue_v1'
PAIRED_TIMEOUT_MODE = 'paired_wall900_stall240_zero_continue_v1'


def validate_timeout_policy(config):
    policy = config.get('timeout_policy')
    if policy is None:
        return False
    if (not isinstance(policy, dict) or policy.get('mode') not in (TIMEOUT_MODE, PAIRED_TIMEOUT_MODE, UNFILTERED_TIMEOUT_MODE)
            or policy.get('episode_seconds') != 900 or policy.get('stall_seconds') != 240
            or not policy.get('user_authorization') or not config.get('fixed_scenes')
            or config.get('expert_quota') or config.get('valid_per_group')):
        raise ValueError('Timeout scoring requires an authorized fixed 900/240 plan')
    if policy['mode'] == TIMEOUT_MODE and config.get('expert_check'):
        raise ValueError('Legacy wall-time policy requires unfiltered scenes')
    if policy['mode'] == PAIRED_TIMEOUT_MODE and not config.get('paired_reference'):
        raise ValueError('Paired timeout policy requires the frozen scene reference')
    if policy['mode'] == UNFILTERED_TIMEOUT_MODE and (config.get('expert_check') or config.get('paired_reference')):
        raise ValueError('Unfiltered timeout mode requires experts off and no E0 reference')
    return True


def timeout_failure(config, row):
    if row.get('status') != 'policy_timeout':
        return False
    if not validate_timeout_policy(config):
        return False
    mode = config['timeout_policy']['mode']
    kind = row.get('timeout_kind')
    guards = {'episode_wall_clock': 900}
    if mode in (PAIRED_TIMEOUT_MODE, UNFILTERED_TIMEOUT_MODE):
        guards['action_chunk_stall'] = 240
    return (row.get('timeout_policy') == mode and kind in guards
            and row.get('timeout_seconds') == guards.get(kind)
            and (mode == PAIRED_TIMEOUT_MODE or row.get('expert_attempts') == 0)
            and row.get('success') is False and row.get('scored') is True
            and row.get('policy_completed') is False
            and isinstance(row.get('timeout_evidence'), dict)
            and bool(row['timeout_evidence'].get('files_sha256')))


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def build_timeout_record(config, scene, evidence_folder, *, error, elapsed, progress, segment,
                         kind='episode_wall_clock'):
    if not validate_timeout_policy(config) or scene not in config['fixed_scenes']:
        raise ValueError('Timeout scene is not in the approved fixed plan')
    folder = Path(evidence_folder)
    mode = config['timeout_policy']['mode']
    limit = {'episode_wall_clock': 900, 'action_chunk_stall': 240}.get(kind)
    if limit is None or (kind == 'action_chunk_stall' and mode not in (PAIRED_TIMEOUT_MODE, UNFILTERED_TIMEOUT_MODE)):
        raise ValueError('Timeout kind is outside the authorized policy')
    if (folder/'episode.json').exists():
        raise ValueError('Do not replace an existing terminal episode record')
    if any(progress.get(k) != scene[k] for k in ('task', 'setting', 'seed')):
        raise ValueError('Timeout progress does not identify the selected scene')
    pred, traj = [_lines(folder/name) for name in ('predictions.jsonl', 'trajectory.jsonl')]
    if not pred and mode in (TIMEOUT_MODE, UNFILTERED_TIMEOUT_MODE):
        raise ValueError('No policy prediction evidence; diagnose setup or process failure')
    for i,p in enumerate(pred):
        actions=p['predicted_actions']
        if (p['chunk'] != i or p['action_count_before'] != i*config['execute'] or len(actions) != 50
                or any(len(a)!=14 or any(not math.isfinite(v) for v in a) for a in actions)):
            raise ValueError('Invalid saved timeout prediction')
    if len(traj) not in (len(pred)-1,len(pred)):
        raise ValueError('Timeout trajectory is not a completed prefix')
    for p,t in zip(pred,traj):
        if (any(t.get(k)!=v for k,v in p.items()) or t.get('action_count_after')!=(p['chunk']+1)*config['execute']
                or t.get('success') is not False):
            raise ValueError('Timeout cannot replace a successful or inconsistent trajectory')
    instruction_path = folder/'instruction.json'
    instruction=json.loads(instruction_path.read_text()) if instruction_path.exists() else {}
    if instruction and (any(instruction.get(k)!=scene[k] for k in scene) or not instruction.get('instruction')):
        raise ValueError('Timeout instruction identity mismatch')
    if pred and not instruction:
        raise ValueError('Policy predictions require instruction evidence')
    if mode == UNFILTERED_TIMEOUT_MODE:
        initial_path = folder/'initial-inputs.json'
        if progress.get('event') not in ('policy_started', 'chunk_completed') or not initial_path.exists():
            raise ValueError('A setup/infrastructure interruption cannot become a policy zero')
        initial = json.loads(initial_path.read_text())
        if any(initial.get(k) != scene[k] for k in ('task','setting','seed')):
            raise ValueError('Unfiltered timeout initial input identity differs')
        if initial.get('expert_attempts') != 0 or len(initial.get('initial_state',[])) != 14:
            raise ValueError('Unfiltered timeout lacks its zero-expert 14-joint input receipt')
    expert_path = folder/'expert.json'
    expert = json.loads(expert_path.read_text()) if expert_path.exists() else None
    paired_path = folder/'paired-input-check.json'
    paired = json.loads(paired_path.read_text()) if paired_path.exists() else None
    if mode == PAIRED_TIMEOUT_MODE and pred and (not paired or
            not paired.get('instruction_equal') or not paired.get('initial_state_equal')):
        raise ValueError('Paired policy timeout requires the initial input check')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    frames=_lines(folder/'video-frames.jsonl')
    return dict(**scene,status='policy_timeout',success=False,scored=True,policy_completed=False,
        timeout_policy=mode,timeout_kind=kind,timeout_seconds=limit,
        timeout_phase=progress.get('event'),expert_attempts=int(expert is not None),
        expert=expert or {},paired_input_check=paired,
        seconds=float(elapsed),error=str(error),instruction=instruction.get('instruction'),
        execution_segment=segment,protocol_stage=mode,
        actions=traj[-1]['action_count_after'] if traj else 0,chunks=len(traj),prediction_blocks=len(pred),
        last_observed_action=frames[-1]['action_count'] if frames else None,
        last_observed_action_may_be_incomplete=True,video_complete=False,
        partial_videos={name:str(Path(scene['task'])/scene['setting']/str(scene['seed'])/file)
                        for name,file in [('head_camera','episode.mp4'),('observer_camera','observer.mp4')]
                        if (folder/file).exists()},
        timeout_evidence=dict(files_sha256=files,last_progress=progress,
                              note='Interrupted data; no complete trajectory or terminal frame claim.'))


def write_timeout_result(output, config, records, row):
    from .planning import next_candidate, planned_count
    if not timeout_failure(config,row) or next_candidate(config,records) != {k:row[k] for k in ('task','setting','seed')}:
        raise ValueError('Timeout result must settle the next exact fixed scene')
    folder=Path(output)/row['task']/row['setting']/str(row['seed'])
    with (folder/'episode.json').open('x') as stream:
        json.dump(row,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    with (Path(output)/'episodes.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
    records=[*records,row]
    summary=dict(protocol=config['protocol'],attempted_episodes=len(records),planned_episodes=planned_count(config),
        policy_completed=sum(r['status']=='policy_completed' for r in records),
        timeout_failures=sum(timeout_failure(config,r) for r in records),
        scored=sum(r['status']=='policy_completed' or timeout_failure(config,r) for r in records),
        setup_errors=sum(r['status']=='setup_error' for r in records),policy_errors=0,infrastructure_errors=0,
        successes=sum(r['status']=='policy_completed' and r['success'] for r in records))
    (Path(output)/'summary.json').write_text(json.dumps(summary,indent=2))
    return records
