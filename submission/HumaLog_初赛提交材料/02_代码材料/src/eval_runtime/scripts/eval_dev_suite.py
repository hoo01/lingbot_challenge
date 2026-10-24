"""Development evaluation: fixed scenes and optional official expert screening.

Private scene metadata is used solely to fill the official language templates,
as in each task's play_once(). The policy receives only RGB, joints and language.
Outputs are evaluation records and are never read by the clean training loader.
"""
import argparse
import hashlib
import importlib
import json
import random
import shutil
import sys
import time
import traceback
import faulthandler
from eval_traceback import PeriodicTraceback
from eval_expert import select_policy_entry, ExpertRejected, ExpertExecutionError, EXPERT_FLOW
from eval_platform.expert_outcomes import (OUTER_EXCEPTION_POLICY, EXCEPTION_STAGE,
    acceptable_record, exception_observed, settled_expert_error)
from eval_official_flow import (EXECUTION_FLOW, environment_config as official_environment_config,
    expert_instruction, execute_policy, close_policy_scene, validate_flow_resume)
import signal
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from eval_telemetry import joint_limits_for_json
from eval_protocol_contract import validate_profile
from eval_video import (ActionVideoRecorder, SimulationVideoRecorder, VideoEncoder,
                        DEFAULT_RECORDING_FPS, simulation_video_description, open_video_recording)

ROOT = Path(os.environ.get('LINGBOT_ROOT', Path(__file__).resolve().parents[1])).resolve()
REPO = ROOT / 'third_party/RoboTwin'
sys.path[:0] = [str(REPO), str(REPO / 'description/utils'), str(ROOT / 'third_party/lingbot-vla-v2-official-be969b8f')]
from generate_episode_instructions import generate_episode_descriptions
from deploy.websocket_client_policy import WebsocketClientPolicy
from eval_platform.planning import candidate_seeds, validate_records, planned_count, validate_fixed_scenes
from eval_platform.expert_quota import QUOTA_FLOW, budget, verify_launch_journal
from eval_policy_entry import policy_entry

TASKS = ['adjust_bottle', 'click_bell', 'beat_block_hammer', 'open_microwave', 'place_shoe']
ALL_TASKS = sorted(p.stem for p in (REPO / 'description/task_instruction').glob('*.json'))


def diagnostics(task, env):
    """Read-only simulator telemetry for failure analysis; NEVER policy input."""
    names = {'adjust_bottle': ['bottle'], 'click_bell': ['bell'],
             'beat_block_hammer': ['hammer', 'block'],
             'open_microwave': ['microwave'], 'place_shoe': ['shoe']}.get(task, [])
    if not names:
        names = [name for name, value in vars(env).items()
                 if not name.startswith('_') and callable(getattr(value, 'get_pose', None))]
    result = {'privileged_logging_only': True, 'actors': {}}
    for name in names:
        actor = getattr(env, name)
        pose = actor.get_pose()
        result['actors'][name] = {'position': np.asarray(pose.p).tolist(), 'quaternion': np.asarray(pose.q).tolist()}
    if task == 'open_microwave':
        result.update(door_qpos=np.asarray(env.microwave.get_qpos()).tolist(),
                      door_qlimits=joint_limits_for_json(env.microwave.get_qlimits()))
    elif task == 'adjust_bottle':
        result['bottle_functional_point'] = np.asarray(env.bottle.get_functional_point(0)).tolist()
    elif task == 'beat_block_hammer':
        result['hammer_block_contact'] = bool(env.check_actors_contact(env.hammer.get_name(), env.block.get_name()))
        hammer_target = np.asarray(env.hammer.get_functional_point(0, 'pose').p)
        block_target = np.asarray(env.block.get_functional_point(1, 'pose').p)
        result.update(hammer_target=hammer_target.tolist(), block_target=block_target.tolist(),
                      target_xy_error=np.abs(hammer_target[:2] - block_target[:2]).tolist())
    elif task == 'click_bell':
        result['gripper_bell_contacts'] = np.asarray(env.get_gripper_actor_contact_position('050_bell')).tolist()
    return result


def save_cameras(obs, folder, stem):
    for camera, suffix in [('head_camera', ''), ('left_camera', '-left-wrist'), ('right_camera', '-right-wrist')]:
        Image.fromarray(obs['observation'][camera]['rgb']).save(folder / f'{stem}{suffix}.png')


def scene_language_parameters(task, env):
    # Copied from the info dictionaries in the pinned official task implementations.
    if task == 'adjust_bottle':
        return {'{A}': f'001_bottle/base{env.model_id}', '{a}': 'right' if env.qpose_tag == 1 else 'left'}
    if task == 'click_bell':
        return {'{A}': f'050_bell/base{env.bell_id}', '{a}': 'right' if env.bell.get_pose().p[0] > 0 else 'left'}
    if task == 'beat_block_hammer':
        return {'{A}': '020_hammer/base0', '{a}': 'left' if env.block.get_functional_point(0, 'pose').p[0] < 0 else 'right'}
    if task == 'open_microwave':
        return {'{A}': f'{env.model_name}/base{env.model_id}', '{a}': 'left'}
    if task == 'place_shoe':
        return {'{A}': f'041_shoe/base{env.shoe_id}', '{a}': 'left' if env.shoe.get_pose().p[0] < 0 else 'right'}
    from eval_task_metadata import scene_language_parameters as all_task_parameters
    return all_task_parameters(task, env)


def environment_config(task, setting):
    return official_environment_config(REPO, task, setting)


def summarize(records, config=None):
    config = config or {}
    from eval_platform.timeout_outcomes import timeout_failure
    completed = [r for r in records if r['status'] == 'policy_completed']
    per_setting = {}
    for setting in sorted({r['setting'] for r in records}):
        tasks = {}
        for task in sorted({r['task'] for r in records if r['setting'] == setting}):
            rows = [r for r in records if r['setting'] == setting and r['task'] == task]
            valid = [r for r in rows if r['status'] == 'policy_completed']
            successes = sum(r['success'] for r in valid)
            timeouts = sum(timeout_failure(config,r) for r in rows)
            scored = len(valid)+timeouts
            tasks[task] = {'planned': len(rows), 'completed': len(valid), 'scored':scored,'timeout_failures':timeouts,'successes': successes,
                          'setup_errors': sum(r['status'] == 'setup_error' for r in rows),
                          'execution_errors': sum(not acceptable_record(config,r) for r in rows),
                          'success_rate': successes / scored if scored else None}
        rates = [r['success_rate'] for r in tasks.values() if r['success_rate'] is not None]
        per_setting[setting] = {'tasks': tasks, 'macro_success_rate': float(np.mean(rates)) if rates else None}
    return {'protocol': 'fixed_unfiltered_development_seeds_v3', 'official_benchmark': False,
            'planned_episodes': len(records), 'policy_completed': len(completed),
            'timeout_failures':sum(timeout_failure(config,r) for r in records),
            'scored':len(completed)+sum(timeout_failure(config,r) for r in records),
            'successes': sum(r['success'] for r in completed),
            'setup_errors': sum(r['status'] == 'setup_error' for r in records),
            'infrastructure_errors': sum(r['status'] == 'infrastructure_error' and not settled_expert_error(config,r) for r in records),
            'policy_errors': sum(r['status'] == 'policy_error' for r in records),
            'expert_attempts': sum(r.get('expert_attempts', 0) for r in records),
            'expert_rejections': sum(r['status'] == 'expert_rejected' for r in records),
            'expert_exceptions': sum(exception_observed(config,r) for r in records),
            'settings': per_setting}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=ALL_TASKS, default=TASKS)
    parser.add_argument('--all-tasks', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--no-observer-video', action='store_true',
                        help='Record the head camera only; leave policy cameras and simulation unchanged')
    parser.add_argument('--expert-check', action='store_true',
                        help='Screen each planned seed with the official expert before resetting it for the policy')
    parser.add_argument('--video-fps', type=int, choices=(10, 25), default=DEFAULT_RECORDING_FPS,
                        help='Simulation-time sampling and playback fps, only used with --save-video')
    parser.add_argument('--protocol-id', default='fixed_unfiltered_development_seeds_v3')
    parser.add_argument('--expert-quota', type=json.loads)
    parser.add_argument('--expert-exception-policy', choices=('record_and_stop',OUTER_EXCEPTION_POLICY), default='record_and_stop')
    parser.add_argument('--settled-expert-errors', type=json.loads, default=[])
    parser.add_argument('--expected-execute', type=int, default=50)
    parser.add_argument('--settings', nargs='+', choices=['demo_clean', 'demo_randomized'],
                        default=['demo_clean', 'demo_randomized'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[100010, 100011])
    parser.add_argument('--fixed-scenes', type=json.loads)
    parser.add_argument('--fixed-scenes-file', type=Path)
    parser.add_argument('--fixed-scenes-sha256')
    parser.add_argument('--paired-reference', type=json.loads)
    parser.add_argument('--paired-camera-policy', type=json.loads)
    parser.add_argument('--scene-replay', type=json.loads,
                        help='Bound E0 clutter/setup and instruction RNG receipt; no expert calls')
    parser.add_argument('--paired-entry', choices=('frozen_e0_no_recheck_v1',))
    parser.add_argument('--timeout-policy', type=json.loads)
    parser.add_argument('--valid-per-group', type=int)
    parser.add_argument('--max-consecutive-nonvalid', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-checkpoint', type=Path, required=True)
    parser.add_argument('--expected-precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--expected-vision-device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--expected-embedding-device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--expected-moe-backend', choices=['official', 'torch_reference'], default='official')
    parser.add_argument('--port', type=int, default=18006)
    parser.add_argument('--inference-profile', choices=['paired_legacy','aug16000_h50_execute25_no_video_v1','research4_paired_v1'], default='paired_legacy')
    parser.add_argument('--resume-completed', action='store_true')
    parser.add_argument('--one-episode', action='store_true',
                        help='Exit after one remaining episode so the supervisor can release simulator resources')
    args = parser.parse_args()
    if args.fixed_scenes_file is not None:
        if args.fixed_scenes is not None:raise ValueError('Conflicting fixed scene input mechanisms')
        from eval_platform.fork2000_plan import read_scene_file
        args.fixed_scenes=read_scene_file(args.fixed_scenes_file,args.fixed_scenes_sha256,ROOT)
    elif args.fixed_scenes_sha256 is not None:
        raise ValueError('A scene SHA requires its fixed scene input file')
    if args.all_tasks:
        args.tasks = ALL_TASKS
        assert len(args.tasks) == 50
    assert len(args.seeds) == len(set(args.seeds)), 'Duplicate scene seeds'
    expected_episodes = len(args.tasks) * len(args.settings) * (args.valid_per_group or len(args.seeds))
    if args.fixed_scenes is not None:
        validate_fixed_scenes(vars(args))
        expected_episodes = len(args.fixed_scenes)
    from eval_platform.fork2000_plan import active, TARGET
    if active(vars(args)):expected_episodes=len(args.tasks)*len(args.settings)*TARGET
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_completed:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    source = args.output / (f'source-resume-{stamp}' if args.resume_completed else 'source')
    source.mkdir()
    for filename in ('eval_dev_suite.py', 'eval_isolated.py', 'eval_expert.py', 'eval_official_flow.py', 'eval_policy_entry.py', 'eval_policy_entry_sources.json', 'eval_telemetry.py', 'eval_task_metadata.py', 'eval_protocol_contract.py', 'eval_video.py', 'eval_traceback.py', 'serve_official_checkpoint.py', 'team_eval.py'):
        shutil.copy2(Path(__file__).parent / filename, source / filename)
    shutil.copytree(Path(sys.modules[candidate_seeds.__module__].__file__).parent, source / 'eval_platform',
                    ignore=shutil.ignore_patterns('__pycache__'))
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)
    torch.set_num_threads(4)
    render_device = os.environ.get('LINGBOT_RENDER_DEVICE')
    if render_device:
        import sapien
        def create_scene_on_assigned_gpu(engine, config=None):
            if config is not None:
                sapien.physx.set_scene_config(config)
            system=sapien.render.RenderSystem(render_device)
            device=system.device
            prop=lambda n:getattr(device,n)() if callable(getattr(device,n)) else getattr(device,n)
            scope=json.loads((ROOT/'artifacts/eval2000-scope.json').read_text())
            actual=prop('pci_string')
            normalize=lambda bus:':'.join(str(int(v,16)) for v in bus.lower().split(':')[:-1])+':'+bus.lower().split(':')[-1]
            assert normalize(actual)==normalize(scope['physical_pci_bus_id']) and prop('can_render'), 'Renderer physical GPU differs from model UUID'
            (ROOT/'artifacts'/('render-binding-'+str(os.getpid())+'.json')).write_text(json.dumps(dict(passed=True,physical_gpu_uuid=scope['physical_gpu'],actual_pci_bus_id=actual,cuda_id=prop('cuda_id'),name=prop('name'))))
            return sapien.Scene([sapien.physx.PhysxCpuSystem(),system])
        sapien.Engine.create_scene = create_scene_on_assigned_gpu
    policy = WebsocketClientPolicy(host='127.0.0.1', port=args.port)
    metadata = policy.get_server_metadata()
    validate_profile(metadata, args.inference_profile)
    assert metadata.get('execute') == args.expected_execute, metadata
    assert metadata.get('dtype') == args.expected_precision, metadata
    assert metadata.get('vision_device') == args.expected_vision_device, metadata
    assert metadata.get('embedding_device', 'cuda') == args.expected_embedding_device, metadata
    assert metadata.get('moe_backend') == args.expected_moe_backend, metadata
    assert Path(metadata['checkpoint']).resolve() == args.expected_checkpoint.resolve(), metadata
    protocol = {'protocol': args.protocol_id, 'official_benchmark': False,
                'tasks': args.tasks, 'seeds': args.seeds, 'settings': args.settings,
                'instruction_types': {'demo_clean': 'seen', 'demo_randomized': 'unseen'},
                'expert_check': args.expert_check,
                'execution_flow': EXECUTION_FLOW,'execute':args.expected_execute,'prediction_horizon':50,
                'execution_protocol':'H50_first25_reobserve_v1',
                'expert_flow': EXPERT_FLOW if args.expert_check else None,
                'expert_exception_policy': args.expert_exception_policy,
                'settled_expert_errors': args.settled_expert_errors,
                'language_rng': 'python_per_scene_seed_isolated_numpy_from_setup',
                'instruction_template_count': (args.valid_per_group or len(args.seeds)) if args.expert_check else 1,
                'scene_sampling': 'fixed_planned_seeds_no_replacement', 'server': metadata,
                'video': simulation_video_description(args.video_fps, not args.no_observer_video) if args.save_video else False,
                'recording': {'enabled': args.save_video, 'fps': args.video_fps, 'time_axis': 'simulation time',
                              'cameras': (['head_camera'] if args.no_observer_video else ['head_camera', 'observer_camera']),
                              'observer_video': not args.no_observer_video, 'shared_clock': True},
                'render_device': render_device or 'upstream_default',
                'renderer': json.loads((ROOT / 'artifacts/oidn-blackwell.json').read_text()),
                'execution': {'simulator_process': 'one_episode' if args.one_episode else 'shared'},
                'failure_images': 'All three cameras before each policy request and at end; no extra simulation steps.',
                'diagnostics': 'Read-only privileged actor poses/contact/door position; never passed to policy or training.'}
    if args.fixed_scenes is not None:
        protocol.update(fixed_scenes=args.fixed_scenes, scene_sampling='explicit_scene_list_no_replacement')
    if args.timeout_policy is not None:
        protocol['timeout_policy'] = args.timeout_policy
        if active(protocol):protocol['policy_target_per_group']=TARGET
    if args.valid_per_group is not None:
        protocol.update(valid_per_group=args.valid_per_group,
                        max_consecutive_nonvalid=args.max_consecutive_nonvalid,
                        scene_sampling='ascending_seeds_until_valid_target',
                        instruction_template_count=(args.valid_per_group or len(args.seeds)) if args.expert_check else 1)
    if args.expert_quota:
        if not args.expert_check or not args.valid_per_group:
            raise ValueError('Expert quota requires expert screening and a valid target')
        protocol['expert_quota'] = args.expert_quota
        protocol['fallback_policy_entry'] = 'pinned_object_templates_and_score_prelude_v1'
    if args.expert_exception_policy == OUTER_EXCEPTION_POLICY and not args.expert_quota:
        raise ValueError('Counted expert exceptions require the approved finite quota')
    paired = {}
    if args.paired_entry:
        protocol['paired_entry'] = args.paired_entry
    if args.paired_reference:
        from eval_platform.paired_inputs import read_reference, validate_entry
        if not args.expert_check and not validate_entry(protocol | {'paired_reference':args.paired_reference}):
            raise ValueError('Paired entry requires expert flow')
        paired = read_reference(args.paired_reference,args.fixed_scenes)
        protocol['paired_reference'] = args.paired_reference
        protocol['instruction_template_count'] = 5
        protocol['scene_sampling'] = 'e0_final_fixed_keys_no_replacement_original_instructions'
    from eval_platform.paired_inputs import validate_entry
    validate_entry(protocol)
    scene_replay = {}
    if args.scene_replay:
        if not args.paired_entry or args.expert_check or not args.one_episode:
            raise ValueError('Scene replay requires isolated frozen E0 entry with experts off')
        from eval_platform.scene_replay import read_scene_replay
        scene_replay = read_scene_replay(args.scene_replay, args.paired_reference, args.fixed_scenes, REPO)
        protocol['scene_replay'] = args.scene_replay
        protocol['language_rng'] = 'frozen_text_replay_E0_numpy_choice5_v1'
    camera_policy = None
    if args.paired_camera_policy:
        from eval_platform.camera_pairing import read_camera_policy
        camera_policy = read_camera_policy(args.paired_camera_policy,root=ROOT,checkpoint=args.expected_checkpoint,
            norm_sha256=protocol['server']['normalization_sha256'],paired_reference=args.paired_reference,scene_replay=args.scene_replay)
        protocol['paired_camera_policy'] = args.paired_camera_policy
    from eval_platform.timeout_outcomes import validate_timeout_policy
    validate_timeout_policy(protocol)
    records = []
    if args.resume_completed:
        previous = json.loads((args.output / 'protocol.json').read_text())
        validate_flow_resume(previous, protocol)
        assert previous.get('expert_check', False) == args.expert_check, 'Expert screening differs'
        for key in ('protocol', 'tasks', 'seeds', 'settings', 'instruction_types', 'server', 'renderer'):
            assert previous[key] == protocol[key], f'Resumed protocol differs: {key}'
        for key in ('valid_per_group', 'fixed_scenes', 'timeout_policy', 'max_consecutive_nonvalid', 'scene_sampling', 'instruction_template_count', 'expert_quota', 'fallback_policy_entry', 'settled_expert_errors', 'paired_reference', 'paired_entry', 'scene_replay', 'paired_camera_policy'):
            assert previous.get(key) == protocol.get(key), f'Resumed sampling differs: {key}'
        records = [json.loads(line) for line in (args.output / 'episodes.jsonl').read_text().splitlines()]
        from eval_platform.expert_outcomes import acceptable_record
        assert all(acceptable_record(protocol, r) for r in records), 'Preserve and resolve recorded errors before starting a new attempt'
        (args.output / f'resume-protocol-{stamp}.json').write_text(json.dumps(protocol, indent=2))
    else:
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    completed = {(r['task'], r['setting'], r['seed']) for r in records}
    assert len(completed) == len(records)
    validate_records(protocol, records)
    launch_journal = args.output / 'expert-launches.jsonl'
    if args.expert_quota:
        verify_launch_journal(launch_journal, records)

    def progress(event, task, setting, seed, **extra):
        payload = {'event': event, 'time': datetime.now(timezone.utc).isoformat(),
                   'task': task, 'setting': setting, 'seed': seed, **extra}
        temporary = args.output / 'worker-progress.tmp'
        temporary.write_text(json.dumps(payload))
        temporary.replace(args.output / 'worker-progress.json')

    for task in args.tasks:
        task_type = getattr(importlib.import_module(f'envs.{task}'), task)
        for setting in args.settings:
            config = environment_config(task, setting)
            for episode, seed in candidate_seeds(protocol, records, task, setting):
                # Upstream now_id counts policy episodes, not rejected candidates.
                episode = sum(r['task'] == task and r['setting'] == setting and
                              r['status'] == 'policy_completed' for r in records)
                # This new unfiltered experiment binds index to the fixed seed ordinal,
                # independent of earlier successes/timeouts; it is not historical E0 replay.
                if args.timeout_policy and args.timeout_policy.get('mode') == 'unfiltered_wall900_stall240_zero_continue_v1':
                    episode = args.seeds.index(seed)
                reference = paired.get((task,setting,seed))
                if reference:
                    episode = reference['policy_episode_index']
                if (task, setting, seed) in completed:
                    continue
                folder = args.output / task / setting / str(seed)
                if folder.exists():
                    raise FileExistsError(f'Unrecorded output must be preserved and recovered through an explicit manifest: {folder}')
                folder.mkdir(parents=True)
                env = task_type()
                row = {'task': task, 'setting': setting, 'seed': seed, 'success': False,
                       'expert_attempts': 0, 'sampling_seed': metadata.get('sampling_seed_per_episode'),
                       'sampling_seed_at_process_start': metadata.get('sampling_seed_at_process_start'),
                       'inference_profile': args.inference_profile}
                row['requested_seed'] = 0
                row['actual_environment_seed'] = seed
                row['model_revision'] = '2b1e8a52668d38dbc91a01bf0d83085ba7a8f8a1'
                row['frozen_protocol_sha256'] = '8508ec80bd206a42274d62a5bbe9331ecfe343fa4a31e1b64f302412f4199dae'
                row['execution_segment'] = os.environ.get('LINGBOT_EVAL_SEGMENT', 'original')
                row['execution_flow'] = EXECUTION_FLOW
                if args.timeout_policy:
                    row['protocol_stage'] = args.timeout_policy['mode']
                row['policy_episode_index'] = episode
                if args.timeout_policy and args.timeout_policy.get('mode') == 'unfiltered_wall900_stall240_zero_continue_v1':
                    row['index_source'] = 'fixed_request_seed_ordinal'
                    row['expert'] = dict(phase='not_launched',attempts=0,passed=False,
                                         reason='fork_fixed500_unfiltered_experts_disabled')
                row['language_seed'] = seed
                row['instruction_template_count'] = protocol['instruction_template_count']
                row['video_requested'] = args.save_video
                row['requested_video_fps'] = args.video_fps
                row['observer_video_requested'] = args.save_video and not args.no_observer_video
                if args.expert_quota:
                    row.update(quota_flow=QUOTA_FLOW,
                               expert_budget=budget(protocol, records, task, setting))
                row['expert_exception_policy'] = args.expert_exception_policy
                if args.expert_exception_policy == OUTER_EXCEPTION_POLICY:
                    row['protocol_stage'] = EXCEPTION_STAGE
                row['simulator_openblas_threads'] = os.environ.get('OPENBLAS_NUM_THREADS')
                phase, started = 'setup', time.monotonic()
                videos, extra_views, recorder = {}, {}, None
                policy_setup_started = False
                tracebacks = PeriodicTraceback(120).start()
                progress('setup_started', task, setting, seed)
                print(json.dumps({'event': 'setup_started', 'task': task, 'setting': setting, 'seed': seed}), flush=True)
                try:
                    forced_policy = False
                    frozen_entry = bool(reference and args.paired_entry)
                    if frozen_entry:
                        row['policy_entry_reason'] = 'frozen_e0_expert_provenance_no_recheck'
                        row['expert'] = dict(phase='not_launched',passed=False,attempts=0,
                            reason='reuse_frozen_E0_expert_provenance',
                            reference_passed=reference['expert_passed'],
                            reference_budget_exhausted=reference['expert_budget_exhausted'])
                        (folder/'expert.json').write_text(json.dumps(row['expert'],indent=2))
                    elif reference and reference['expert_budget_exhausted']:
                        forced_policy = True
                        row['policy_entry_reason'] = 'e0_reference_expert_budget_exhausted'
                        row['expert'] = dict(phase='not_launched',passed=False,attempts=0,
                            reason='preserved_E0_exhausted_entry_no_new_expert_budget')
                        (folder/'expert.json').write_text(json.dumps(row['expert'],indent=2))
                    elif args.expert_check:
                        phase = 'expert'
                        forced_policy, parameters = select_policy_entry(env, config, episode, seed, folder, row,
                            lambda event: progress(event, task, setting, seed),
                            budget=row.get('expert_budget'), journal=launch_journal,
                            exception_policy=args.expert_exception_policy)
                    if args.expert_quota:
                        row['expert_budget']['used_after'] = row['expert_budget']['used_before'] + row['expert_attempts']
                        row['policy_entry_reason'] = 'expert_budget_exhausted' if forced_policy else 'expert_passed'
                        if forced_policy and row['expert_attempts'] == 0:
                            row['expert'] = dict(phase='not_launched', passed=False, attempts=0,
                                                 reason='inherited_or_accumulated_budget_exhausted')
                            (folder / 'expert.json').write_text(json.dumps(row['expert'], indent=2))
                    phase = 'setup'
                    policy_setup_started = True
                    if scene_replay:
                        from envs._base_task import Base_Task
                        from eval_platform.scene_replay import install_clutter_copy
                        replay = scene_replay[(task, setting, seed)]
                        row['scene_replay'] = install_clutter_copy(Base_Task, replay['prior_clutter_setups'])
                        row['scene_replay']['e0_episode_record'] = replay['e0_episode_record']
                        from eval_platform.asset_enumeration import install_for_scene
                        row['asset_enumeration'] = install_for_scene(REPO,task,args.output,args.paired_reference)
                    env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **config)
                    if frozen_entry:
                        from eval_platform.paired_inputs import prepare_policy_entry
                        parameters, row['policy_entry'] = prepare_policy_entry(task, env, REPO, reference)
                    elif forced_policy:
                        parameters, row['policy_entry'] = policy_entry(task, env, REPO)
                    if frozen_entry:
                        instruction = reference['instruction']
                        if scene_replay:
                            from eval_platform.scene_replay import replay_instruction_choice
                            row['scene_replay'].update(replay_instruction_choice(replay['instruction_choice_count']))
                            (folder/'scene-replay.json').write_text(json.dumps(row['scene_replay'], indent=2))
                    elif args.expert_check:
                        instruction = expert_instruction(task, parameters, setting,
                            protocol['instruction_template_count'], generate_episode_descriptions, language_seed=seed)
                    else:
                        parameters = scene_language_parameters(task, env)
                        random.seed(seed)
                        descriptions = generate_episode_descriptions(task, [parameters], 1)
                        instruction = descriptions[0]['seen' if setting == 'demo_clean' else 'unseen'][0]
                    if reference:
                        if parameters != reference['parameters']:
                            raise ValueError('Official scene parameters differ from E0; preserve seed and stop')
                        row['generated_instruction'] = instruction
                        instruction = reference['instruction']
                        row['instruction_source'] = 'frozen_E0_final_episode'
                    env.set_instruction(instruction=instruction)
                    # Keep upstream cached-frame recording disabled in both modes.
                    env.eval_video_path = None
                    row['instruction'] = instruction
                    (folder / 'instruction.json').write_text(json.dumps({
                        'instruction': instruction, 'parameters': parameters,
                        'task': task, 'setting': setting, 'seed': seed}))
                    if args.save_video:
                        videos, extra_views, recorder = open_video_recording(env, folder,
                            config['head_camera_w'], config['head_camera_h'], args.video_fps,
                            observer_video=not args.no_observer_video)
                    phase = 'policy'
                    progress('policy_started', task, setting, seed)
                    print(json.dumps({'event': 'policy_started', 'task': task, 'setting': setting, 'seed': seed}), flush=True)
                    def record_observation(obs, chunks):
                        if chunks == 0 and reference:
                            from eval_platform.paired_inputs import compare_initial
                            row['paired_input_check'] = compare_initial(reference,obs,instruction)
                            # Save the actual observation without an extra camera capture.
                            save_cameras(obs,folder,'chunk-000')
                            if scene_replay:
                                from eval_platform.camera_pairing import enforce_camera_policy
                                enforce_camera_policy(row['paired_input_check'],camera_policy)
                            (folder/'paired-input-check.json').write_text(json.dumps(row['paired_input_check'],indent=2))
                        if chunks == 0 and os.environ.get('LINGBOT_RESUME_MANIFEST'):
                            from eval_platform.resume import verify_recovered_scene
                            row['recovery_scene_verified'] = verify_recovered_scene(
                                os.environ['LINGBOT_RESUME_MANIFEST'], task, setting, seed,
                                obs, instruction, parameters)
                        head = obs['observation']['head_camera']['rgb']
                        if recorder is not None and recorder.frames == 0:
                            recorder.write_initial(head, env.take_action_cnt)
                        if chunks==0:save_cameras(obs, folder, 'chunk-000')
                        if chunks == 0 and args.timeout_policy and args.timeout_policy.get('mode') == 'unfiltered_wall900_stall240_zero_continue_v1':
                            initial_state = np.asarray(obs['joint_action']['vector'])
                            if initial_state.shape != (14,) or not np.isfinite(initial_state).all():
                                raise ValueError('Invalid initial physical joint vector')
                            camera_rows = {}
                            for camera in ('head_camera','left_camera','right_camera'):
                                pixels = np.asarray(obs['observation'][camera]['rgb'])
                                if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                                    raise ValueError('The policy RGB boundary must be uint8 HWC RGB')
                                camera_rows[camera] = dict(rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                                    dtype=str(pixels.dtype),shape=list(pixels.shape),minimum=int(pixels.min()),maximum=int(pixels.max()))
                            initial = dict(task=task,setting=setting,seed=seed,policy_episode_index=episode,
                                instruction=instruction,parameters=parameters,initial_state=initial_state.tolist(),
                                cameras=camera_rows,expert_attempts=0,extra_policy_observations=0,
                                extra_physics_steps=0,image_augmentation=False,e0_pixel_identity_claimed=False,execute=25,prediction_horizon=50)
                            (folder/'initial-inputs.json').write_text(json.dumps(initial,indent=2,allow_nan=False))

                    def record_prediction(inputs, actions, chunks):
                        trace = {'chunk': chunks, 'action_count_before': env.take_action_cnt,
                                 'state_before': np.asarray(inputs['observation.state']).tolist(),
                                 'predicted_actions': actions.tolist(), 'diagnostics_before': diagnostics(task, env)}
                        # Preserve the currently executing chunk even if native
                        # simulator code crashes before its completion record.
                        with (folder / 'predictions.jsonl').open('a') as file:
                            file.write(json.dumps(trace, allow_nan=False) + '\n')
                            file.flush()
                            os.fsync(file.fileno())
                        return trace

                    def record_chunk(trace, chunks):
                        trace.update(action_count_after=env.take_action_cnt, success=bool(env.eval_success),
                                     diagnostics_after=diagnostics(task, env))
                        with (folder / 'trajectory.jsonl').open('a') as file:
                            file.write(json.dumps(trace, allow_nan=False) + '\n')
                        progress('chunk_completed', task, setting, seed, chunks=chunks + 1,
                                 actions=env.take_action_cnt)

                    outcome = execute_policy(env, policy, args.expected_execute,
                        record_observation, record_prediction, record_chunk)
                    final_obs = env.get_obs()
                    save_cameras(final_obs, folder, 'final')
                    if recorder is not None:
                        recorder.finish(final_obs['observation']['head_camera']['rgb'], env.take_action_cnt)
                        if 'observer_camera' in recorder.terminal_views:
                            Image.fromarray(recorder.terminal_views['observer_camera']).save(folder / 'final-observer.png')
                    row.update(outcome)
                except Exception as error:
                    setup_phase = phase == 'setup' or (phase == 'expert' and row['expert']['phase'] == 'setup')
                    status = ('expert_rejected' if isinstance(error, ExpertRejected) else
                        'expert_exception' if isinstance(error, ExpertExecutionError) else
                        'setup_error' if setup_phase and type(error).__name__ == 'UnStableError' else
                        'policy_error' if phase == 'policy' else 'infrastructure_error')
                    if row.get('expert', {}).get('cleanup_error'):
                        status = 'infrastructure_error'
                    row.update(status=status, error=traceback.format_exc(), native_initialization_error=(status=='setup_error'), policy_entered=(phase=='policy'), error_type=type(error).__name__)
                finally:
                    tracebacks.close()
                    if recorder is not None:
                        row['video_capture'] = recorder.metadata()
                        recorder.close()
                    for view in extra_views.values():
                        view.close()
                    for camera_name, video in videos.items():
                        try:
                            video.finish()
                            path = video.path.relative_to(args.output).as_posix()
                            if camera_name == 'head_camera':
                                row['video'] = path
                            row.setdefault('videos', {})[camera_name] = {
                                'path': path, 'capture': recorder.metadata(camera_name) if recorder else None}
                        except Exception:
                            row.update(status='infrastructure_error', video_error=traceback.format_exc())
                    env.eval_video_path = None
                    if policy_setup_started:
                        close_policy_scene(env, row)
                row['seconds'] = time.monotonic() - started
                if args.expert_quota:
                    row['expert_budget']['used_after'] = row['expert_budget']['used_before'] + row['expert_attempts']
                records.append(row)
                (folder / 'episode.json').write_text(json.dumps(row, indent=2))
                with (args.output / 'episodes.jsonl').open('a') as file:
                    file.write(json.dumps(row) + '\n')
                    file.flush()
                    os.fsync(file.fileno())
                print(json.dumps(row), flush=True)
                progress('episode_recorded', task, setting, seed, status=row['status'])
                if reference and row['status'] != 'policy_completed':
                    raise RuntimeError('Fixed E0 scene did not complete; no retry budget or replacement seed')
                if row['status'] in ('policy_error', 'infrastructure_error'):
                    report = {**summarize(records, protocol), 'protocol':args.protocol_id, 'attempted_episodes': len(records), 'planned_episodes': expected_episodes}
                    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
                    raise SystemExit(2)
                if args.one_episode:
                    report = {**summarize(records, protocol), 'protocol':args.protocol_id, 'attempted_episodes': len(records),
                              'planned_episodes': expected_episodes}
                    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
                    return
    report = {**summarize(records, protocol), 'protocol':args.protocol_id, 'attempted_episodes': len(records), 'planned_episodes': expected_episodes}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if report['setup_errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
