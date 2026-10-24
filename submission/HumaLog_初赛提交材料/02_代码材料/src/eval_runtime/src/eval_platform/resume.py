"""Explicit, hash-bound recovery segments for an existing evaluation job."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def inside(folder, relative):
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder) or path == folder:
        raise ValueError('Recovery path must stay inside its job folder')
    return path


def load_resume(folder):
    pointer = Path(folder) / 'active-resume.json'
    if not pointer.exists():
        return None
    ref = json.loads(pointer.read_text())
    manifest_path = inside(folder, ref['manifest'])
    if sha256(manifest_path) != ref['sha256']:
        raise ValueError('Recovery manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('version') != 1 or manifest.get('job_id') != Path(folder).name.removeprefix('eval-'):
        raise ValueError('Recovery manifest does not identify this job')
    return manifest


def shard_folder(folder, number):
    manifest = load_resume(folder)
    if manifest is None:
        return Path(folder) / 'shards' / str(number)
    matches = [s for s in manifest['shards'] if s['id'] == number]
    if len(matches) != 1:
        raise ValueError('Recovery manifest has no unique shard mapping')
    return inside(folder, matches[0]['output'])


def verify_resume(root, folder, config, number):
    """Check the diagnosis, unchanged source records, and seeded prefix before launch."""
    manifest = load_resume(folder)
    if manifest is None:
        return None
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest['config_sha256']:
        raise ValueError('Recovery configuration differs from the fixed request')
    for item in manifest['required_files']:
        path = inside(root, item['path'])
        if sha256(path) != item['sha256']:
            raise ValueError('Recovery prerequisite changed: ' + item['path'])
    extension = manifest.get('purpose') == 'valid_target_extension'
    receipt_key = 'preparation_receipt' if extension else 'diagnosis'
    diagnostic = json.loads(inside(root, manifest[receipt_key]).read_text())
    if diagnostic.get('passed') is not True or diagnostic.get('scored') is not False:
        raise ValueError('Launch requires a passed, unscored preparation or diagnosis')
    shard = next(s for s in manifest['shards'] if s['id'] == number)
    output = shard_folder(folder, number)
    records_path = output / 'evaluation/episodes.jsonl'
    if sha256(records_path) != shard['seeded_records_sha256']:
        raise ValueError('Recovery completed prefix changed before launch')
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    from .expert_outcomes import acceptable_record
    if any(not acceptable_record(config,r) for r in records):
        raise ValueError('Recovery prefix contains unresolved execution errors')
    assigned = next(s for s in config['shards'] if s['id'] == number)
    plan = [(t, s, seed) for t in assigned['tasks'] for s in assigned['settings'] for seed in config['seeds']]
    keys = [(r['task'], r['setting'], r['seed']) for r in records]
    if manifest.get('purpose') == 'fork_scored_target_continuation':
        from .fork2000_resume import verify
        return verify(root,folder,config,manifest,diagnostic,records,shard)
    if manifest.get('purpose') == 'paired_host_migration_transition':
        from .host_migration import verify_transition
        return verify_transition(root,folder,config,manifest,diagnostic,records,shard)
    if manifest.get('purpose') in ('paired_fixed_resume', 'paired_reference_entry_transition'):
        from .timeout_outcomes import PAIRED_TIMEOUT_MODE, validate_timeout_policy
        from .planning import next_candidate, validate_records, group_seeds
        from .paired_inputs import FROZEN_ENTRY, validate_entry
        validate_timeout_policy(config)
        prior = json.loads(inside(root, manifest['source_config']).read_text())
        parent = manifest['parent_resume']
        parent_path = inside(folder, parent['manifest'])
        transition = manifest['purpose'] == 'paired_reference_entry_transition'
        expected = ({**prior, 'protocol':'e0_fixed500_frozen_expert_policy_only_v3',
                     'expert_check':False, 'paired_entry':FROZEN_ENTRY} if transition else prior)
        prior_hash = hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if (sha256(parent_path) != parent['sha256'] or config != expected
                or json.loads(parent_path.read_text())['config_sha256'] != prior_hash
                or config['timeout_policy']['mode'] != PAIRED_TIMEOUT_MODE
                or not diagnostic.get('user_authorization') or diagnostic.get('job_id') != manifest['job_id']):
            raise ValueError('Paired continuation must keep the authorized configuration and parent')
        if transition and (prior.get('paired_entry') or not validate_entry(config)):
            raise ValueError('Frozen entry transition must only remove repeat expert screening')
        source = inside(root, shard['source_records'])
        excluded = shard.get('excluded_expert_rechecks', [])
        if excluded and not transition:
            raise ValueError('Normal paired continuation cannot omit any recorded outcome')
        retained, matched = [], []
        for line in source.read_bytes().splitlines(keepends=True):
            row = json.loads(line)
            item = dict(task=row['task'], setting=row['setting'], seed=row['seed'],
                        original_line_sha256=hashlib.sha256(line).hexdigest())
            if item in excluded:
                expert = row.get('expert', {})
                if (item in matched or row['status'] != 'expert_rejected' or row.get('expert_attempts') != 1
                        or expert.get('phase') != 'completed' or expert.get('passed') is not False
                        or expert.get('cleanup_error') or expert.get('error_type')):
                    raise ValueError('Only a normal archived expert recheck rejection can return to policy evaluation')
                history = [h for h in manifest['historical_interruptions']
                           if h.get('kind') == 'expert_recheck_rejected'
                           and all(h.get(k) == item[k] for k in item)]
                if len(history) != 1 or not history[0].get('counts_as_attempt') or history[0].get('counted_in_candidate_records'):
                    raise ValueError('Archived expert recheck must be disclosed separately from its policy attempt')
                matched.append(item)
            else:
                retained.append(line)
        if len(matched) != len(excluded) or b''.join(retained) != records_path.read_bytes():
            raise ValueError('Paired continuation must preserve every recorded outcome byte for byte')
        current = {**config, 'tasks': assigned['tasks'], 'settings': assigned['settings'],
                   'fixed_scenes': [s for s in config['fixed_scenes']
                                    if s['task'] in assigned['tasks'] and s['setting'] in assigned['settings']]}
        validate_records(current, records)
        sparse = [(t, s, seed) for t in assigned['tasks'] for s in assigned['settings']
                  for seed in group_seeds(current, t, s)]
        if keys != sparse[:len(keys)] or shard['first_remaining'] != next_candidate(current, records):
            raise ValueError('Paired continuation must resume the exact remaining sparse plan')
        return manifest
    if manifest.get('purpose') == 'paired_timeout_transition':
        from .timeout_outcomes import PAIRED_TIMEOUT_MODE, validate_timeout_policy, timeout_failure
        from .planning import next_candidate, validate_records
        validate_timeout_policy(config)
        prior = json.loads(inside(root, manifest['source_config']).read_text())
        expected = {**prior, 'protocol': 'e0_fixed500_paired_inputs_timeouts_v2',
                    'timeout_policy': config['timeout_policy']}
        if (config != expected or prior.get('timeout_policy') or not prior.get('paired_reference')
                or config['timeout_policy']['mode'] != PAIRED_TIMEOUT_MODE
                or not diagnostic.get('user_authorization')):
            raise ValueError('Paired recovery may only add the authorized timeout outcome policy')
        source = inside(root, shard['source_records'])
        original, seeded = source.read_bytes(), records_path.read_bytes()
        if not seeded.startswith(original):
            raise ValueError('Paired recovery must preserve every original result byte for byte')
        extra = seeded[len(original):].splitlines(keepends=True)
        settlements = shard.get('settled_timeout_rows', [])
        if len(extra) != len(settlements):
            raise ValueError('Unexpected paired timeout settlement rows')
        for line, item in zip(extra, settlements):
            row = json.loads(line)
            if (not timeout_failure(config, row) or hashlib.sha256(line).hexdigest() != item['line_sha256']
                    or any(row[k] != item[k] for k in ('task', 'setting', 'seed'))):
                raise ValueError('Paired timeout settlement changed')
            events = inside(root, item['source_events'])
            event = json.loads(item['original_event'])
            prefix = ("TimeoutError('No completed action chunk for 240s;" if row['timeout_kind'] == 'action_chunk_stall'
                      else "TimeoutError('Episode exceeded 900s;")
            if (sha256(events) != item['events_sha256']
                    or event not in [json.loads(x) for x in events.read_text().splitlines()]
                    or event.get('event') != 'infrastructure_interruption'
                    or not event.get('error', '').startswith(prefix)
                    or any(event.get('last_progress', {}).get(k) != row[k] for k in ('task', 'setting', 'seed'))):
                raise ValueError('Paired settlement requires its original timeout event')
            for name, digest in row['timeout_evidence']['files_sha256'].items():
                evidence = inside(root, item['source_episode'] + '/' + name)
                if sha256(evidence) != digest:
                    raise ValueError('Original timeout evidence changed')
            history = [h for h in manifest['historical_interruptions']
                       if h.get('kind') in ('episode_wall_timeout', 'episode_stall_timeout')
                       and all(h.get(k) == row[k] for k in ('task', 'setting', 'seed'))]
            if (len(history) != 1 or history[0].get('counts_as_attempt') is not False
                    or not history[0].get('counted_in_candidate_records')):
                raise ValueError('Paired timeout must remain disclosed without double counting')
        current = {**config, 'tasks': assigned['tasks'], 'settings': assigned['settings'],
                   'fixed_scenes': [s for s in config['fixed_scenes']
                                    if s['task'] in assigned['tasks'] and s['setting'] in assigned['settings']]}
        validate_records(current, records)
        plan = [(t, s, seed) for t in assigned['tasks'] for s in assigned['settings']
                for seed in [r['seed'] for r in current['fixed_scenes'] if r['task'] == t and r['setting'] == s]]
        if keys != plan[:len(keys)] or shard['first_remaining'] != next_candidate(current, records):
            raise ValueError('Paired recovery must continue the exact remaining sparse plan')
        return manifest
    if manifest.get('purpose') == 'fixed_timeout_transition':
        from .timeout_outcomes import validate_timeout_policy, timeout_failure
        from .planning import next_candidate, validate_records
        validate_timeout_policy(config)
        prior = json.loads(inside(root,manifest['source_config']).read_text())
        expected = {**prior,'protocol':'user_base_wall900_zero_continue_v1','timeout_policy':config['timeout_policy']}
        if config != expected or prior.get('timeout_policy') or not diagnostic.get('user_authorization'):
            raise ValueError('Timeout transition may only add the authorized zero-score wall-time policy')
        source = inside(root,shard['source_records'])
        original = source.read_bytes()
        seeded = records_path.read_bytes()
        if not seeded.startswith(original):
            raise ValueError('Timeout transition must preserve every original row byte for byte')
        extra = seeded[len(original):].splitlines(keepends=True)
        settlements = shard.get('settled_timeout_rows',[])
        if len(extra) != len(settlements):
            raise ValueError('Unexpected rows appended during timeout settlement')
        for line, item in zip(extra,settlements):
            row=json.loads(line)
            if (not timeout_failure(config,row) or hashlib.sha256(line).hexdigest()!=item['line_sha256']
                    or any(row[k]!=item[k] for k in ('task','setting','seed'))):
                raise ValueError('Timeout settlement does not match the approved evidence')
            events=inside(root,item['source_events'])
            if sha256(events)!=item['events_sha256']:
                raise ValueError('Original timeout event changed')
            event=json.loads(item['original_event'])
            if (event not in [json.loads(x) for x in events.read_text().splitlines()]
                    or event.get('event')!='infrastructure_interruption'
                    or not event.get('error','').startswith("TimeoutError('Episode exceeded 900s;")
                    or any(event.get('last_progress',{}).get(k)!=row[k] for k in ('task','setting','seed'))):
                raise ValueError('Settlement requires the original 900-second timeout event')
            history=[h for h in manifest['historical_interruptions'] if h.get('kind')=='episode_wall_timeout'
                     and all(h.get(k)==row[k] for k in ('task','setting','seed'))]
            if len(history)!=1 or history[0].get('counts_as_attempt') is not False or not history[0].get('counted_in_candidate_records'):
                raise ValueError('Settled timeout must remain in history without double counting')
        current={**config,'tasks':assigned['tasks'],'settings':assigned['settings'],
                 'fixed_scenes':[s for s in config['fixed_scenes'] if s['task'] in assigned['tasks'] and s['setting'] in assigned['settings']]}
        validate_records(current,records)
        plan=[(t,s,seed) for t in assigned['tasks'] for s in assigned['settings']
              for seed in [r['seed'] for r in current['fixed_scenes'] if r['task']==t and r['setting']==s]]
        if keys!=plan[:len(keys)] or shard['first_remaining']!=next_candidate(current,records):
            raise ValueError('Timeout transition must continue the exact remaining sparse plan')
        return manifest
    if extension:
        from .planning import next_candidate
        prior = json.loads(inside(root, manifest['source_config']).read_text())
        expected = {**prior, 'valid_per_group': len(prior['seeds']),
                    'max_consecutive_nonvalid': 20}
        if config != expected or prior.get('valid_per_group'):
            raise ValueError('Extension may only change fixed candidates into the same valid target')
        if diagnostic.get('job_id') != manifest['job_id'] or not diagnostic.get('user_request'):
            raise ValueError('Extension preparation must identify the user-authorized job')
        parent = manifest['parent_resume']
        parent_path = inside(folder, parent['manifest'])
        if sha256(parent_path) != parent['sha256']:
            raise ValueError('Extension parent manifest changed')
        if keys != plan:
            raise ValueError('Extension must preserve the entire completed fixed plan')
        source = inside(root, shard['source_records'])
        if source.read_bytes() != records_path.read_bytes():
            raise ValueError('Extension source records were not preserved byte for byte')
        current = {**config, 'tasks': assigned['tasks'], 'settings': assigned['settings']}
        if shard['first_remaining'] != next_candidate(current, records):
            raise ValueError('Extension must begin at the first untried seed of a deficient group')
        return manifest
    if config.get('valid_per_group'):
        from .planning import next_candidate
        transition = manifest.get('purpose') == 'expert_quota_transition'
        exception_transition = manifest.get('purpose') == 'expert_exception_transition'
        if manifest.get('purpose') != 'valid_target_resume' and not transition and not exception_transition:
            raise ValueError('Valid-target recovery requires an explicit valid-target resume manifest')
        parent = manifest['parent_resume']
        parent_path = inside(folder, parent['manifest'])
        if sha256(parent_path) != parent['sha256']:
            raise ValueError('Recovery parent manifest changed')
        if exception_transition:
            from .expert_outcomes import OUTER_EXCEPTION_POLICY, settled_expert_error
            from .expert_quota import validate_quota
            prior = json.loads(inside(root,manifest['source_config']).read_text())
            if hashlib.sha256(json.dumps(prior,sort_keys=True,separators=(',',':')).encode()).hexdigest() != json.loads(parent_path.read_text())['config_sha256']:
                raise ValueError('Exception transition source configuration changed')
            expected = {**prior,'protocol':'user_pending_policy_expert5_outer_exception_v2',
                'expert_exception_policy':OUTER_EXCEPTION_POLICY,
                'settled_expert_errors':config.get('settled_expert_errors'),
                'expert_quota':{**prior['expert_quota'],'inherited':config['expert_quota']['inherited']}}
            if config != expected or not diagnostic.get('user_authorization') or not diagnostic.get('budget_persistence_verified'):
                raise ValueError('Exception transition may only change its approved error boundary and inherited ledger')
            validate_quota(config['expert_quota'])
            resolved = [r for r in records if settled_expert_error(config,r)]
            assigned_keys = {(t,s) for t in assigned['tasks'] for s in assigned['settings']}
            entries = [e for e in config['settled_expert_errors'] if (e['task'],e['setting']) in assigned_keys]
            if len(resolved) != len(entries):
                raise ValueError('Each settled expert error must remain in the immutable prefix')
            for r in resolved:
                history = [h for h in manifest['historical_interruptions']
                           if h.get('execution_segment')==r.get('execution_segment') and
                           (h['task'],h['setting'],h.get('seed'))==(r['task'],r['setting'],r['seed'])]
                if len(history)!=1 or history[0].get('counts_as_attempt') is not False or not history[0].get('counted_in_candidate_records'):
                    raise ValueError('Settled expert error must be disclosed without double counting its attempt')
        elif transition:
            from .expert_quota import validate_quota
            prior = json.loads(inside(root, manifest['source_config']).read_text())
            if hashlib.sha256(json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != json.loads(parent_path.read_text())['config_sha256']:
                raise ValueError('Protocol transition source configuration changed')
            expected = {**prior, 'protocol': 'user_pending_policy_expert5_v1',
                        'expert_quota': config.get('expert_quota')}
            if config != expected or prior.get('expert_quota'):
                raise ValueError('Protocol transition may only add the approved expert quota')
            validate_quota(config['expert_quota'])
            if (not diagnostic.get('user_authorization') or not diagnostic.get('policy_entry_verified')
                    or not diagnostic.get('budget_persistence_verified')):
                raise ValueError('Protocol transition needs verified entry, quota and authorization')
        elif json.loads(parent_path.read_text()).get('config_sha256') != manifest['config_sha256']:
            raise ValueError('Recovery cannot change the valid-target configuration')
        if diagnostic.get('job_id') != manifest['job_id']:
            raise ValueError('Recovery diagnosis must identify this job')
        source = inside(root, shard['source_records'])
        excluded = shard.get('excluded_original_errors', [])
        if (transition or exception_transition) and excluded:
            raise ValueError('Protocol transition cannot exclude any source result')
        retained, matched = [], []
        for line in source.read_bytes().splitlines(keepends=True):
            row = json.loads(line)
            expected = dict(task=row['task'], setting=row['setting'], seed=row['seed'],
                            status=row['status'], original_line_sha256=hashlib.sha256(line).hexdigest())
            if expected in excluded:
                if row['status'] not in ('infrastructure_error', 'policy_error') or expected in matched:
                    raise ValueError('Recovery may only exclude unique archived execution error rows')
                matched.append(expected)
            else:
                retained.append(line)
        if len(matched) != len(excluded):
            raise ValueError('Recovery excluded-error ledger does not match the original rows')
        if b''.join(retained) != records_path.read_bytes():
            raise ValueError('Recovery source records were not preserved byte for byte')
        if transition or exception_transition:
            from .expert_quota import inherited_budget
            baseline = config['expert_quota']['inherited']
            for task in assigned['tasks']:
                for setting in assigned['settings']:
                    expected = inherited_budget(records, manifest['historical_interruptions'], task, setting)
                    if baseline.count(expected) != 1:
                        raise ValueError('Transition would reset or miscount an inherited expert budget')
        current = {**config, 'tasks': assigned['tasks'], 'settings': assigned['settings']}
        if shard['first_remaining'] != next_candidate(current, records):
            raise ValueError('Recovery must resume the interrupted original seed of a deficient group')
        return manifest
    if keys != plan[:len(keys)]:
        raise ValueError('Recovery must retain the exact completed planned prefix')
    first = shard['first_remaining']
    if tuple(first[k] for k in ('task', 'setting', 'seed')) != plan[len(keys)]:
        raise ValueError('Recovery must resume the interrupted original seed')
    return manifest


def verify_recovered_scene(manifest_path, task, setting, seed, obs, instruction, parameters):
    manifest = json.loads(Path(manifest_path).read_text())
    scenes = [s for s in manifest['scene_checks']
              if (s['task'], s['setting'], s['seed']) == (task, setting, seed)]
    if not scenes:
        return False
    scene = scenes[0]
    if instruction != scene['instruction'] or parameters != scene['instruction_parameters']:
        raise ValueError('Recovered scene instruction or expert parameters changed')
    for name, expected in scene['camera_rgb_sha256'].items():
        actual = hashlib.sha256(obs['observation'][name]['rgb'].tobytes()).hexdigest()
        if actual != expected:
            raise ValueError('Recovered initial camera changed: ' + name)
    if list(obs['joint_action']['vector']) != scene['initial_joints']:
        raise ValueError('Recovered initial joints changed')
    return True
