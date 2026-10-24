"""Deterministic scene scheduling; policy failures count towards valid targets."""

from .expert_outcomes import acceptable_record


def validate_fixed_scenes(config):
    """An explicit sparse plan selects scenes only; it never replaces a seed."""
    scenes = config.get('fixed_scenes')
    if scenes is None:
        return
    if config.get('valid_per_group') is not None or config.get('expert_quota'):
        raise ValueError('Fixed scenes cannot use candidate replacement or expert quotas')
    if not isinstance(scenes, list) or not 1 <= len(scenes) <= 15000:
        raise ValueError('Fixed scenes must be a nonempty bounded list')
    seen = set()
    for scene in scenes:
        if not isinstance(scene, dict) or set(scene) != {'task', 'setting', 'seed'}:
            raise ValueError('A fixed scene must contain task, setting and seed only')
        key = (scene['task'], scene['setting'], scene['seed'])
        if (not isinstance(key[0], str) or not isinstance(key[1], str)
                or type(key[2]) is not int or not 0 <= key[2] < 2**31
                or key[0] not in config['tasks'] or key[1] not in config['settings']
                or key[2] not in config['seeds'] or key in seen):
            raise ValueError('Invalid or duplicate fixed scene: ' + str(key))
        seen.add(key)


def group_seeds(config, task, setting):
    if config.get('fixed_scenes') is not None:
        return [r['seed'] for r in config['fixed_scenes'] if r['task'] == task and r['setting'] == setting]
    return config['seeds']


def planned_count(config):
    from .fork2000_plan import active, TARGET
    if active(config):return len(config['tasks'])*len(config['settings'])*TARGET
    if config.get('fixed_scenes') is not None:
        return len(config['fixed_scenes'])
    return len(config['tasks']) * len(config['settings']) * config.get('valid_per_group', len(config['seeds']))


def validate_records(config, records):
    from .fork2000_plan import active, validate_history
    if active(config):return validate_history(config,records)
    validate_fixed_scenes(config)
    from .timeout_outcomes import validate_timeout_policy
    validate_timeout_policy(config)
    seeds = config['seeds']
    target = config.get('valid_per_group')
    if target is not None:
        if type(target) is not int or target < 1:
            raise ValueError('Valid target must be a positive integer')
        if not seeds or seeds != list(range(seeds[0], seeds[0] + len(seeds))):
            raise ValueError('Valid-target seeds must be ascending and contiguous')
        if config.get('max_consecutive_nonvalid', 20) < 1:
            raise ValueError('Nonvalid guard must be positive')
    groups = {(t, s): [] for t in config['tasks'] for s in config['settings']}
    seen = set()
    for row in records:
        key = (row['task'], row['setting'], row['seed'])
        group = groups.get(key[:2])
        if key in seen or group is None or type(key[2]) is not int:
            raise ValueError('Duplicate or unassigned scene: ' + str(key))
        if not acceptable_record(config, row):
            raise ValueError('Unresolved execution error: ' + str(key))
        if target is None:
            if key[2] not in group_seeds(config, key[0], key[1]):
                raise ValueError('Scene seed outside fixed request: ' + str(key))
        else:
            if key[2] != seeds[0] + len(group):
                raise ValueError('Valid-target scene seeds skipped or reordered: ' + str(key))
            if sum(r['status'] == 'policy_completed' for r in group) >= target:
                raise ValueError('Scene recorded after group reached valid target: ' + str(key))
        seen.add(key)
        group.append(row)
    return groups


def candidate_seeds(config, records, task, setting):
    """Read the append-only record list again after each yielded candidate."""
    from .fork2000_plan import active, candidates
    if active(config):
        yield from candidates(config,records,task,setting)
        return
    while True:
        group = validate_records(config, records)[task, setting]
        target = config.get('valid_per_group')
        if target is None:
            done = {r['seed'] for r in group}
            remaining = [seed for seed in group_seeds(config, task, setting) if seed not in done]
            if not remaining:
                return
            seed = remaining[0]
        else:
            if sum(r['status'] == 'policy_completed' for r in group) == target:
                return
            trailing = 0
            for row in reversed(group):
                if row['status'] == 'policy_completed':
                    break
                trailing += 1
            if config.get('expert_quota'):
                from .expert_quota import budget, QUOTA_FLOW
                budget(config, records, task, setting)  # verify inherited scope
                # A diagnosed protocol transition retains the old guard event.
                # Its old rejected tail is frozen; the new policy-only path has
                # a fresh *initialization* guard, not a fresh expert budget.
                trailing = 0
                for row in reversed(group):
                    if row['status'] == 'policy_completed' or row.get('quota_flow') != QUOTA_FLOW:
                        break
                    trailing += 1
            if trailing >= config.get('max_consecutive_nonvalid', 20):
                raise RuntimeError(f'{task}/{setting}: {trailing} consecutive nonvalid candidates; diagnose before continuing')
            seed = config['seeds'][0] + len(group)
        yield config['seeds'].index(seed) if target is None else len(group), seed


def next_candidate(config, records):
    from .fork2000_plan import next_scene
    return next_scene(config,records)


def report_complete(data):
    basis = data.get('completion_basis', 'attempted')
    return (not data['record_error'] and not data['execution_errors']
            and data[basis] == data['planned']
            and all(g[basis] == g['planned'] for g in data['groups']))
