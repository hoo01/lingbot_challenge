"""User-approved local budget: five expert initiations per pending policy slot.

Normal records consume launches even on setup failure. A durable reservation
precedes each invocation; an unaccounted reservation requires explicit recovery.
Legacy launches are inherited, never refunded. No simulator methods are changed.
"""
import json
import os
from pathlib import Path

QUOTA_FLOW = 'pending_policy_expert5_v1'


def inherited_budget(records, history, task, setting):
    group = [r for r in records if (r['task'], r['setting']) == (task, setting)]
    completed = [r for r in group if r['status'] == 'policy_completed']
    last_seed = completed[-1]['seed'] if completed else -1
    launches = sum(r.get('expert_attempts', 0) for r in group if r['seed'] > last_seed)
    # Legacy interruptions each launched one expert. New policy-only entries
    # explicitly carry zero; errors kept in records are not added a second time.
    launches += sum(h.get('expert_launches', 1) for h in history
                    if h.get('counts_as_attempt', True) and h.get('seed') is not None
                    and h['seed'] > last_seed and (h['task'], h['setting']) == (task, setting))
    return dict(task=task, setting=setting, policy_index=len(completed),
                launches=launches, candidate_count=len(group))


def validate_quota(quota):
    if not (quota['version'] == QUOTA_FLOW and quota['limit'] == 5
            and quota['scope'] == 'task_setting_pending_policy_index'
            and quota['user_authorization']):
        raise ValueError('Unsupported or unapproved expert budget')
    keys = [(b['task'], b['setting']) for b in quota['inherited']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate inherited budget')
    for b in quota['inherited']:
        if any(type(b[k]) is not int or b[k] < 0 for k in
               ('policy_index', 'launches', 'candidate_count')):
            raise ValueError('Invalid inherited budget')


def budget(config, records, task, setting):
    quota = config['expert_quota']
    validate_quota(quota)
    group = [r for r in records if (r['task'], r['setting']) == (task, setting)]
    slot = sum(r['status'] == 'policy_completed' for r in group)
    inherited = next((b for b in quota['inherited']
                      if (b['task'], b['setting']) == (task, setting)), None)
    tail = []
    for r in reversed(group):
        if r['status'] == 'policy_completed':
            break
        tail.append(r)
    if inherited and slot == inherited['policy_index']:
        if len(group) < inherited['candidate_count']:
            raise ValueError('Missing legacy candidate prefix')
        used = inherited['launches']
        new = group[inherited['candidate_count']:]
        if any(r.get('quota_flow') != QUOTA_FLOW for r in new):
            raise ValueError('Untagged candidate after protocol transition')
        used += sum(r['expert_attempts'] for r in new)
    else:
        if tail and any(r.get('quota_flow') != QUOTA_FLOW for r in tail):
            raise ValueError('Legacy tail requires an explicit inherited budget')
        used = sum(r['expert_attempts'] for r in tail)
    return dict(scope=quota['scope'], policy_index=slot, limit=5,
                used_before=used, expert_permitted=used < 5,
                inherited_over_limit=max(0, (inherited or {}).get('launches', 0) - 5)
                if inherited and slot == inherited['policy_index'] else 0)


def verify_launch_journal(path, records):
    path = Path(path)
    launches = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    seen = set()
    for event in launches:
        key = tuple(event[k] for k in ('task', 'setting', 'seed'))
        if key in seen or not 1 <= event['launch_number'] <= 5:
            raise ValueError('Invalid or duplicate expert reservation')
        seen.add(key)
        matches = [r for r in records if tuple(r[k] for k in ('task', 'setting', 'seed')) == key]
        if len(matches) != 1 or matches[0].get('expert_attempts') != 1:
            raise ValueError('Unresolved expert reservation; explicit same-seed recovery required')
        row = matches[0]
        if row.get('quota_flow') != QUOTA_FLOW or row['expert_budget']['policy_index'] != event['policy_index']:
            raise ValueError('Expert reservation scope changed')
    return launches


def reserve_launch(path, state, task, setting, seed):
    if not state['expert_permitted'] or state['used_before'] >= 5:
        raise ValueError('A sixth expert launch is forbidden')
    item = dict(event='expert_launch_reserved', task=task, setting=setting, seed=seed,
                policy_index=state['policy_index'], launch_number=state['used_before'] + 1)
    # The existing shard/GPU lease serializes writers. fsync before setup_demo.
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(item) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return item
