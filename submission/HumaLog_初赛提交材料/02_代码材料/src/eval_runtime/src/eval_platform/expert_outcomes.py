"""Expert exceptions are distinct outcomes; acknowledged old errors stay intact."""
import hashlib
import json

OUTER_EXCEPTION_POLICY = 'official_outer_exception_counted_v1'
EXCEPTION_STAGE = 'pending_policy_expert5_outer_exception_v2'
NORMAL_STATUSES = {'policy_completed', 'setup_error', 'expert_rejected'}


def record_digest(row):
    # Reports add these two display fields; they are not part of the saved row.
    value = {k: v for k, v in row.items() if k not in ('shard', 'artifact_root')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def settled_expert_error(config, row):
    entries = config.get('settled_expert_errors', [])
    key = tuple(row[k] for k in ('task', 'setting', 'seed'))
    matches = [e for e in entries if tuple(e[k] for k in ('task', 'setting', 'seed')) == key]
    if not matches:
        return False
    if len(matches) != 1:
        raise ValueError('Duplicate expert error disposition')
    entry = matches[0]
    if (config.get('expert_exception_policy') != OUTER_EXCEPTION_POLICY
            or not entry.get('user_authorization') or entry.get('canonical_sha256') != record_digest(row)
            or row['status'] != 'infrastructure_error' or row.get('expert_attempts') != 1
            or row.get('expert', {}).get('phase') not in ('setup', 'execution')
            or not row.get('expert', {}).get('error_type') or row['expert'].get('cleanup_error')):
        raise ValueError('Expert error disposition does not match its immutable source')
    return True


def acceptable_record(config, row):
    from .fork2000_plan import active, retained_error
    if active(config):
        if row.get('expert_attempts') != 0:return False
        if retained_error(row):return True
        if row.get('status') not in ('policy_completed','policy_timeout'):return False
    if row['status'] in NORMAL_STATUSES:
        return True
    if row['status'] == 'policy_timeout':
        from .timeout_outcomes import timeout_failure
        return timeout_failure(config, row)
    if row['status'] == 'expert_exception':
        expert = row.get('expert', {})
        return (config.get('expert_exception_policy') == OUTER_EXCEPTION_POLICY
                and row.get('expert_exception_policy') == OUTER_EXCEPTION_POLICY
                and row.get('expert_attempts') == 1 and expert.get('outcome') == 'exception'
                and expert.get('phase') in ('setup', 'execution') and not expert.get('cleanup_error'))
    return settled_expert_error(config, row)


def exception_observed(config, row):
    return row.get('expert', {}).get('outcome') == 'exception' or settled_expert_error(config, row)
