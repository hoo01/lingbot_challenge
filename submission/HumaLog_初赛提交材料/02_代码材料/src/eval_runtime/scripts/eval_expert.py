"""One upstream RoboTwin expert attempt, with logging outside task execution.

No task/planner methods are replaced. The default stops on exceptions. The
explicitly selected upstream outer exception branch records them separately
and consumes the same five-attempt budget; cleanup and recording errors stop.
"""
import json
import time
import traceback
from eval_platform.expert_outcomes import OUTER_EXCEPTION_POLICY

EXPERT_FLOW = 'upstream_play_once_no_internal_hooks_v1'


class ExpertRejected(Exception):
    """A completed expert attempt did not satisfy the upstream success gate."""


class ExpertExecutionError(Exception):
    """An expert setup/execution Exception, not a normal rejection or policy result."""


def select_policy_entry(env, config, episode, seed, folder, row, progress, *, budget=None, journal=None,
                        exception_policy='record_and_stop'):
    """One expert initiation or a mandatory policy entry, never a sixth call."""
    if budget is not None and not budget['expert_permitted']:
        return True, None
    if budget is not None:
        from eval_platform.expert_quota import reserve_launch
        reserve_launch(journal, budget, row['task'], row['setting'], seed)
    try:
        parameters = screen_scene(env, config, episode, seed, folder, row, progress,
                                  exception_policy=exception_policy)
    except (ExpertRejected, ExpertExecutionError):
        if budget is None or budget['used_before'] + 1 < budget['limit']:
            raise
        return True, None
    return False, parameters


def screen_scene(env, config, episode, seed, folder, row, progress, *, exception_policy='record_and_stop'):
    if exception_policy not in ('record_and_stop', OUTER_EXCEPTION_POLICY):
        raise ValueError('Unknown expert exception policy')
    started = time.monotonic()
    result = {'seed': seed, 'phase': 'setup', 'passed': False,
              'flow': EXPERT_FLOW, 'exception_policy': exception_policy}
    row.update(expert_attempts=1, expert=result)
    close_attempted = False
    task_call = False
    expert_config = {**config, 'render_freq': 0}
    try:
        progress('expert_setup_started')
        task_call = True
        env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **expert_config)
        task_call = False
        result['setup_seconds'] = time.monotonic() - started
        result['phase'] = 'execution'
        progress('expert_started')
        executing = time.monotonic()
        task_call = True
        info = env.play_once()
        task_call = False
        result['execution_seconds'] = time.monotonic() - executing
        # Match the pinned LingBot client: close, then check, then reset this
        # SAME task object for the policy. In particular retain arm_tag/info.
        result['phase'] = 'close'
        close_attempted = True
        env.close_env()
        result['phase'] = 'success_check'
        result.update(plan_success=bool(env.plan_success),
                      success=bool(env.plan_success and env.check_success()))
        result['passed'] = result['plan_success'] and result['success']
        result['instruction_parameters'] = info['info']
        if not result['passed']:
            result['rejection_reason'] = 'official_success_gate_failed'
        result['phase'] = 'completed'
        progress('expert_completed')
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=traceback.format_exc())
        if not close_attempted:
            try:
                env.close_env()
            except Exception:
                result['cleanup_error'] = traceback.format_exc()
        if (exception_policy == OUTER_EXCEPTION_POLICY and isinstance(error, Exception)
                and task_call and type(error).__name__ != 'UnStableError' and result['phase'] in ('setup', 'execution')
                and not result.get('cleanup_error')):
            result['outcome'] = 'exception'
            raise ExpertExecutionError('Official expert attempt raised ' + type(error).__name__) from error
        raise
    finally:
        result['total_seconds'] = time.monotonic() - started
        (folder / 'expert.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['passed']:
        raise ExpertRejected('Completed expert attempt did not pass the official success gate')
    return result['instruction_parameters']
