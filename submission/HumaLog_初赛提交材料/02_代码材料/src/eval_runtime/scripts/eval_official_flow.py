"""Thin single-episode adapter for the pinned upstream LingBot RoboTwin client.

Reference: be969b8f, experiment/robotwin/eval_policy_client_lingbotvla.py.
Use official task methods, instruction generator, websocket inference and
take_action. Scheduling, strict fault handling and evidence are separate.
The source differential tests execute the actual upstream eval_policy function.
"""
from pathlib import Path
import random
import traceback

import numpy as np
import yaml

EXECUTION_FLOW = 'lingbot_h50_first25_reobserve_v1'


def environment_config(repo, task, setting):
    """Resolve embodiment/camera from the same official configuration files."""
    repo = Path(repo)
    config = yaml.safe_load((repo / f'task_config/{setting}.yml').read_text())
    config.update(task_name=task, task_config=setting, ckpt_setting='round2_development',
                  policy_name='lingbot_competition', eval_mode=True, save_data=False,
                  collect_data=False, eval_video_log=False, render_freq=0)
    embodiments = yaml.safe_load((repo / 'task_config/_embodiment_config.yml').read_text())
    selection = config['embodiment']
    if len(selection) == 1:
        left = right = selection[0]
        config['dual_arm_embodied'] = True
    elif len(selection) == 3:
        left, right, distance = selection
        config.update(dual_arm_embodied=False, embodiment_dis=distance)
    else:
        raise ValueError('Official embodiment configuration must have one or three entries')
    for side, name in [('left', left), ('right', right)]:
        path = embodiments[name]['file_path']
        if not path:
            raise ValueError(f'No embodiment files: {name}')
        robot = (repo / path).resolve()
        config[f'{side}_robot_file'] = str(robot)
        config[f'{side}_embodiment_config'] = yaml.safe_load((robot / 'config.yml').read_text())
    cameras = yaml.safe_load((repo / 'task_config/_camera_config.yml').read_text())
    head = cameras[config['camera']['head_camera_type']]
    config.update(head_camera_h=head['h'], head_camera_w=head['w'])
    return config


def expert_instruction(task, parameters, setting, count, generate, *, language_seed=None):
    """Official templates + NumPy selection. Optional local RNG convention is explicit.

    Upstream does not seed Python's template RNG per episode. Our historical
    per-seed language convention is retained, isolated, and recorded, not called
    upstream bitwise equivalence. Never reseed model/simulator NumPy here.
    """
    if setting not in ('demo_clean', 'demo_randomized') or type(count) is not int or count < 1:
        raise ValueError('Invalid instruction setting or template count')
    state = random.getstate() if language_seed is not None else None
    try:
        if language_seed is not None:
            random.seed(language_seed)
        descriptions = generate(task, [parameters], count)
        return str(np.random.choice(descriptions[0]['seen' if setting == 'demo_clean' else 'unseen']))
    finally:
        if state is not None:
            random.setstate(state)


def policy_inputs(env, observation):
    return {
        'observation.images.cam_high': observation['observation']['head_camera']['rgb'],
        'observation.images.cam_left_wrist': observation['observation']['left_camera']['rgb'],
        'observation.images.cam_right_wrist': observation['observation']['right_camera']['rgb'],
        'observation.state': observation['joint_action']['vector'],
        'task': env.get_instruction(),
    }


def execute_policy(env, policy, expected_execute, on_observation, on_prediction, on_chunk):
    """Upstream reset -> observation -> inference -> execute chunk -> success gate.

    Callbacks only record already available data. No expert/observer information
    is included in the request. take_action retains upstream joint control,
    interpolation, task success checks and action-limit behavior unchanged.
    """
    policy.infer(dict(reset=True, robo_name='robotwin', path_to_pi_model=None))
    chunks, latencies = 0, []
    while env.take_action_cnt < env.step_lim and not env.eval_success:
        observation = env.get_obs()
        on_observation(observation, chunks)
        inputs = policy_inputs(env, observation)
        response = policy.infer(inputs)
        actions = np.asarray(response['action'])
        if expected_execute != 25 or actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid policy action chunk: {actions.shape}')
        latency = float(response['server_timing']['infer_ms'])
        if not np.isfinite(latency) or latency < 0:
            raise ValueError('Invalid inference timing')
        latencies.append(latency)
        trace = on_prediction(inputs, actions, chunks)
        for action in actions[:expected_execute]:
            env.take_action(action)
            if env.eval_success:
                break
        on_chunk(trace, chunks)
        chunks += 1
    return dict(status='policy_completed', success=bool(env.eval_success),
                actions=env.take_action_cnt, action_limit=env.step_lim, chunks=chunks,
                mean_inference_ms=float(np.mean(latencies)) if latencies else None)


def close_policy_scene(env, row):
    """A teardown failure must not silently consume a normal episode quota."""
    try:
        env.close_env(clear_cache=True)
    except Exception:
        row.update(status_before_close_error=row.get('status'),
                   status='infrastructure_error', close_error=traceback.format_exc())


def validate_flow_resume(previous, current):
    for name in ('execution_flow', 'expert_flow', 'expert_exception_policy', 'language_rng'):
        if previous.get(name) != current.get(name):
            raise ValueError(f'Execution flow changed; an explicit audited protocol transition is required: {name}')
