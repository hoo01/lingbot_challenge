"""Small unfiltered-seed policy test, not the official 100-episode benchmark.

Calls no expert and writes no training examples. Scene metadata is used only by
the official task-language generator; policy inputs are RGB, joints and language.
"""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

root = Path('/data/lingbotvla')
repo = root / 'third_party/RoboTwin'
sys.path[:0] = [str(repo), str(repo / 'description/utils'), str(root / 'third_party/lingbot-vla-v2')]
from envs.adjust_bottle import adjust_bottle
from generate_episode_instructions import generate_episode_descriptions
from deploy.websocket_client_policy import WebsocketClientPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--setting', choices=['demo_clean', 'demo_randomized'], default='demo_clean')
    p.add_argument('--seeds', type=int, nargs='+', default=[100000, 100001])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18006)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    config = yaml.safe_load((repo / f'task_config/{a.setting}.yml').read_text())
    config.update(task_name='adjust_bottle', task_config=a.setting, ckpt_setting='local_smoke',
                  policy_name='lingbot_competition', eval_mode=True, save_data=False,
                  collect_data=False, eval_video_log=False, render_freq=0, dual_arm_embodied=True)
    robot = repo / 'assets/embodiments/aloha-agilex'
    robot_config = yaml.safe_load((robot / 'config.yml').read_text())
    config.update(left_robot_file=str(robot), right_robot_file=str(robot),
                  left_embodiment_config=robot_config, right_embodiment_config=robot_config)
    cameras = yaml.safe_load((repo / 'task_config/_camera_config.yml').read_text())
    head = cameras[config['camera']['head_camera_type']]
    config.update(head_camera_h=head['h'], head_camera_w=head['w'])
    instruction_type = 'seen' if a.setting == 'demo_clean' else 'unseen'
    policy = WebsocketClientPolicy(host='127.0.0.1', port=a.port)
    metadata = policy.get_server_metadata()
    (a.output / 'protocol.json').write_text(json.dumps({
        'protocol': 'unfiltered_seed_smoke', 'setting': a.setting, 'seeds': a.seeds,
        'instruction_type': instruction_type, 'expert_attempts': 0,
        'renderer': json.loads((root / 'artifacts/oidn-blackwell.json').read_text()),
        'server': metadata, 'official_benchmark': False}, indent=2))
    records = []
    for episode, seed in enumerate(a.seeds):
        env = adjust_bottle()
        row = {'episode': episode, 'seed': seed, 'success': False, 'expert_attempts': 0}
        started = time.monotonic()
        try:
            random.seed(seed)
            env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **config)
            # Same metadata fields as the official adjust_bottle.play_once output.
            scene_info = {'{A}': f'001_bottle/base{env.model_id}',
                          '{a}': 'right' if env.qpose_tag == 1 else 'left'}
            descriptions = generate_episode_descriptions('adjust_bottle', [scene_info], 1)
            instruction = descriptions[0][instruction_type][0]
            env.set_instruction(instruction=instruction)
            row['instruction'] = instruction
            policy.reset('robotwin')
            latencies = []
            chunks = 0
            while env.take_action_cnt < env.step_lim and not env.eval_success:
                obs = env.get_obs()
                camera = obs['observation']['head_camera']['rgb']
                if chunks == 0:
                    Image.fromarray(camera).save(a.output / f'{seed}-initial.png')
                inputs = {'observation.images.cam_high': camera,
                    'observation.images.cam_left_wrist': obs['observation']['left_camera']['rgb'],
                    'observation.images.cam_right_wrist': obs['observation']['right_camera']['rgb'],
                    'observation.state': obs['joint_action']['vector'], 'task': instruction}
                response = policy.infer(inputs)
                action = np.asarray(response['action'])
                assert action.ndim == 2 and action.shape[1] == 14 and np.isfinite(action).all()
                latencies.append(response['server_timing']['infer_ms'])
                for command in action:
                    env.take_action(command)
                    if env.eval_success or env.take_action_cnt >= env.step_lim:
                        break
                chunks += 1
            Image.fromarray(env.get_obs()['observation']['head_camera']['rgb']).save(a.output / f'{seed}-final.png')
            row.update(success=bool(env.eval_success), status='policy_completed',
                       actions=env.take_action_cnt, action_limit=env.step_lim,
                       chunks=chunks, mean_inference_ms=float(np.mean(latencies)))
        except Exception:
            row.update(status='environment_or_policy_error', error=traceback.format_exc())
        finally:
            try:
                env.close_env(clear_cache=True)
            except Exception:
                row['close_error'] = traceback.format_exc()
        row['seconds'] = time.monotonic() - started
        records.append(row)
        with (a.output / 'episodes.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    report = {'protocol': 'unfiltered_seed_smoke', 'requested_episodes': len(a.seeds),
              'policy_completed': sum(r['status'] == 'policy_completed' for r in records),
              'successes': sum(r['success'] for r in records), 'expert_attempts': 0,
              'official_benchmark': False}
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if report['policy_completed'] != len(a.seeds):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
