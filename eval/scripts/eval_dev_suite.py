"""Development evaluation: fixed scenes, validated inference profile, no experts.

Private scene metadata is used solely to fill the official language templates,
as in each task's play_once(). The policy receives only RGB, joints and language.
Outputs are evaluation records and are never read by the clean training loader.
"""
import argparse
import importlib
import json
import random
import shutil
import sys
import time
import traceback
import faulthandler
from eval_traceback import PeriodicTraceback
import signal
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from eval_telemetry import joint_limits_for_json
from eval_protocol_contract import validate_profile
from eval_video import SimulationVideoRecorder, DEFAULT_RECORDING_FPS, simulation_video_description

ROOT = Path(os.environ.get('LINGBOT_ROOT', Path(__file__).resolve().parents[1])).resolve()
REPO = ROOT / 'third_party/RoboTwin'
sys.path[:0] = [str(REPO), str(REPO / 'description/utils'), str(ROOT / 'third_party/lingbot-vla-v2-official-be969b8f')]
from generate_episode_instructions import generate_episode_descriptions
from deploy.websocket_client_policy import WebsocketClientPolicy

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
    config = yaml.safe_load((REPO / f'task_config/{setting}.yml').read_text())
    config.update(task_name=task, task_config=setting, ckpt_setting='round2_development',
                  policy_name='lingbot_competition', eval_mode=True, save_data=False,
                  collect_data=False, eval_video_log=False, render_freq=0, dual_arm_embodied=True)
    robot = REPO / 'assets/embodiments/aloha-agilex'
    robot_config = yaml.safe_load((robot / 'config.yml').read_text())
    config.update(left_robot_file=str(robot), right_robot_file=str(robot),
                  left_embodiment_config=robot_config, right_embodiment_config=robot_config)
    head = yaml.safe_load((REPO / 'task_config/_camera_config.yml').read_text())[config['camera']['head_camera_type']]
    config.update(head_camera_h=head['h'], head_camera_w=head['w'])
    return config


def summarize(records):
    completed = [r for r in records if r['status'] == 'policy_completed']
    per_setting = {}
    for setting in sorted({r['setting'] for r in records}):
        tasks = {}
        for task in sorted({r['task'] for r in records if r['setting'] == setting}):
            rows = [r for r in records if r['setting'] == setting and r['task'] == task]
            valid = [r for r in rows if r['status'] == 'policy_completed']
            successes = sum(r['success'] for r in valid)
            tasks[task] = {'planned': len(rows), 'completed': len(valid), 'successes': successes,
                          'setup_errors': sum(r['status'] == 'setup_error' for r in rows),
                          'execution_errors': sum(r['status'] in ('infrastructure_error','policy_error') for r in rows),
                          'success_rate': successes / len(valid) if valid else None}
        rates = [r['success_rate'] for r in tasks.values() if r['success_rate'] is not None]
        per_setting[setting] = {'tasks': tasks, 'macro_success_rate': float(np.mean(rates)) if rates else None}
    return {'protocol': 'fixed_unfiltered_development_seeds_v3', 'official_benchmark': False,
            'planned_episodes': len(records), 'policy_completed': len(completed),
            'successes': sum(r['success'] for r in completed),
            'setup_errors': sum(r['status'] == 'setup_error' for r in records),
            'infrastructure_errors': sum(r['status'] == 'infrastructure_error' for r in records),
            'policy_errors': sum(r['status'] == 'policy_error' for r in records),
            'expert_attempts': 0, 'settings': per_setting}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=ALL_TASKS, default=TASKS)
    parser.add_argument('--all-tasks', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--video-fps', type=int, choices=(10, 25), default=DEFAULT_RECORDING_FPS,
                        help='Simulation-time sampling and playback fps, only used with --save-video')
    parser.add_argument('--protocol-id', default='fixed_unfiltered_development_seeds_v3')
    parser.add_argument('--expected-execute', type=int, default=50)
    parser.add_argument('--settings', nargs='+', choices=['demo_clean', 'demo_randomized'],
                        default=['demo_clean', 'demo_randomized'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[100010, 100011])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-checkpoint', type=Path, required=True)
    parser.add_argument('--expected-precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--expected-vision-device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--expected-embedding-device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--expected-moe-backend', choices=['official', 'torch_reference'], default='official')
    parser.add_argument('--port', type=int, default=18006)
    parser.add_argument('--inference-profile', choices=['paired_legacy','official_defaults_v1','research4_paired_v1'], default='paired_legacy')
    parser.add_argument('--resume-completed', action='store_true')
    parser.add_argument('--one-episode', action='store_true',
                        help='Exit after one remaining episode so the supervisor can release simulator resources')
    args = parser.parse_args()
    if args.all_tasks:
        args.tasks = ALL_TASKS
        assert len(args.tasks) == 50
    assert len(args.seeds) == len(set(args.seeds)), 'Duplicate scene seeds'
    expected_episodes = len(args.tasks) * len(args.settings) * len(args.seeds)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_completed:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    source = args.output / (f'source-resume-{stamp}' if args.resume_completed else 'source')
    source.mkdir()
    for filename in ('eval_dev_suite.py', 'eval_isolated.py', 'eval_telemetry.py', 'eval_task_metadata.py', 'eval_protocol_contract.py', 'eval_video.py', 'eval_traceback.py', 'serve_official_checkpoint.py', 'team_eval.py'):
        shutil.copy2(ROOT / 'scripts' / filename, source / filename)
    shutil.copytree(ROOT / 'src/eval_platform', source / 'eval_platform',
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
            return sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(render_device)])
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
                'expert_attempts': 0, 'server': metadata,
                'video': simulation_video_description(args.video_fps) if args.save_video else False,
                'recording': {'enabled': args.save_video, 'fps': args.video_fps, 'time_axis': 'simulation time'},
                'render_device': render_device or 'upstream_default',
                'renderer': json.loads((ROOT / 'artifacts/oidn-blackwell.json').read_text()),
                'execution': {'simulator_process': 'one_episode' if args.one_episode else 'shared'},
                'failure_images': 'All three cameras before each policy request and at end; no extra simulation steps.',
                'diagnostics': 'Read-only privileged actor poses/contact/door position; never passed to policy or training.'}
    records = []
    if args.resume_completed:
        previous = json.loads((args.output / 'protocol.json').read_text())
        for key in ('tasks', 'seeds', 'settings', 'instruction_types', 'server', 'renderer'):
            assert previous[key] == protocol[key], f'Resumed protocol differs: {key}'
        records = [json.loads(line) for line in (args.output / 'episodes.jsonl').read_text().splitlines()]
        assert all(r['status'] in ('policy_completed', 'setup_error') for r in records), 'Preserve and resolve recorded errors before starting a new attempt'
        (args.output / f'resume-protocol-{stamp}.json').write_text(json.dumps(protocol, indent=2))
    else:
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    completed = {(r['task'], r['setting'], r['seed']) for r in records}
    assert len(completed) == len(records)
    assert completed <= {(t, s, n) for t in args.tasks for s in args.settings for n in args.seeds}

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
            for episode, seed in enumerate(args.seeds):
                if (task, setting, seed) in completed:
                    continue
                folder = args.output / task / setting / str(seed)
                if folder.exists():
                    assert args.resume_completed
                    folder.rename(folder.with_name(folder.name + f'.interrupted-{stamp}'))
                folder.mkdir(parents=True)
                env = task_type()
                row = {'task': task, 'setting': setting, 'seed': seed, 'success': False,
                       'expert_attempts': 0, 'sampling_seed': metadata.get('sampling_seed_per_episode'),
                       'sampling_seed_at_process_start': metadata.get('sampling_seed_at_process_start'),
                       'inference_profile': args.inference_profile}
                row['execution_segment'] = os.environ.get('LINGBOT_EVAL_SEGMENT', 'original')
                row['video_requested'] = args.save_video
                row['requested_video_fps'] = args.video_fps
                row['simulator_openblas_threads'] = os.environ.get('OPENBLAS_NUM_THREADS')
                phase, started = 'setup', time.monotonic()
                video = video_log = recorder = None
                tracebacks = PeriodicTraceback(120).start()
                progress('setup_started', task, setting, seed)
                print(json.dumps({'event': 'setup_started', 'task': task, 'setting': setting, 'seed': seed}), flush=True)
                try:
                    random.seed(seed)
                    env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **config)
                    parameters = scene_language_parameters(task, env)
                    random.seed(seed)
                    descriptions = generate_episode_descriptions(task, [parameters], 1)
                    instruction = descriptions[0]['seen' if setting == 'demo_clean' else 'unseen'][0]
                    env.set_instruction(instruction=instruction)
                    # Keep upstream cached-frame recording disabled in both modes.
                    env.eval_video_path = None
                    row['instruction'] = instruction
                    if args.save_video:
                        video_log = (folder / 'ffmpeg.log').open('wb')
                        video = subprocess.Popen(['ffmpeg','-nostdin','-y','-loglevel','error',
                            '-f','rawvideo','-pixel_format','rgb24','-video_size',
                            f"{config['head_camera_w']}x{config['head_camera_h']}",
                            '-framerate',str(args.video_fps),'-i','-','-pix_fmt','yuv420p','-vcodec','libx264',
                            '-threads','2','-crf','23','-movflags','+faststart',str(folder / 'episode.mp4')],
                            stdin=subprocess.PIPE, stderr=video_log)
                        # The upstream callback writes cached now_obs, which stays
                        # unchanged within an action chunk. Record fresh frames below.
                        env.eval_video_path = None
                        recorder = SimulationVideoRecorder(env, video.stdin, folder / 'video-frames.jsonl',
                                                       config['head_camera_w'], config['head_camera_h'], fps=args.video_fps)
                    phase = 'policy'
                    progress('policy_started', task, setting, seed)
                    print(json.dumps({'event': 'policy_started', 'task': task, 'setting': setting, 'seed': seed}), flush=True)
                    policy.reset('robotwin')
                    latencies, chunks = [], 0
                    while env.take_action_cnt < env.step_lim and not env.eval_success:
                        obs = env.get_obs()
                        head = obs['observation']['head_camera']['rgb']
                        if recorder is not None and recorder.frames == 0:
                            recorder.write_initial(head, env.take_action_cnt)
                        save_cameras(obs, folder, f'chunk-{chunks:03d}')
                        inputs = {'observation.images.cam_high': head,
                            'observation.images.cam_left_wrist': obs['observation']['left_camera']['rgb'],
                            'observation.images.cam_right_wrist': obs['observation']['right_camera']['rgb'],
                            'observation.state': obs['joint_action']['vector'], 'task': instruction}
                        response = policy.infer(inputs)
                        actions = np.asarray(response['action'])
                        assert actions.shape == (args.expected_execute, 14) and np.isfinite(actions).all()
                        latencies.append(response['server_timing']['infer_ms'])
                        trace = {'chunk': chunks, 'action_count_before': env.take_action_cnt,
                                 'state_before': np.asarray(inputs['observation.state']).tolist(),
                                 'predicted_actions': actions.tolist(), 'diagnostics_before': diagnostics(task, env)}
                        # Preserve the currently executing chunk even if native
                        # simulator code crashes before its completion record.
                        with (folder / 'predictions.jsonl').open('a') as file:
                            file.write(json.dumps(trace, allow_nan=False) + '\n')
                            file.flush()
                            os.fsync(file.fileno())
                        for action in actions:
                            env.take_action(action)
                            if env.eval_success or env.take_action_cnt >= env.step_lim:
                                break
                        trace.update(action_count_after=env.take_action_cnt, success=bool(env.eval_success),
                                     diagnostics_after=diagnostics(task, env))
                        with (folder / 'trajectory.jsonl').open('a') as file:
                            file.write(json.dumps(trace, allow_nan=False) + '\n')
                        chunks += 1
                        progress('chunk_completed', task, setting, seed, chunks=chunks,
                                 actions=env.take_action_cnt)
                    final_obs = env.get_obs()
                    save_cameras(final_obs, folder, 'final')
                    if recorder is not None:
                        recorder.finish(final_obs['observation']['head_camera']['rgb'], env.take_action_cnt)
                    row.update(status='policy_completed', success=bool(env.eval_success), actions=env.take_action_cnt,
                               action_limit=env.step_lim, chunks=chunks, mean_inference_ms=float(np.mean(latencies)))
                except Exception as error:
                    status = 'setup_error' if phase == 'setup' and type(error).__name__ == 'UnStableError' else (
                        'infrastructure_error' if phase == 'setup' else 'policy_error')
                    row.update(status=status, error=traceback.format_exc())
                finally:
                    tracebacks.close()
                    if video is not None:
                        try:
                            video.stdin.close()
                            code = video.wait(timeout=30)
                            if code != 0:
                                raise RuntimeError(f'ffmpeg failed: {code}')
                            row['video'] = str((folder / 'episode.mp4').relative_to(args.output))
                        except Exception:
                            video.kill()
                            video.wait()
                            row.update(status='infrastructure_error', video_error=traceback.format_exc())
                        finally:
                            if recorder is not None:
                                row['video_capture'] = recorder.metadata()
                                recorder.close()
                            video_log.close()
                            env.eval_video_path = None
                    try:
                        env.close_env(clear_cache=True)
                    except Exception:
                        row['close_error'] = traceback.format_exc()
                row['seconds'] = time.monotonic() - started
                records.append(row)
                (folder / 'episode.json').write_text(json.dumps(row, indent=2))
                with (args.output / 'episodes.jsonl').open('a') as file:
                    file.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
                progress('episode_recorded', task, setting, seed, status=row['status'])
                if row['status'] in ('policy_error', 'infrastructure_error'):
                    report = {**summarize(records), 'protocol':args.protocol_id, 'attempted_episodes': len(records), 'planned_episodes': expected_episodes}
                    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
                    raise SystemExit(2)
                if args.one_episode:
                    report = {**summarize(records), 'protocol':args.protocol_id, 'attempted_episodes': len(records),
                              'planned_episodes': expected_episodes}
                    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
                    return
    report = {**summarize(records), 'protocol':args.protocol_id, 'attempted_episodes': len(records), 'planned_episodes': expected_episodes}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if report['setup_errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
