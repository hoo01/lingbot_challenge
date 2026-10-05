"""Record fresh head-camera frames without advancing the scene or policy RNG."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CAPTURE_VERSION = 'head_camera_after_action_v1'
VIDEO_DESCRIPTION = 'fresh head-camera frame after every executed action, plus initial frame; 10 fps'
SIMULATION_CAPTURE_VERSION = 'head_camera_simulation_time_v1'
SIMULATION_VIDEO_FPS = 25
SIMULATION_VIDEO_DESCRIPTION = 'head-camera frames sampled on simulation time at 25 fps; 1x simulation playback'
DEFAULT_RECORDING_FPS = 10


def simulation_video_description(fps):
    return f'head-camera frames sampled on simulation time at {fps} fps; 1x simulation playback'


class ActionVideoRecorder:
    """The caller owns ffmpeg; recording never calls get_obs or _update_render.

    RoboTwin's take_action already synchronizes rendering after its physics
    steps. Taking a picture of that synchronized scene avoids another lighting
    randomization and leaves env.now_obs and the policy observation cadence alone.
    """

    def __init__(self, env, sink, frame_log: Path, width: int, height: int):
        cameras = [camera for camera, name in zip(
            env.cameras.static_camera_list, env.cameras.static_camera_name)
            if name == 'head_camera']
        if len(cameras) != 1:
            raise ValueError('Expected exactly one head camera for action video')
        self.camera = cameras[0]
        self.sink = sink
        self.shape = (height, width, 3)
        self.log = frame_log.open('x', encoding='utf-8')
        self.frames = self.changed_frames = 0
        self.previous_hash = None
        self.last_action_count = None

    def write_initial(self, rgb, action_count=0):
        if self.frames:
            raise RuntimeError('Initial video frame was already written')
        self._write(rgb, action_count, 'policy_initial_observation')

    def capture_after_action(self, action_count):
        if self.last_action_count is None or action_count != self.last_action_count + 1:
            raise ValueError('Video requires exactly one frame per executed action')
        self.camera.take_picture()
        color = self.camera.get_picture('Color')
        rgb = (np.asarray(color)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        self._write(rgb, action_count, 'fresh_head_camera')

    def _write(self, rgb, action_count, source, **timing):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.shape != self.shape:
            raise ValueError(f'Invalid video frame: {rgb.shape}, {rgb.dtype}; expected {self.shape}, uint8')
        payload = np.ascontiguousarray(rgb).tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        changed = self.previous_hash is not None and digest != self.previous_hash
        self.sink.write(payload)
        self.log.write(json.dumps({'frame': self.frames, 'action_count': action_count,
                                   'source': source, 'rgb_sha256': digest,
                                   'changed_from_previous': changed, **timing}) + '\n')
        self.log.flush()
        self.frames += 1
        self.changed_frames += int(changed)
        self.previous_hash = digest
        self.last_action_count = action_count

    def metadata(self):
        return {'version': CAPTURE_VERSION, 'fps': 10,
                'sampling': 'initial observation + fresh head camera after each executed action',
                'time_axis': 'action index, not wall-clock or simulation time',
                'frames': self.frames, 'changed_frames': self.changed_frames,
                'extra_physics_steps': 0, 'extra_policy_observations': 0}

    def close(self):
        self.log.close()


class SimulationVideoRecorder(ActionVideoRecorder):
    """Observe existing physics steps and capture after existing render updates.

    No additional scene.step/get_obs/_update_render calls are made. Initial
    settling is excluded because the recorder is attached after setup_demo.
    """

    def __init__(self, env, sink, frame_log: Path, width: int, height: int, fps=SIMULATION_VIDEO_FPS):
        if type(fps) is not int or fps not in (10, 25):
            raise ValueError('Simulation video fps must be 10 or 25')
        self.fps = fps
        super().__init__(env, sink, frame_log, width, height)
        self.env = env
        self.scene = env.scene
        self.timestep = float(self.scene.get_timestep())
        if not math.isfinite(self.timestep) or not 0 < self.timestep <= 1 / self.fps:
            self.log.close()
            raise ValueError(f'Unsupported physics timestep for video: {self.timestep}')
        self.simulation_steps = 0
        self.simulation_seconds = 0.0
        self.last_capture_seconds = None
        self.max_sampling_lag_seconds = 0.0
        self.finished = False
        self.original_step = self.scene.step
        self.original_update = env._update_render
        self.saved_step = vars(self.scene).get('step')
        self.saved_update = vars(env).get('_update_render')
        self.step_hook = self._step
        self.update_hook = self._update_render
        self.scene.step = self.step_hook
        self.env._update_render = self.update_hook

    def _step(self, *args, **kwargs):
        timestep = float(self.scene.get_timestep())
        if timestep != self.timestep:
            raise RuntimeError('Physics timestep changed during recorded episode')
        result = self.original_step(*args, **kwargs)
        self.simulation_steps += 1
        self.simulation_seconds = self.simulation_steps * self.timestep
        return result

    def _update_render(self, *args, **kwargs):
        result = self.original_update(*args, **kwargs)
        if self.frames and not self.finished:
            target = self.frames / self.fps
            if self.simulation_seconds + 1e-9 >= target:
                lag = self.simulation_seconds - target
                if lag > self.timestep + 1e-8:
                    raise RuntimeError('Missed simulation-time video sampling boundary')
                self.max_sampling_lag_seconds = max(self.max_sampling_lag_seconds, lag)
                self.camera.take_picture()
                color = self.camera.get_picture('Color')
                rgb = (np.asarray(color)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                self._write_timed(rgb, self.env.take_action_cnt, 'simulation_tick')
        return result

    def _write_timed(self, rgb, action_count, source):
        super()._write(rgb, action_count, source, simulation_step=self.simulation_steps,
                       simulation_time_seconds=self.simulation_seconds,
                       video_time_seconds=self.frames / self.fps)
        self.last_capture_seconds = self.simulation_seconds

    def write_initial(self, rgb, action_count=0):
        if self.frames or self.simulation_steps:
            raise RuntimeError('Initial frame must precede policy physics steps')
        self._write_timed(rgb, action_count, 'policy_initial_observation')

    def capture_after_action(self, action_count):
        raise RuntimeError('Simulation video is sampled inside the existing physics loop')

    def finish(self, rgb, action_count):
        if self.finished:
            return
        if self.last_capture_seconds is not None and self.simulation_seconds > self.last_capture_seconds + 1e-9:
            # Include the actual terminal state. CFR rounds this partial final
            # interval up by less than one frame; record the exact simulation time.
            self._write_timed(rgb, action_count, 'terminal_partial_interval')
        self.finished = True

    def metadata(self):
        return {'version': SIMULATION_CAPTURE_VERSION, 'fps': self.fps,
                'sampling': f'initial frame + every {1 / self.fps:g} simulation seconds + terminal partial interval',
                'time_axis': 'simulation time', 'playback_speed': 1.0,
                'physics_timestep_seconds': self.timestep, 'simulation_steps': self.simulation_steps,
                'simulation_seconds': self.simulation_seconds,
                'video_duration_seconds': self.frames / self.fps,
                'max_sampling_lag_seconds': self.max_sampling_lag_seconds,
                'frames': self.frames, 'changed_frames': self.changed_frames,
                'extra_physics_steps': 0, 'extra_policy_observations': 0}

    def close(self):
        if vars(self.scene).get('step') is self.step_hook:
            if self.saved_step is None:
                delattr(self.scene, 'step')
            else:
                self.scene.step = self.saved_step
        if vars(self.env).get('_update_render') is self.update_hook:
            if self.saved_update is None:
                delattr(self.env, '_update_render')
            else:
                self.env._update_render = self.saved_update
        super().close()
