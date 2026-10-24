"""Record synchronized cameras without advancing the scene or policy RNG."""
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


CAPTURE_VERSION = 'head_camera_after_action_v1'
VIDEO_DESCRIPTION = 'fresh head-camera frame after every executed action, plus initial frame; 10 fps'
SIMULATION_CAPTURE_VERSION = 'head_camera_simulation_time_v2'
SIMULATION_VIDEO_FPS = 25
SIMULATION_VIDEO_DESCRIPTION = 'head-camera frames sampled on simulation time at 25 fps; 1x simulation playback'
DEFAULT_RECORDING_FPS = 10


def simulation_video_description(fps, observer_video=True):
    cameras = 'head + observer' if observer_video else 'head'
    return f'{cameras} camera frames on a shared simulation clock at {fps} fps; 1x simulation playback'


def open_video_recording(env, folder, width, height, fps, observer_video=True):
    """Only attach requested recording cameras; leave policy observations alone."""
    videos, extra_views = {}, {}
    videos['head_camera'] = VideoEncoder(folder / 'episode.mp4', width, height, fps, folder / 'ffmpeg.log')
    if observer_video:
        videos['observer_camera'] = VideoEncoder(folder / 'observer.mp4', 320, 240, fps,
                                                folder / 'ffmpeg-observer.log')
        extra_views['observer_camera'] = ActionVideoRecorder(env, videos['observer_camera'].sink,
            folder / 'observer-video-frames.jsonl', 320, 240, camera_name='observer_camera')
    env.eval_video_path = None
    recorder = SimulationVideoRecorder(env, videos['head_camera'].sink, folder / 'video-frames.jsonl',
                                      width, height, fps=fps, extra_views=extra_views)
    return videos, extra_views, recorder


class VideoEncoder:
    """Own one ffmpeg pipe and wait for its buffered final frames to finish."""

    def __init__(self, path, width, height, fps, log_path):
        self.path = path
        self.log = log_path.open('xb')
        try:
            self.process = subprocess.Popen(['ffmpeg', '-nostdin', '-n', '-loglevel', 'error',
                '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                '-framerate', str(fps), '-i', '-', '-pix_fmt', 'yuv420p', '-vcodec', 'libx264',
                '-threads', '2', '-crf', '23', '-movflags', '+faststart', str(path)],
                stdin=subprocess.PIPE, stderr=self.log)
        except BaseException:
            self.log.close()
            raise
        self.sink = self.process.stdin

    def finish(self):
        try:
            self.sink.close()
            code = self.process.wait(timeout=30)
            if code != 0:
                raise RuntimeError(f'ffmpeg failed for {self.path.name}: {code}')
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


class ActionVideoRecorder:
    """The caller owns ffmpeg; recording never calls get_obs or _update_render.

    RoboTwin's take_action already synchronizes rendering after its physics
    steps. Taking a picture of that synchronized scene avoids another lighting
    randomization and leaves env.now_obs and the policy observation cadence alone.
    """

    def __init__(self, env, sink, frame_log: Path, width: int, height: int, camera_name='head_camera'):
        self.camera_name = camera_name
        if camera_name == 'observer_camera':
            self.camera = env.cameras.observer_camera
        else:
            cameras = [camera for camera, name in zip(
                env.cameras.static_camera_list, env.cameras.static_camera_name) if name == camera_name]
            if len(cameras) != 1:
                raise ValueError(f'Expected exactly one {camera_name} for video')
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
        self._write(self.capture_rgb(), action_count, 'fresh_head_camera')

    def capture_rgb(self):
        self.camera.take_picture()
        color = self.camera.get_picture('Color')
        return (np.asarray(color)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

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

    def __init__(self, env, sink, frame_log: Path, width: int, height: int, fps=SIMULATION_VIDEO_FPS,
                 extra_views=None):
        if type(fps) is not int or fps not in (10, 25):
            raise ValueError('Simulation video fps must be 10 or 25')
        self.fps = fps
        self.extra_views = dict(extra_views or {})
        self.terminal_views = {}
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
                self._write_timed(self.capture_rgb(), self.env.take_action_cnt, 'simulation_tick')
        return result

    def _write_timed(self, rgb, action_count, source, extra_rgb=None):
        timing = dict(simulation_step=self.simulation_steps,
                      simulation_time_seconds=self.simulation_seconds, video_time_seconds=self.frames / self.fps)
        for name, view in self.extra_views.items():
            if view.frames != self.frames:
                raise RuntimeError(f'Camera video frame counts differ: {name}')
        super()._write(rgb, action_count, source, **timing)
        for name, view in self.extra_views.items():
            view._write(extra_rgb[name] if extra_rgb is not None else view.capture_rgb(),
                        action_count, source, **timing)
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
        if not self.frames:
            raise RuntimeError('Terminal video frame requires an initial frame')
        self.terminal_views = {'head_camera': np.asarray(rgb).copy(),
                               **{name: view.capture_rgb() for name, view in self.extra_views.items()}}
        partial = self.simulation_seconds > self.last_capture_seconds + 1e-9
        # A final get_obs can refresh the image even at an exact sampling boundary.
        # Preserve that state as well as a partial final interval, on every view.
        changed = any(hashlib.sha256(np.ascontiguousarray(self.terminal_views[name]).tobytes()).hexdigest()
                      != view.previous_hash or action_count != view.last_action_count
                      for name, view in {'head_camera': self, **self.extra_views}.items())
        if partial or changed:
            self._write_timed(rgb, action_count, 'terminal_partial_interval' if partial else 'terminal_state',
                              self.terminal_views)
        self.finished = True

    def metadata(self, camera_name='head_camera'):
        view = self if camera_name == 'head_camera' else self.extra_views[camera_name]
        return {'version': SIMULATION_CAPTURE_VERSION if camera_name == 'head_camera' else 'observer_camera_simulation_time_v1',
                'camera': camera_name, 'fps': self.fps, 'shared_clock': 'policy_simulation_steps',
                'sampling': f'initial frame + every {1 / self.fps:g} simulation seconds + terminal state',
                'time_axis': 'simulation time', 'playback_speed': 1.0,
                'physics_timestep_seconds': self.timestep, 'simulation_steps': self.simulation_steps,
                'simulation_seconds': self.simulation_seconds,
                'video_duration_seconds': view.frames / self.fps,
                'max_sampling_lag_seconds': self.max_sampling_lag_seconds,
                'frames': view.frames, 'changed_frames': view.changed_frames,
                'terminal_frame_included': self.finished, 'last_action_count': view.last_action_count,
                'terminal_rgb_sha256': view.previous_hash if self.finished else None,
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
        for view in self.extra_views.values():
            view.close()
