"""One shared Linux lock for CLI, UI and other users of this GPU."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess


def lock_path(root):
    config_path = Path(root) / 'configs/runtime.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return Path(os.environ.get('LINGBOT_GPU_LOCK') or config.get('gpu_lock') or
                Path(root) / 'runs/gpu.lock')


@contextmanager
def gpu_lock(root):
    import fcntl
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_gpu(gpu):
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
    total, used = [int(x.strip()) for x in raw.strip().split(',')]
    if total < 31000 or used > 768:
        raise RuntimeError(f'GPU unavailable for FP32 evaluation: {used}/{total} MiB. '
                           'Release other GPU workloads first; do not start a second client.')
    return dict(total_mib=total, used_mib=used)
