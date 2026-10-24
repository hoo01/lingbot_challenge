"""Physical GPU leases, compatible with the original exclusive fleet lock."""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import subprocess
import re


def lock_path(root):
    config_path = Path(root) / 'configs/runtime.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return Path(os.environ.get('LINGBOT_GPU_LOCK') or config.get('gpu_lock') or
                Path(root) / 'runs/gpu.lock')


def resolve_gpus(values):
    result = []
    for value in values:
        rows = subprocess.check_output(['nvidia-smi', '-i', str(value),
            '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
        if len(rows) != 1 or not re.fullmatch(r'GPU-[a-fA-F0-9-]+', rows[0].strip()):
            raise ValueError('Each GPU selector must resolve to one physical GPU UUID')
        result.append(rows[0].strip())
    if not result or len(set(result)) != len(result):
        raise ValueError('GPU selectors must be nonempty and distinct')
    return result


def gpu_port(gpu, base=18147):
    index=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=index',
                                   '--format=csv,noheader'],text=True,timeout=10).strip()
    if not index.isdigit() or not 1 <= base+int(index) <= 65535:
        raise ValueError('Invalid inference port allocation')
    return base+int(index)


class Lease:
    def __init__(self, streams):
        self.fds = tuple(s.fileno() for s in streams)

    def fileno(self):
        return self.fds[0]


@contextmanager
def gpu_lock(root, gpu=None):
    import fcntl
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open('a'))]
        # Legacy callers take EX and exclude the whole fleet. Modern callers
        # share that gate, then exclusively own one physical device.
        fcntl.flock(streams[0], (fcntl.LOCK_EX if gpu is None else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        if gpu is not None:
            if not re.fullmatch(r'GPU-[a-fA-F0-9-]+', gpu):
                raise ValueError('GPU lease requires a canonical physical UUID')
            device_path = path.with_name(path.name + '.' + gpu)
            streams.append(stack.enter_context(device_path.open('a')))
            fcntl.flock(streams[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Closing preserves inherited leases until the last descendant exits.
        yield Lease(streams)


def inherited_fds():
    return tuple(int(x) for x in os.environ.get('LINGBOT_LEASE_FDS', '').split(',') if x)


def check_gpu(gpu):
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
    total, used = [int(x.strip()) for x in raw.strip().split(',')]
    if total < 31000 or used > 768:
        raise RuntimeError(f'GPU unavailable for FP32 evaluation: {used}/{total} MiB. '
                           'Release other GPU workloads first; do not start a second client.')
    return dict(total_mib=total, used_mib=used)
