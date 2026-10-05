"""Loopback UI; GPU worker requires explicit --with-worker."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(os.environ['LINGBOT_ROOT']).resolve()
sys.path.insert(0, str(root/'src'))
from eval_platform.servers import load_catalog, select_server, server_state
name, server = select_server(load_catalog())
if server_state(root)['current']['id'] != name:
    raise ValueError('Start the web service through the configured runtime entry')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--with-worker', action='store_true')
parser.add_argument('--gpu', default=str(server['gpu']))
parser.add_argument('--port', type=int, default=server['web_port'])
args = parser.parse_args()
if args.port != server['web_port']:
    parser.error('Web port differs from the server catalog; update that configuration first')
state = root/'platform-state'
state.mkdir(exist_ok=True)
lock = (state/'manager.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
env = dict(os.environ, LINGBOT_ROOT=str(root), PYTHONPATH=str(root/'src'), PYTHONUNBUFFERED='1')
commands = [('web', [sys.executable, '-m', 'uvicorn', 'eval_platform.app:app',
                     '--host', '127.0.0.1', '--port', str(args.port)])]
if args.with_worker:
    gpu_ids = subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=uuid',
                                       '--format=csv,noheader'], text=True).splitlines()
    if len(gpu_ids) != 1:
        raise ValueError('Select exactly one GPU')
    commands.append(('worker', [sys.executable, '-m', 'eval_platform.worker',
                               '--root', str(root), '--gpu', gpu_ids[0].strip(), '--port', '18100']))
children = []
def stop(signum, frame):
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
(state/'manager.pid').write_text(str(os.getpid()))
try:
    for name, command in commands:
        with (state/f'{name}.log').open('ab') as log:
            children.append(subprocess.Popen(command, env=env, stdout=log, stderr=log))
    print(f'UI: http://127.0.0.1:{args.port}; GPU worker: {args.with_worker}', flush=True)
    while all(child.poll() is None for child in children):
        time.sleep(3)
    raise RuntimeError('A platform component exited; inspect platform-state/*.log')
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
