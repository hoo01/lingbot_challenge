"""One physical GPU worker executing a fixed shard at a time."""
import argparse
import os
import signal
import subprocess
import sys
import time
from .store import Store
from .resources import gpu_lock, check_gpu, resolve_gpus


def stop_group(process):
    # A crashed leader can leave model/simulator descendants behind.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(store, shard, gpu, port, lock):
    identifier, number = shard['job_id'], shard['shard']
    folder = store.shard_folder(identifier, number)
    current = None
    status, error = 'interrupted', 'Worker stopped; inspect logs and preserve records'
    try:
        env = dict(os.environ, LINGBOT_ROOT=str(store.root), PYTHONPATH=str(store.root/'src'),
                   LINGBOT_LEASE_FDS=','.join(map(str, lock.fds)))
        with (folder/'worker.log').open('xb') as log:
            current = subprocess.Popen([sys.executable, '-m', 'eval_platform.runner',
                '--root', str(store.root), '--job', identifier, '--shard', str(number),
                '--gpu', gpu, '--port', str(port)], stdout=log, stderr=log, env=env,
                start_new_session=True, pass_fds=lock.fds)
        (folder/'runner.pid').write_text(str(current.pid))
        while current.poll() is None:
            store.heartbeat(gpu, os.getpid(), identifier)
            parent = store.job(identifier)
            if parent['status']=='cancel_requested' or parent['error']:
                stop_group(current)
                status, error = 'cancelled', None
                break
            time.sleep(2)
        else:
            status = 'completed' if current.returncode==0 else 'failed'
            error = None if status=='completed' else f'Shard {number} exited {current.returncode}; see logs'
    except (OSError, RuntimeError) as exc:
        status, error = 'failed', str(exc)
    finally:
        if current:
            stop_group(current)
        store.finish_shard(identifier, number, status, error)
        store.heartbeat(gpu, os.getpid())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--port', type=int, default=18100)
    parser.add_argument('--job', help='Run only this CLI job and exit when it is terminal')
    args = parser.parse_args()
    gpu = resolve_gpus([args.gpu])[0]
    store = Store(args.root)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    while True:
        store.heartbeat(gpu, os.getpid())
        if args.job and store.job(args.job)['status'] not in ('queued','running','cancel_requested'):
            return
        if (store.state/'maintenance.json').exists() or not store.pending(args.lane,args.job):
            time.sleep(2)
            continue
        try:
            with gpu_lock(store.root, gpu) as lock:
                if (store.state/'maintenance.json').exists():
                    continue
                check_gpu(gpu)
                shard = store.claim_shard(gpu,os.getpid(),args.lane,args.job)
                if shard:
                    execute(store,shard,gpu,args.port,lock)
        except BlockingIOError:
            time.sleep(3)
        except RuntimeError as error:
            print(str(error), flush=True)
            time.sleep(5)


if __name__ == '__main__':
    main()
