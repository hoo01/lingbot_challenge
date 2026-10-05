"""Single GPU worker: acquire the shared GPU lock before claiming a job."""
import argparse
import os
import signal
import subprocess
import sys
import time
from .store import Store
from .resources import gpu_lock, check_gpu


def stop_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--port', type=int, default=18100)
    args = parser.parse_args()
    store = Store(args.root)
    current = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    while True:
        store.heartbeat(args.gpu, os.getpid())
        if (store.state / 'maintenance.json').exists():
            time.sleep(3)
            continue
        try:
            with gpu_lock(store.root) as lock:
                if any(j['status'] in ('running', 'cancel_requested') for j in store.jobs()):
                    print('Existing running record: inspect processes and logs before recovery', flush=True)
                    time.sleep(10)
                    continue
                if not any(j['status']=='queued' for j in store.jobs()):
                    time.sleep(3)
                    continue
                check_gpu(args.gpu)
                job = store.claim(args.gpu, os.getpid())
                if not job:
                    continue
                folder = store.run_folder(job['id'])
                try:
                    with (folder / 'worker.log').open('ab') as log:
                        current = subprocess.Popen([sys.executable, '-m', 'eval_platform.runner',
                            '--root', str(store.root), '--job', job['id'], '--gpu', args.gpu,
                            '--port', str(args.port)], stdout=log, stderr=log,
                            start_new_session=True, pass_fds=(lock.fileno(),))
                    (folder / 'runner.pid').write_text(str(current.pid))
                    while current.poll() is None:
                        store.heartbeat(args.gpu, os.getpid(), job['id'])
                        if store.job(job['id'])['status'] == 'cancel_requested':
                            stop_group(current)
                            break
                        time.sleep(3)
                    cancelled = store.job(job['id'])['status'] == 'cancel_requested'
                    store.update(job['id'], 'cancelled' if cancelled else
                                 'completed' if current.returncode == 0 else 'failed',
                                 None if current.returncode == 0 or cancelled else 'See logs; no automatic retry')
                except BaseException:
                    store.update(job['id'], 'interrupted', 'Worker stopped; inspect logs and preserve records')
                    raise
                finally:
                    if current:
                        stop_group(current)
                    current = None
        except BlockingIOError:
            time.sleep(5)
        except RuntimeError as error:
            print(str(error), flush=True)
            time.sleep(10)


if __name__ == '__main__':
    main()
