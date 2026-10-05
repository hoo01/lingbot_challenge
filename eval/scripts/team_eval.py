"""Command line entry sharing the web queue and fixed inference contract."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(os.environ['LINGBOT_ROOT']).resolve()
sys.path.insert(0, str(ROOT / 'src'))
from eval_platform.store import Store, digest, now
from eval_platform.resources import gpu_lock, check_gpu
from eval_platform.contracts import source_lock


def register_base(store):
    """Original-base registration is separate from clean SFT handoffs."""
    source, lock = source_lock(store.root)
    if digest(source / 'assets/norm_stats/robotwin.json') != lock['normalization_sha256']:
        raise ValueError('Original upstream normalization differs')
    from prepare_official_base import prepare
    name = 'original-base-' + uuid.uuid4().hex[:12]
    prepare(store.root, name)
    run = store.root / 'runs' / name
    manifest = json.loads((run / 'official-defaults-manifest.json').read_text())
    checkpoint = Path(manifest['checkpoint'])
    shards = sorted(checkpoint.glob('*.safetensors'))
    if len(shards) != 6:
        raise ValueError('The pinned original base requires six safetensors shards')
    identifier = uuid.uuid4().hex[:12]
    spec = dict(id=identifier, name='Original pretrained LingBot-VLA v2',
        kind='original_pretrained_base_official_inference_defaults',
        checkpoint=str(checkpoint), cli=str(run / 'lingbotvla_cli.yaml'),
        norm=manifest['normalization'], norm_sha256=manifest['normalization_sha256'],
        files=[dict(path=str(f), bytes=f.stat().st_size, sha256=digest(f)) for f in shards],
        metadata_files=[dict(path=str(run/'lingbotvla_cli.yaml'), sha256=digest(run/'lingbotvla_cli.yaml'))],
        notes='Raw pretrained base, upstream normalization; no RoboTwin SFT', created=now())
    with store.connect() as db:
        db.execute('INSERT INTO checkpoints VALUES (?,?)', (identifier, json.dumps(spec)))
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--base', action='store_true')
    choice.add_argument('--checkpoint', help='Complete merged HF directory inside the runtime')
    choice.add_argument('--checkpoint-id', help='Previously registered model')
    parser.add_argument('--cli')
    parser.add_argument('--norm')
    parser.add_argument('--tasks', nargs='+', default=['click_bell'])
    parser.add_argument('--all-tasks', action='store_true')
    parser.add_argument('--settings', nargs='+', default=['demo_clean', 'demo_randomized'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[100010])
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--title', default='Team development evaluation')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--port', type=int, default=18100)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--enqueue', action='store_true', help='Only queue for the UI worker')
    mode.add_argument('--register-only', action='store_true', help='Register the model without creating a job or using the GPU')
    args = parser.parse_args()
    store = Store(ROOT)

    def register():
        if args.base:
            model = register_base(store)
        elif args.checkpoint:
            if not args.cli or not args.norm:
                parser.error('--checkpoint requires --cli and --norm')
            model = store.register(args.title, args.checkpoint, args.cli, args.norm)
        else:
            model = store.checkpoint(args.checkpoint_id)
        return model

    def submit():
        model = register()
        tasks = store.tasks() if args.all_tasks else args.tasks
        if args.all_tasks and len(tasks) != 50:
            raise ValueError('Expected all 50 RoboTwin tasks')
        return store.submit(args.title, dict(checkpoint_id=model['id'], tasks=tasks,
            settings=args.settings, seeds=args.seeds, video=args.video))

    if args.register_only:
        model = register()
        print(json.dumps(dict(checkpoint_id=model['id'], name=model['name'], status='registered'), ensure_ascii=False))
        return 0
    if args.enqueue:
        job = submit()
        print(json.dumps(dict(job_id=job['id'], server=job['config']['server_id'], status='queued', output=str(store.run_folder(job['id'])))))
        return 0
    with gpu_lock(ROOT) as lock:
        check_gpu(args.gpu)
        if any(j['status'] in ('running', 'cancel_requested') for j in store.jobs()):
            raise RuntimeError('Inspect existing running records before starting another evaluation')
        job = submit()
        folder = store.run_folder(job['id'])
        with store.connect() as db:
            db.execute("UPDATE jobs SET status='running',gpu=?,pid=?,updated=? WHERE id=?",
                       (args.gpu, os.getpid(), now(), job['id']))
        print(json.dumps(dict(job_id=job['id'], server=job['config']['server_id'], output=str(folder))), flush=True)
        from eval_platform.worker import stop_group
        import signal
        process = None
        def interrupted(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, interrupted)
        try:
            env = dict(os.environ, LINGBOT_ROOT=str(ROOT), PYTHONPATH=str(ROOT/'src'))
            with (folder / 'worker.log').open('xb') as log:
                process = subprocess.Popen([sys.executable, '-m', 'eval_platform.runner',
                    '--root', str(ROOT), '--job', job['id'], '--gpu', args.gpu, '--port', str(args.port)],
                    env=env, stdout=log, stderr=log, start_new_session=True, pass_fds=(lock.fileno(),))
            (folder / 'runner.pid').write_text(str(process.pid))
            while process.poll() is None:
                if store.job(job['id'])['status'] == 'cancel_requested':
                    stop_group(process)
                    store.update(job['id'], 'cancelled')
                    return 130
                time.sleep(1)
            code = process.returncode
            store.update(job['id'], 'completed' if code == 0 else 'failed',
                         None if code == 0 else 'See logs; no automatic retry')
            return code
        except BaseException:
            store.update(job['id'], 'interrupted', 'CLI interrupted; records retained')
            raise
        finally:
            if process:
                stop_group(process)


if __name__ == '__main__':
    raise SystemExit(main())
