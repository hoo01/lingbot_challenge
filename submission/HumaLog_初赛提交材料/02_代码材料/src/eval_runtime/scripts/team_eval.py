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
from eval_platform.resources import resolve_gpus, gpu_port
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
    parser.add_argument('--expert-check', action='store_true', help='Screen the fixed planned seeds with the official expert')
    parser.add_argument('--title', default='Team development evaluation')
    parser.add_argument('--gpu', help='Use one selected GPU')
    parser.add_argument('--gpus', help='GPU indices or UUIDs separated by commas')
    parser.add_argument('--parallelism',type=int,choices=(1,2))
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
            settings=args.settings, seeds=args.seeds, video=args.video, expert_check=args.expert_check,
            parallelism=args.parallelism or len(gpus),gpu_uuids=gpus))

    if args.register_only:
        model = register()
        print(json.dumps(dict(checkpoint_id=model['id'], name=model['name'], status='registered'), ensure_ascii=False))
        return 0
    from eval_platform.servers import load_catalog,select_server
    _,server=select_server(load_catalog())
    selectors=([args.gpu] if args.gpu else args.gpus.split(',') if args.gpus else server.get('gpus',[server['gpu']]))
    if args.parallelism and args.parallelism>len(selectors):
        parser.error('parallelism exceeds the selected GPU count')
    gpus=resolve_gpus(selectors)
    job=submit()
    folder=store.run_folder(job['id'])
    print(json.dumps(dict(job_id=job['id'],server=job['config']['server_id'],status='queued',output=str(folder))),flush=True)
    if args.enqueue:
        return 0
    # CLI and web workers share atomic shard claims and physical GPU leases.
    # CLI helpers are restricted to this job; the persistent workers may also
    # claim it, but no shard or GPU can be claimed twice.
    from eval_platform.worker import stop_group
    import signal
    processes=[]
    def interrupted(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        env=dict(os.environ,LINGBOT_ROOT=str(ROOT),PYTHONPATH=str(ROOT/'src'))
        for lane,gpu in enumerate(gpus[:job['config']['parallelism']]):
            with (folder/f'cli-worker-{lane}.log').open('xb') as log:
                processes.append(subprocess.Popen([sys.executable,'-m','eval_platform.worker',
                    '--root',str(ROOT),'--gpu',gpu,'--lane',str(lane),
                    '--port',str(gpu_port(gpu,args.port)),'--job',job['id']],env=env,
                    stdout=log,stderr=log,start_new_session=True))
        while store.job(job['id'])['status'] in ('queued','running','cancel_requested'):
            if all(p.poll() is not None for p in processes):
                raise RuntimeError('CLI workers exited; inspect logs')
            time.sleep(2)
        final=store.job(job['id'])
        print(json.dumps(dict(job_id=job['id'],status=final['status'],output=str(folder))),flush=True)
        return 0 if final['status']=='completed' else 130 if final['status']=='cancelled' else 2
    except BaseException:
        store.cancel(job['id'])
        raise
    finally:
        for process in processes:
            stop_group(process)


if __name__ == '__main__':
    raise SystemExit(main())
