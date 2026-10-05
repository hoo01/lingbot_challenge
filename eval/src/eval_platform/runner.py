"""One GPU job, with model and simulator in their separate pinned environments."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from .store import Store, digest
from .reports import report, csv_report
from .contracts import write_inference_manifest
from .resources import check_gpu


def run(root, identifier, gpu, port):
    started=time.monotonic(); timings={}
    store=Store(root); job=store.job(identifier); config=job['config']
    if config.get('history_import'):
        raise ValueError('Imported historical jobs cannot be executed')
    ckpt=store.checkpoint(config['checkpoint_id']); folder=store.run_folder(identifier)
    check_gpu(gpu)
    for file in ckpt['files']:
        path=Path(file['path'])
        if path.stat().st_size != file['bytes'] or digest(path) != file['sha256']:
            raise RuntimeError(f'Checkpoint changed after registration: {path.name}')
    if digest(ckpt['norm']) != ckpt['norm_sha256']: raise RuntimeError('Normalization changed')
    for record in ckpt.get('metadata_files', []):
        if digest(record['path']) != record['sha256']:
            raise RuntimeError('Checkpoint metadata changed after registration')
    write_inference_manifest(root, ckpt['checkpoint'], ckpt['norm'], folder, ckpt['kind'])
    timings['checkpoint_verification_seconds']=time.monotonic()-started
    (folder/'checkpoint.json').write_text(json.dumps(ckpt,indent=2))
    env=dict(os.environ, LINGBOT_ROOT=str(root), CUDA_VISIBLE_DEVICES=gpu,
        PYTHONPATH=f'{root}/src:{root}/third_party/lingbot-vla-v2-official-be969b8f',
        QWEN3VL_PATH=f'{root}/models/Qwen3-VL-4B-Instruct', PYTHONNOUSERSITE='1',
        PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        XDG_CACHE_HOME=f'{root}/cache',CUDA_HOME=f'{root}/envs/sim',
        LD_LIBRARY_PATH=f'{root}/third_party/oidn-2.3.3.x86_64.linux/lib:'+os.environ.get('LD_LIBRARY_PATH',''),
        PATH=f'{root}/envs/sim/bin:'+os.environ['PATH'], LINGBOT_RENDER_DEVICE='cuda:0')
    if Path('/etc/vulkan/icd.d/nvidia_icd.json').exists():
        env['VK_ICD_FILENAMES']='/etc/vulkan/icd.d/nvidia_icd.json'
    base=[str(root/'envs/train/bin/python'),str(root/'scripts/serve_official_checkpoint.py'),
          '--run',str(folder),'--port',str(port)]
    server=worker=None
    def interrupted(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    try:
        loading_started=time.monotonic()
        with (folder/'server.log').open('xb') as log:
            server=subprocess.Popen(base,env=env,cwd=root/'third_party/lingbot-vla-v2-official-be969b8f',stdout=log,stderr=log)
        deadline=time.monotonic()+240
        while True:
            if server.poll() is not None: raise RuntimeError('Policy server exited; see server.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as response:
                    if response.status==200:break
            except OSError: pass
            if time.monotonic()>deadline: raise TimeoutError('Policy server startup exceeded 240s')
            time.sleep(1)
        timings['policy_startup_seconds']=time.monotonic()-loading_started
        command=[str(root/'envs/sim/bin/python'),str(root/'scripts/eval_isolated.py'),
          '--output',str(folder/'evaluation'),'--tasks',*config['tasks'],'--settings',*config['settings'],
          '--seeds',*[str(x) for x in config['seeds']], '--expected-checkpoint',ckpt['checkpoint'],
          '--expected-precision',config['precision'],'--expected-vision-device','cuda',
          '--expected-embedding-device','cuda','--expected-moe-backend','official',
          '--inference-profile','official_defaults_v1',
          '--expected-execute',str(config['execute']),'--port',str(port),'--protocol-id',config['protocol']]
        # Requests created before fps was configurable used 25 fps; preserve them.
        command.extend(['--video-fps',str(config.get('video_fps',25))])
        if config['video']:command.append('--save-video')
        with (folder/'evaluation.log').open('xb') as log:
            worker=subprocess.Popen(command,env={**env,'OPENBLAS_NUM_THREADS':'1'},cwd=root/'third_party/RoboTwin',stdout=log,stderr=log)
        eval_started=time.monotonic()
        code=worker.wait()
        timings['evaluation_wall_seconds']=time.monotonic()-eval_started
        data=report(folder,config)
        (folder/'report.json').write_text(json.dumps(data,indent=2))
        (folder/'scores.csv').write_text(csv_report(data),encoding='utf-8')
        if code not in (0,1) or data['attempted']!=data['planned'] or data['execution_errors'] or data['record_error']:
            raise RuntimeError(f'Evaluation interrupted (exit={code}); preserve logs and episode records')
        return 0
    finally:
        for process in (worker,server):
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        timings['total_runner_seconds']=time.monotonic()-started
        (folder/'timing.json').write_text(json.dumps(timings,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',required=True);p.add_argument('--gpu',required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();raise SystemExit(run(a.root.resolve(),a.job,a.gpu,a.port))
