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
from .resources import check_gpu, inherited_fds
from .resume import verify_resume
from .planning import report_complete


def execution_report(folder, config, shard=None):
    """Read the runner's assigned output without nesting the global shard plan."""
    reporting_config = {**config, 'shards': []} if shard is not None else config
    return report(folder, reporting_config)


def run(root, identifier, gpu, port, shard=None):
    started=time.monotonic(); timings={}
    store=Store(root); job=store.job(identifier); config=job['config']
    if config.get('history_import'):
        raise ValueError('Imported historical jobs cannot be executed')
    folder=store.run_folder(identifier)
    recovery=verify_resume(root,folder,config,shard) if shard is not None else None
    if shard is not None:
        config=next(s['config'] for s in store.shards(identifier) if s['shard']==shard)
        folder=store.shard_folder(identifier,shard)
    ckpt=store.checkpoint(config['checkpoint_id'])
    leases=inherited_fds()
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
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        XDG_CACHE_HOME=f'{root}/cache',CUDA_HOME=f'{root}/envs/sim',
        LD_LIBRARY_PATH=f'{root}/third_party/oidn-2.3.3.x86_64.linux/lib:'+os.environ.get('LD_LIBRARY_PATH',''),
        PATH=f'{root}/envs/sim/bin:'+os.environ['PATH'], LINGBOT_RENDER_DEVICE='cuda:0')
    if recovery:
        env['LINGBOT_EVAL_SEGMENT']=recovery['segment']
        pointer=json.loads((store.run_folder(identifier)/'active-resume.json').read_text())
        env['LINGBOT_RESUME_MANIFEST']=str(store.run_folder(identifier)/pointer['manifest'])
    if Path('/etc/vulkan/icd.d/nvidia_icd.json').exists():
        env['VK_ICD_FILENAMES']='/etc/vulkan/icd.d/nvidia_icd.json'
    runtime_config=json.loads((root/'configs/runtime.json').read_text())
    if bool(runtime_config.get('release_unused_cuda_cache')) != config.get('release_unused_cuda_cache',False):
        raise ValueError('CUDA allocator configuration differs from the frozen job request')
    if runtime_config.get('release_unused_cuda_cache'):
        env['LINGBOT_RELEASE_UNUSED_CUDA_CACHE']='1'
    base=[str(root/'envs/train/bin/python'),str(root/'scripts/serve_official_checkpoint.py'),
          '--run',str(folder),'--port',str(port)]
    server=worker=None
    def interrupted(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    try:
        loading_started=time.monotonic()
        with (folder/'server.log').open('xb') as log:
            server=subprocess.Popen(base,env=env,cwd=root/'third_party/lingbot-vla-v2-official-be969b8f',stdout=log,stderr=log,pass_fds=leases)
        (folder/'process-segment.json').write_text(json.dumps(dict(gpu_uuid=gpu,server_pid=server.pid,
            port=port,shard=shard,tasks=config['tasks'],settings=config['settings'],seeds=config['seeds'],
            sampling_seed_at_process_start=42,sampling_seed_per_episode=None,
            cpu_threads=8,started_at=time.time(),execution_segment=recovery['segment'] if recovery else 'original',
            prior_rng_restored=False if recovery else None),indent=2))
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
          '--inference-profile','aug16000_h50_execute25_no_video_v1',
          '--expected-execute',str(config['execute']),'--port',str(port),'--protocol-id',config['protocol']]
        # Requests created before fps was configurable used 25 fps; preserve them.
        command.extend(['--video-fps',str(config.get('video_fps',25))])
        if config['video']:command.append('--save-video')
        if config.get('observer_video', True) is False:command.append('--no-observer-video')
        if config.get('fixed_scenes') is not None:
            from .fork2000_plan import write_scene_file
            scene_file,scene_sha=write_scene_file(folder,config['fixed_scenes'])
            command.extend(['--fixed-scenes-file',scene_file,'--fixed-scenes-sha256',scene_sha])
        if config.get('paired_reference'):
            command.extend(['--paired-reference', json.dumps(config['paired_reference'])])
        if config.get('paired_entry'):
            command.extend(['--paired-entry', config['paired_entry']])
        if config.get('scene_replay'):
            command.extend(['--scene-replay', json.dumps(config['scene_replay'])])
        if config.get('paired_camera_policy'):
            command.extend(['--paired-camera-policy',json.dumps(config['paired_camera_policy'])])
        if config.get('timeout_policy'):
            command.extend(['--timeout-policy',json.dumps(config['timeout_policy'])])
        if config.get('expert_check'):command.append('--expert-check')
        if config.get('expert_quota'):
            command.extend(['--expert-quota', json.dumps(config['expert_quota'])])
        if config.get('expert_exception_policy'):
            command.extend(['--expert-exception-policy', config['expert_exception_policy']])
        if config.get('settled_expert_errors'):
            command.extend(['--settled-expert-errors', json.dumps(config['settled_expert_errors'])])
        if config.get('valid_per_group'):
            command.extend(['--valid-per-group',str(config['valid_per_group']),
                            '--max-consecutive-nonvalid',str(config.get('max_consecutive_nonvalid',20))])
        if recovery:command.append('--resume-completed')
        with (folder/'evaluation.log').open('xb') as log:
            worker=subprocess.Popen(command,env={**env,'OPENBLAS_NUM_THREADS':'1'},cwd=root/'third_party/RoboTwin',stdout=log,stderr=log,pass_fds=leases)
        eval_started=time.monotonic()
        code=worker.wait()
        timings['evaluation_wall_seconds']=time.monotonic()-eval_started
        data=execution_report(folder,config,shard)
        (folder/'report.json').write_text(json.dumps(data,indent=2))
        (folder/'scores.csv').write_text(csv_report(data),encoding='utf-8')
        if code not in (0,1) or not report_complete(data):
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
    p.add_argument('--shard',type=int)
    a=p.parse_args();raise SystemExit(run(a.root.resolve(),a.job,a.gpu,a.port,a.shard))
