"""Fixed research profile and valid scene quotas, independent of policy outcomes."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from eval_isolated import run_worker

ROOT=Path('/data/lingbotvla')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--execute',type=int,choices=[10,25,50],default=50)
    a=p.parse_args(); suite=json.loads(a.suite.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]; scenes=[]; server=None
    started=time.monotonic()
    ref=json.loads(a.reference.read_text()) if a.reference else None
    if ref:
        assert ref['suite']==suite and ref['complete']
    def state(event,**fields):
        value={'event':event,'time':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
               'elapsed_seconds':time.monotonic()-started,'valid':len(scenes),'attempted':len(rows),**fields}
        tmp=a.output/'status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(a.output/'status.json')
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    lock=(ROOT/'runs/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    os.environ.update(PYTHONPATH=f'{ROOT}/src:{ROOT}/third_party/lingbot-vla-v2',HF_HUB_OFFLINE='1',
        QWEN3VL_PATH=f'{ROOT}/models/Qwen3-VL-4B-Instruct',CUDA_VISIBLE_DEVICES='0',
        PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
        XDG_CACHE_HOME=f'{ROOT}/cache',CUDA_HOME=f'{ROOT}/envs/sim',OPENBLAS_NUM_THREADS='1')
    os.environ['PATH']=f'{ROOT}/envs/sim/bin:'+os.environ['PATH']
    os.environ['LD_LIBRARY_PATH']=f'{ROOT}/third_party/oidn-2.3.3.x86_64.linux/lib:'+os.environ.get('LD_LIBRARY_PATH','')
    os.chdir(ROOT/'third_party/RoboTwin')
    try:
        state('starting_server',checkpoint=str(a.checkpoint),execute=a.execute)
        with (a.output/'server.log').open('x') as log:
            server=subprocess.Popen([str(ROOT/'envs/train/bin/python'),str(ROOT/'scripts/serve_checkpoint.py'),
                '--checkpoint',str(a.checkpoint),'--execute',str(a.execute),'--precision','fp32',
                '--moe-backend','torch_reference','--vision-device','cpu','--embedding-device','cpu',
                '--profile','research4_paired_v1','--port','18016'],stdout=log,stderr=subprocess.STDOUT,
                cwd=ROOT/'third_party/lingbot-vla-v2')
        ready=False
        for _ in range(180):
            if server.poll() is not None:raise RuntimeError('Research server failed; see server.log')
            try:
                urllib.request.urlopen('http://127.0.0.1:18016/healthz',timeout=1).close();ready=True;break
            except OSError:time.sleep(1)
        if not ready:raise TimeoutError('Research server startup')
        for task in suite['tasks']:
            for setting in suite['settings']:
                valid=0;seed=suite['seed_start'];consecutive=0
                group_ref=[r for r in ref['scenes'] if (r['task'],r['setting'])==(task,setting)] if ref else None
                if group_ref is not None:assert len(group_ref)==suite['valid_per_group']
                while valid<suite['valid_per_group']:
                    if group_ref is not None:seed=group_ref[valid]['seed']
                    attempt=a.output/'attempts'/task/setting/str(seed)
                    state('evaluating',task=task,setting=setting,seed=seed,attempt=str(attempt))
                    command=[str(ROOT/'envs/sim/bin/python'),str(ROOT/'scripts/eval_dev_suite.py'),
                        '--output',str(attempt),'--one-episode','--tasks',task,'--settings',setting,'--seeds',str(seed),
                        '--expected-checkpoint',str(a.checkpoint),'--expected-precision','fp32',
                        '--expected-vision-device','cpu','--expected-embedding-device','cpu',
                        '--expected-moe-backend','torch_reference','--expected-execute',str(a.execute),
                        '--inference-profile','research4_paired_v1','--protocol-id',suite['id'],'--port','18016']
                    if valid<suite['video_valid_per_group']:command.append('--save-video')
                    code=run_worker(command,attempt/'worker-progress.json',900,240)
                    episode_path=attempt/'episodes.jsonl'
                    if not episode_path.exists():raise RuntimeError(f'Worker exit {code}, no episode record: {task}/{setting}/{seed}')
                    episode_rows=[json.loads(line) for line in episode_path.read_text().splitlines()]
                    assert len(episode_rows)==1
                    row=episode_rows[0];row['attempt_path']=str(attempt)
                    rows.append(row)
                    with (a.output/'episodes.jsonl').open('a') as f:
                        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
                    if code!=0:raise RuntimeError(f'Worker exited {code}; retry this seed after preserving and diagnosing failure')
                    if row['status']=='setup_error':
                        assert 'UnStableError' in row['error']
                        if group_ref is not None:raise RuntimeError('Reference scene initialization changed; stop paired comparison')
                        consecutive+=1
                        if consecutive>=20:raise RuntimeError('20 consecutive setup failures; diagnosis required')
                    elif row['status']=='policy_completed':
                        folder=attempt/task/setting/str(seed)
                        initial={suffix:hashlib.sha256((folder/f'chunk-000{suffix}.png').read_bytes()).hexdigest()
                                 for suffix in ('','-left-wrist','-right-wrist')}
                        scene={'task':task,'setting':setting,'seed':seed,'instruction':row['instruction'],'initial_images':initial}
                        if group_ref is not None:assert scene==group_ref[valid], 'Initial camera/instruction mismatch'
                        scenes.append(scene);valid+=1;consecutive=0
                    else:raise RuntimeError(f'Execution error: {row["status"]}')
                    seed+=1
        target=len(suite['tasks'])*len(suite['settings'])*suite['valid_per_group']
        assert len(scenes)==target
        groups=[]
        for task in suite['tasks']:
            for setting in suite['settings']:
                subset=[r for r in rows if r['task']==task and r['setting']==setting]
                groups.append({'task':task,'setting':setting,'successes':sum(bool(r['success']) for r in subset if r['status']=='policy_completed'),
                    'valid':sum(r['status']=='policy_completed' for r in subset),'initialization_failures':sum(r['status']=='setup_error' for r in subset)})
        summary={'complete':True,'valid':target,'attempted':len(rows),'successes':sum(g['successes'] for g in groups),
                 'initialization_failures':sum(g['initialization_failures'] for g in groups),'groups':groups,
                 'wall_seconds':time.monotonic()-started,'checkpoint':str(a.checkpoint),'execute':a.execute}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        (a.output/'valid-scenes-manifest.json').write_text(json.dumps({'complete':True,'suite':suite,'scenes':scenes},indent=2))
        state('complete',**summary)
    except BaseException as e:
        state('failed',error=repr(e));raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=='__main__':main()
