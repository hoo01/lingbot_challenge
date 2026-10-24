"""One durable model and serial native simulator, bounded to this16000 root."""
import hashlib,json,math,os,signal,socket,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from eval_platform.resources import gpu_lock,check_gpu

def save(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)

def identity(p):
    proc=Path('/proc')/str(p.pid)
    return dict(pid=p.pid,start_ticks=proc.joinpath('stat').read_text().split()[21],command=proc.joinpath('cmdline').read_bytes().decode().replace('\0',' '))

def stopped():return (ROOT/'user-stop.json').exists() or (ROOT.parent.parent/'user-stop.json').exists()

def main():
    scope=json.loads((ROOT/'artifacts/eval2000-scope.json').read_text());run=ROOT/'run';run.mkdir(exist_ok=True)
    assert not (ROOT/'artifacts/runner-started.json').exists()
    save(ROOT/'artifacts/runner-started.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],time=time.time(),scope=scope['shard_id']))
    gpu=scope['physical_gpu'];env=os.environ.copy()
    env.update(LINGBOT_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'third_party/lingbot-vla-v2-official-be969b8f'),
        QWEN3VL_PATH=str(ROOT/'models/Qwen3-VL-4B-Instruct'),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        XDG_CACHE_HOME=str(ROOT/'cache'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'cache/inductor'),TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch-extensions'))
    children=[]
    def shutdown(*_):raise SystemExit(130)
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    with gpu_lock(ROOT,gpu) as lease:
        check_gpu(gpu);assert not stopped()
        raw=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        assert not raw.strip(),'GPU has existing compute applications'
        env['LINGBOT_LEASE_FDS']=','.join(map(str,lease.fds))
        def spawn(label,cmd,environment,cwd):
            with (run/(label+'.log')).open('xb') as log:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=environment,cwd=cwd,pass_fds=lease.fds)
            children.append(p);save(run/(label+'-started.json'),identity(p));return p
        try:
            server=spawn('model',[str(ROOT/'envs/train/bin/python'),'-B',str(ROOT/'scripts/serve_official_checkpoint.py'),'--run',str(run),'--port',str(scope['policy_port'])],env,ROOT)
            deadline=time.monotonic()+1200
            while time.monotonic()<deadline:
                assert server.poll() is None,'Model startup failed; inspect original model.log'
                if stopped():raise SystemExit(130)
                with socket.socket() as sock:
                    if (run/'server-runtime.json').exists() and sock.connect_ex(('127.0.0.1',scope['policy_port']))==0:break
                time.sleep(2)
            else:raise TimeoutError('Model ready1200s; preserve original root')
            metadata=json.loads((run/'server-runtime.json').read_text());assert metadata['learned_parameter_precision_check']['exact_saved_fp32_values']
            assert str(metadata['cuda_physical_uuid']).removeprefix('GPU-').lower()==gpu.removeprefix('GPU-').lower()
            simenv=env.copy();simenv.update(CUDA_HOME=str(ROOT/'envs/sim'),LINGBOT_RENDER_DEVICE=scope['render_device'],
                VK_ICD_FILENAMES=scope['vulkan_icd'],LD_LIBRARY_PATH=scope['simulation_ld_library_path'])
            simenv.update(scope['simulation_extra_env'])
            # Model CUDA environment deliberately receives no Vulkan/EGL overrides.
            command=[str(ROOT/'envs/sim/bin/python'),'-B',str(ROOT/'scripts/eval_isolated.py'),'--output',str(run/'evaluation'),
                '--tasks',*scope['tasks'],'--settings',*scope['settings'],'--seeds',*map(str,scope['seeds']),
                '--no-observer-video','--expected-checkpoint',scope['checkpoint'],'--expected-precision','fp32',
                '--expected-execute','25','--port',str(scope['policy_port']),'--inference-profile','aug16000_h50_execute25_no_video_v1',
                '--fixed-scenes-file',str(ROOT/'artifacts/fixed-scenes.json'),'--fixed-scenes-sha256',scope['fixed_scenes_sha256'],
                '--timeout-policy',json.dumps(scope['timeout_policy']),'--protocol-id',scope['protocol_id']]
            sim=spawn('simulation',command,simenv,ROOT/'third_party/RoboTwin')
            save(ROOT/'artifacts/formal-dispatch.json',dict(scope=scope,model=identity(server),simulator=identity(sim),time=time.time(),formal_startup_verified=False))
            started=False
            while sim.poll() is None:
                if stopped():raise SystemExit(130)
                assert server.poll() is None,'Model exited during evaluation'
                if not started:
                    for path in (run/'evaluation').glob('*/*/*/trajectory.jsonl'):
                        if not path.stat().st_size:continue
                        row=json.loads(path.open().readline());actions=row['predicted_actions']
                        assert row['chunk']==0 and row['action_count_before']==0 and 0<row['action_count_after']<=25
                        assert len(actions)==50 and all(len(a)==14 and all(math.isfinite(v) for v in a) for a in actions)
                        initial=json.loads((path.parent/'initial-inputs.json').read_text());assert len(initial['initial_state'])==14 and len(initial['cameras'])==3
                        assert all(c['dtype']=='uint8' and c['shape'][-1]==3 for c in initial['cameras'].values())
                        save(ROOT/'artifacts/started.json',dict(passed=True,source_step=16000,revision=scope['model_revision'],protocol_sha256=scope['protocol_sha256'],
                            physical_gpu=gpu,render_device=scope['render_device'],exact_fp32_parameters=1672,persistent_buffers=36,finite_h50=True,actual_first_prefix=row['action_count_after'],
                            own_norm_sha256=scope['norm_sha256'],video=False,observer_video=False,first_scene=dict(task=initial['task'],setting=initial['setting'],actual_seed=initial['seed']),time=time.time()))
                        started=True;break
                time.sleep(2)
            assert sim.returncode==0,'Simulator stopped; exact prefix/partial preserved for diagnosis'
            assert started,'No actual formal trajectory verification'
            # Full CPU completion verification is performed by the separate immutable auditor.
            save(ROOT/'artifacts/finished-unverified.json',dict(time=time.time(),returncode=sim.returncode,audit_required=True))
        finally:
            for p in reversed(children):
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
    # Audit only after model/simulator have exited and released their lease.
    audit_env=dict(os.environ,LINGBOT_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src'))
    with (run/'completion-audit.log').open('xb') as log:
        subprocess.run([str(ROOT/'envs/train/bin/python'),'-B',str(ROOT/'scripts/audit_shard16000.py')],env=audit_env,stdout=log,stderr=log,check=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(ROOT/'artifacts/runner-fatal.json',dict(time=time.time(),error=repr(e),prefix_and_partial_preserved=True))
        raise
