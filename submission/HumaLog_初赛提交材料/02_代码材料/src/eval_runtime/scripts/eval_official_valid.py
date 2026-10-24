"""One resident official server: fixed scan, then outcome-independent valid fill."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from base50_common import read_records,read_jsonl,next_supplement,quotas,commit_attempt
from eval_isolated import run_worker
from official_evaluation_common import load_policy,validate_resume,load_recording,recording_arguments

ROOT=Path(os.environ.get('LINGBOT_ROOT','/data/lingbotvla'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-manifest',type=Path)
    a=p.parse_args();run=a.run;out=a.output
    if out.exists() and not a.resume_manifest:raise FileExistsError('Existing scans require an explicit preserved-prefix and RNG restart manifest')
    lock=(run/'official-coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    restart=validate_resume(run,out,a.resume_manifest) if a.resume_manifest else None
    checkpoint=str(run/'checkpoints/global_step_0/hf_ckpt')
    profile=['--expected-checkpoint',checkpoint,'--expected-precision','fp32','--expected-moe-backend','official',
             '--expected-vision-device','cuda','--expected-embedding-device','cuda','--inference-profile','official_defaults_v1']
    policy=load_policy(ROOT);target=policy['valid_per_group'];total=50*2*target
    recording=load_recording(ROOT);video_args=recording_arguments(recording)
    if restart:
        out.mkdir(exist_ok=True)
        policy_path=out/'valid-completion-policy.json'
        if policy_path.exists():
            assert json.loads(policy_path.read_text())==policy, 'Archive previous quota policy before an authorized change'
        else:policy_path.write_text(json.dumps(policy,indent=2))
    def state(stage,**fields):
        value={'stage':stage,'time':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'target_valid':total,
               'valid_per_group':target,'execution_segment':os.environ.get('LINGBOT_EVAL_SEGMENT','original'),
               'recording':recording,**fields}
        temp=run/'queue-state.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(run/'queue-state.json')
        with (run/'queue-events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    child=None
    try:
        state('initial_fixed_scan',resume_manifest=str(a.resume_manifest) if restart else None)
        command=[sys.executable,str(ROOT/'scripts/eval_isolated.py'),'--output',str(out),*profile,
                 '--all-tasks','--seeds','100010','100011',*video_args,'--protocol-id',policy['original_protocol']]
        if restart:command.append('--resume-completed')
        child=subprocess.Popen(command)
        code=child.wait()
        if code not in (0,1):raise RuntimeError(f'Initial scan failed, exit={code}; preserve evidence and retry same seed after diagnosis')
        protocol=json.loads((out/'protocol.json').read_text())
        assert protocol['protocol']==policy['original_protocol']
        (out/'valid-completion-policy.json').write_text(json.dumps(policy,indent=2))
        rows=read_records(out);scene=next_supplement(rows,protocol,policy)
        while scene is not None:
            task,setting,seed=scene
            attempt=out/'valid-fill/attempts'/task/setting/str(seed)
            if attempt.exists():raise FileExistsError(attempt)
            state('evaluating_supplement',attempt=str(attempt),task=task,setting=setting,seed=seed)
            command=[sys.executable,str(ROOT/'scripts/eval_dev_suite.py'),'--output',str(attempt),'--one-episode',
                     '--tasks',task,'--settings',setting,'--seeds',str(seed),*video_args,*profile,
                     '--protocol-id',policy['protocol']]
            code=run_worker(command,attempt/'worker-progress.json',900,240)
            if read_jsonl(attempt/'episodes.jsonl'):commit_attempt(out,attempt,scene)
            if code!=0:raise RuntimeError(f'Supplement failed, exit={code}; do not replace seed {scene}')
            rows=read_records(out);scene=next_supplement(rows,protocol,policy)
        groups=quotas(rows,protocol,target)
        assert len(groups)==100 and all(g['valid']==target for g in groups)
        result={'protocol':policy['protocol'],'groups':groups,'valid':total,'attempted':len(rows),
                'successes':sum(bool(r['success']) for r in rows if r['status']=='policy_completed'),
                'initialization_failures':sum(r['status']=='setup_error' for r in rows),
                'selection':f'First {target} valid scenes in the fixed scan and ascending supplements; independent of policy success',
                'completed_at':datetime.now(timezone.utc).isoformat()}
        with (out/'valid-scenes-manifest.json').open('x') as stream:json.dump(result,stream,indent=2)
        state('completed',valid=total,groups_completed=100,attempted=len(rows))
    except BaseException as error:
        state('interrupted' if isinstance(error,SystemExit) else 'failed',error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':main()
