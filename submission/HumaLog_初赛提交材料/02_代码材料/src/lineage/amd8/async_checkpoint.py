"""Immutable full CPU snapshots; disk/SSH work never reads live training state."""
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
import torch
import torch.distributed as dist
import checkpoint as ckpt
from policy import ROOT,WORLD,atomic_json,sha


def freeze(value):
    if isinstance(value,torch.Tensor):
        return value.detach().to(device='cpu',copy=True)
    if isinstance(value,dict):return {key:freeze(v) for key,v in value.items()}
    if isinstance(value,list):return [freeze(v) for v in value]
    if isinstance(value,tuple):return tuple(freeze(v) for v in value)
    return copy.deepcopy(value)


def pack_copy(value):
    if isinstance(value,torch.Tensor):
        return dict(tensor=freeze(ckpt.local(value)),dtensor=hasattr(value,'placements'),global_shape=list(value.shape))
    return copy.deepcopy(value)


def freeze_training_state(model,optimizer,step,identity,args,loader,generator,world):
    params=dict(model.named_parameters());names={id(p):name for name,p in params.items()}
    data=dict(step=step,identity=copy.deepcopy(identity),loader=freeze(loader.state_dict()),
              generator=freeze(generator.get_state()),epoch=loader.sampler.epoch)
    state=dict(step=step,world=world,microbatch=args.batch,accum=args.accum,identity=copy.deepcopy(identity),
        rng=freeze(ckpt.rng_state()),schedule=dict(total=10000,next_update=step+1),
        parameters={name:pack_copy(p) for name,p in params.items()},
        buffers={name:pack_copy(b) for name,b in ckpt.persistent_buffers(model).items()},
        optimizer={name:{k:pack_copy(v) for k,v in optimizer.state.get(p,{}).items()} for name,p in params.items()},
        groups=[{**{k:freeze(v) for k,v in g.items() if k!='params'},'params':[names[id(p)] for p in g['params']]}
                for g in optimizer.param_groups])
    return state,data


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):
        assert value.device.type=='cpu' and not hasattr(value,'placements')
        return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(map(tensor_bytes,value))
    return 0


def check_pending(folder):
    if (ROOT/'user-stop.json').exists():raise RuntimeError('Explicit human stop; preserve snapshot/spool')
    failures=sorted(folder.glob('async-error-rank*.json'))
    if failures:raise RuntimeError('Actual async archive failure: '+str(failures[0]))


def wait_until(predicate,folder,message):
    deadline=time.monotonic()+8*3600
    while not predicate():
        check_pending(folder)
        if time.monotonic()>deadline:raise TimeoutError(message)
        time.sleep(1)


def persist_snapshot(payload,data,folder,spool,rank,run):
    """One disk writer experiment-wide bounds workspace use; training is independent."""
    lock=(ROOT/'artifacts/async-checkpoint-io.lock').open('a+')
    wait_until(lambda:acquire(lock),folder,'Timed out waiting for this task checkpoint IO lease')
    started=time.time();total=tensor_bytes(payload)+tensor_bytes(data)
    try:
        # Worst-case raw plus incompressible gzip, with room for metadata and headers.
        assert shutil.disk_usage(ROOT).free>2*total+2*2**30,'Preserve CPU snapshot: insufficient spool reserve'
        data_path=spool/f'data_rank{rank}.pt';raw=spool/f'rank{rank}.uncompressed.pt';compressed=spool/f'rank{rank}.pt'
        assert not any(p.exists() for p in (data_path,raw,compressed)),'Never overwrite a partial or complete snapshot'
        ckpt.write_tensor_file(data_path,data)
        assert ckpt.equal(data,torch.load(data_path,map_location='cpu',weights_only=True))
        data_proof=dict(step=payload['step'],bytes=data_path.stat().st_size,sha256=sha(data_path),readback_equal=True)
        atomic_json(folder/f'data_rank{rank}.json',data_proof)
        ckpt.write_tensor_file(raw,payload)
        restored=torch.load(raw,map_location='cpu',weights_only=True,mmap=True)
        assert ckpt.equal(payload,restored),'Serialized immutable snapshot differs'
        del restored;gc.collect()
        model_proof=dict(step=payload['step'],readback_equal=True,**ckpt.compress_state(raw,compressed))
        atomic_json(folder/f'rank{rank}.json',model_proof)
        relative=str(folder.relative_to(ROOT))
        for path,proof in ((data_path,data_proof),(compressed,model_proof)):
            check_pending(folder);ckpt.request('upload',relative+'/'+path.name,proof);ckpt.release_spool(path)
        current=run/f'progress_rank{rank}.json'
        observed=json.loads(current.read_text())['step'] if current.exists() else payload['step']
        proof=dict(passed=True,step=payload['step'],rank=rank,started=started,finished=time.time(),
            latest_optimizer_step=observed,updates_advanced_during_archive=observed>payload['step'],
            immutable_cpu_snapshot=True,writer_touched_live_model_or_optimizer=False,
            bytes=model_proof['bytes'],readback_equal=True)
        atomic_json(folder/f'async-archived-rank{rank}.json',proof)
        return proof
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def acquire(lock):
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
    except BlockingIOError:return False


def finalize(folder,run,identity,step):
    wait_until(lambda:all((folder/f'async-archived-rank{rank}.json').exists() for rank in range(WORLD)),
        folder,'Timed out waiting for all exact snapshot upload receipts')
    receipts={f'{prefix}{rank}':json.loads((folder/f'{prefix}{rank}.json').read_text())
              for prefix in ('rank','data_rank') for rank in range(WORLD)}
    assert len(receipts)==2*WORLD and all(p['readback_equal'] and p['step']==step for p in receipts.values())
    for name in ('norm_stats.json','norm_stats.provenance.json','robotwin_relative.yaml',
                 'sampling-binding.json','source-binding.json','asset-binding.json'):
        shutil.copy2(ROOT/'artifacts'/name,folder/name)
    for name in ('config.json','model_config.json','lingbotvla_cli.yaml'):shutil.copy2(run/name,folder/name)
    atomic_json(folder/'complete.json',dict(step=step,world=WORLD,identity=identity,receipts=receipts,
        retained=True,format='dense_full_fp32_adam_official_prefetch_remote_gzip_v1',
        snapshot_mode='immutable_cpu_async_single_writer_v1'))
    files=sorted(p.name for p in folder.iterdir() if p.is_file() and p.name!='complete.json')
    ack=ckpt.request('finalize',str(folder.relative_to(ROOT)),small_files=files)
    atomic_json(folder/'archive-complete.json',ack)


class AsyncArchive:
    def __init__(self,run,identity,args):
        self.run,self.identity,self.args=Path(run),identity,args
        self.rank=dist.get_rank();self.thread=None;self.error=None;self.folder=None

    def health(self):
        failed=torch.tensor(int(self.error is not None),device='cuda')
        dist.all_reduce(failed,op=dist.ReduceOp.MAX)
        if int(failed):raise RuntimeError('Asynchronous archive failed; preserve actual logs and snapshot/spool: '+str(self.error))

    def capture(self,model,optimizer,step,loader,generator):
        self.finish()
        folder=self.run/f'checkpoint_sft_{step:06d}';folder.mkdir(exist_ok=True)
        assert not (folder/'complete.json').exists() and not list(folder.glob('async-snapshot-rank*.json'))
        spool=ROOT/'spool'/folder.relative_to(ROOT);spool.mkdir(parents=True,exist_ok=True)
        required=sum(ckpt.local(p).numel()*p.element_size() for p in model.parameters())*3*WORLD+32*2**30
        limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
        if limit.exists() and limit.read_text().strip()!='max':
            available=int(limit.read_text())-int(current.read_text());assert available>required,(available,required)
        dist.barrier();torch.cuda.synchronize();started=time.monotonic()
        payload,data=freeze_training_state(model,optimizer,step,self.identity,self.args,loader,generator,WORLD)
        copied=time.monotonic()-started
        atomic_json(folder/f'async-snapshot-rank{self.rank}.json',dict(passed=True,step=step,rank=self.rank,
            cpu_tensor_bytes=tensor_bytes(payload)+tensor_bytes(data),seconds=copied,
            captured_before_training_continues=True,all_tensor_storage_independent=True,time=time.time()))
        # All ranks have copied the same optimizer boundary before any future update.
        dist.barrier();self.folder=folder
        def writer():
            try:
                persist,complete=persist_snapshot,finalize
                if self.args.role=='formal':
                    from retained_checkpoint import persist_snapshot as persist,finalize as complete
                persist(payload,data,folder,spool,self.rank,self.run)
                if self.rank==0:complete(folder,self.run,self.identity,step)
                wait_until(lambda:(folder/'archive-complete.json').exists(),folder,'Final archive acknowledgement missing')
            except Exception:
                self.error=traceback.format_exc()
                atomic_json(folder/f'async-error-rank{self.rank}.json',dict(error=self.error,time=time.time(),step=step))
        self.thread=threading.Thread(target=writer,name=f'checkpoint-{step}-rank{self.rank}',daemon=True)
        self.thread.start()
        return dict(path=str(folder),step=step,cpu_snapshot_captured=True,archive_pending=True,snapshot_seconds=copied)

    def finish(self):
        if self.thread is None:return
        while True:
            self.thread.join(timeout=1)
            # Every rank calls this method at the same safe boundary.
            self.health()
            all_done=torch.tensor(int(not self.thread.is_alive()),device='cuda')
            dist.all_reduce(all_done,op=dist.ReduceOp.MIN)
            if int(all_done):break
        self.thread.join();self.health();dist.barrier();self.thread=None
