"""Bounded two-slot local full states, with independent asynchronous archives."""
import ctypes
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time
import torch
import checkpoint as ckpt
from policy import ROOT,WORLD,atomic_json,sha

STORAGE=Path('/var/tmp/lingbotvla-sft-amd8-20261017-checkpoints')
SLOT_BYTES=10*2**30
RUN_NAME='formal_sft_10000'


def slot_offset(step,rank):
    assert step in (2000,4000,6000,8000,10000) and 0<=rank<WORLD==8
    return (((step//2000-1)%2)*WORLD+rank)*SLOT_BYTES


def allocate_slot(step,rank,release=False):
    reserve=STORAGE/'two-checkpoint-capacity.reserve'
    assert reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_size==2*WORLD*SLOT_BYTES
    offset=slot_offset(step,rank)
    with reserve.open('r+b') as stream:
        if release:
            libc=ctypes.CDLL(None,use_errno=True)
            libc.fallocate.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_longlong,ctypes.c_longlong]
            if libc.fallocate(stream.fileno(),3,offset,SLOT_BYTES):
                error=ctypes.get_errno();raise OSError(error,os.strerror(error))
        else:os.posix_fallocate(stream.fileno(),offset,SLOT_BYTES)
        os.fsync(stream.fileno())


def storage_path(folder,spool):
    assert folder.parent.name==RUN_NAME and folder.parent.parent==ROOT/'runs'
    assert spool==ROOT/'spool'/folder.relative_to(ROOT)
    assert spool.parent.is_symlink() and spool.parent.resolve()==STORAGE/RUN_NAME
    assert spool.resolve().is_relative_to(STORAGE/RUN_NAME)
    assert shutil.disk_usage(STORAGE).free>2*2**30
    return spool


def local_proof(folder,step):
    assert all((folder/f'local-ready-rank{rank}.json').exists() for rank in range(WORLD))
    receipts={f'{prefix}{rank}':json.loads((folder/f'{prefix}{rank}.json').read_text())
              for prefix in ('rank','data_rank') for rank in range(WORLD)}
    assert len(receipts)==2*WORLD and all(p['readback_equal'] and p['step']==step for p in receipts.values())
    return receipts


def persist_snapshot(payload,data,folder,spool,rank,run):
    from async_checkpoint import acquire,check_pending,wait_until,tensor_bytes
    spool=storage_path(folder,spool);step=payload['step'];started=time.time()
    lock=(ROOT/'artifacts/async-checkpoint-io.lock').open('a+')
    wait_until(lambda:acquire(lock),folder,'Timed out waiting for the local checkpoint writer')
    try:
        assert tensor_bytes(payload)+64*2**20<SLOT_BYTES
        paths=[spool/f'data_rank{rank}.pt',spool/f'rank{rank}.pt']
        assert not any(p.exists() or p.with_name(p.name+'.partial').exists() for p in paths)
        allocate_slot(step,rank,release=True)
        proofs=[]
        for path,value in zip(paths,(data,payload)):
            ckpt.write_tensor_file(path,value)
            loaded=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            assert ckpt.equal(value,loaded),'Local immutable full state readback differs'
            del loaded;gc.collect()
            proof=dict(step=step,bytes=path.stat().st_size,sha256=sha(path),readback_equal=True,
                       local_retained=True,format='torch_save_full_fp32_adam_v1')
            atomic_json(folder/(path.stem+'.json'),proof);proofs.append(proof)
        assert sum(p['bytes'] for p in proofs)<SLOT_BYTES
        atomic_json(folder/f'local-ready-rank{rank}.json',dict(passed=True,step=step,rank=rank,
            bytes=sum(p['bytes'] for p in proofs),both_actual_readbacks_equal=True,time=time.time()))
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if rank==0:publish_local(folder,run,payload['identity'],step)
    # The next rank can write locally while this rank's upload is in progress.
    relative=str(folder.relative_to(ROOT))
    for path,proof in zip(paths,proofs):
        check_pending(folder);ckpt.request('upload',relative+'/'+path.name,proof)
    current=run/f'progress_rank{rank}.json'
    observed=json.loads(current.read_text())['step'] if current.exists() else step
    proof=dict(passed=True,step=step,rank=rank,started=started,finished=time.time(),
        latest_optimizer_step=observed,updates_advanced_during_archive=observed>step,
        immutable_cpu_snapshot=True,writer_touched_live_model_or_optimizer=False,
        bytes=proofs[1]['bytes'],readback_equal=True,local_full_state_retained=True)
    atomic_json(folder/f'async-archived-rank{rank}.json',proof)
    return proof


def prune_superseded(folder,step):
    """Only delete verified older transfer states; keep all metadata and latest."""
    latest=json.loads((ROOT/'artifacts/latest-local-checkpoint.json').read_text())
    assert latest['step']==step and latest['complete_sha256']==sha(folder/'complete.json')
    assert json.loads((folder/'archive-complete.json').read_text())['passed']
    removed=[]
    for previous in sorted(folder.parent.glob('checkpoint_sft_*')):
        old_step=int(previous.name.rsplit('_',1)[1])
        if old_step>=step or not (previous/'complete.json').exists():continue
        if (previous/'local-transfer-pruned.json').exists():continue
        complete=json.loads((previous/'complete.json').read_text())
        assert json.loads((previous/'archive-complete.json').read_text())['passed']
        assert complete['world']==WORLD and len(complete['receipts'])==2*WORLD
        directory=STORAGE/RUN_NAME/previous.name
        assert directory.resolve()==directory and directory.parent==STORAGE/RUN_NAME
        assert {p.name for p in directory.iterdir()}<={n+'.pt' for n in complete['receipts']}
        rows=[]
        for name,proof in complete['receipts'].items():
            p=directory/(name+'.pt')
            assert p.is_file() and not p.is_symlink() and p.stat().st_size==proof['bytes']
            rows.append(dict(name=p.name,bytes=proof['bytes'],archived_sha256=proof['sha256']))
        for row in rows:(directory/row['name']).unlink()
        for rank in range(WORLD):allocate_slot(old_step,rank)
        receipt=dict(passed=True,step=old_step,replaced_by=step,actual_archive_ack_sha256=sha(previous/'archive-complete.json'),
                     files=rows,latest_complete_preserved=True,all_metadata_retained=True,time=time.time())
        atomic_json(previous/'local-transfer-pruned.json',receipt);removed.append(old_step)
    return removed


def publish_local(folder,run,identity,step):
    from async_checkpoint import wait_until
    wait_until(lambda:all((folder/f'local-ready-rank{rank}.json').exists() for rank in range(WORLD)),
        folder,'Waiting for all eight locally saved full states')
    receipts=local_proof(folder,step)
    for name in ('norm_stats.json','norm_stats.provenance.json','robotwin_relative.yaml',
                 'sampling-binding.json','source-binding.json','asset-binding.json'):
        shutil.copy2(ROOT/'artifacts'/name,folder/name)
    for name in ('config.json','model_config.json','lingbotvla_cli.yaml'):shutil.copy2(run/name,folder/name)
    atomic_json(folder/'complete.json',dict(step=step,world=WORLD,identity=identity,receipts=receipts,retained=True,
        format='dense_full_fp32_adam_official_prefetch_local_raw_v2',snapshot_mode='immutable_cpu_async_retained_local_v2',
        local_retained=True,local_storage=str(STORAGE/RUN_NAME/folder.name)))
    atomic_json(ROOT/'artifacts/latest-local-checkpoint.json',dict(passed=True,step=step,
        checkpoint=str(folder.relative_to(ROOT)),complete_sha256=sha(folder/'complete.json'),
        local_actual_readbacks=2*WORLD,archive_pending=True,time=time.time()))


def finalize(folder,run,identity,step):
    from async_checkpoint import wait_until
    assert json.loads((folder/'complete.json').read_text())['identity']==identity
    wait_until(lambda:all((folder/f'async-archived-rank{rank}.json').exists() for rank in range(WORLD)),
        folder,'Waiting for this checkpoint actual remote upload receipts')
    files=sorted(p.name for p in folder.iterdir() if p.is_file() and p.name!='complete.json')
    ack=ckpt.request('finalize',str(folder.relative_to(ROOT)),small_files=files)
    assert ack['passed'] and ack['world']==WORLD and ack['readbacks']==2*WORLD
    atomic_json(folder/'archive-complete.json',ack)
    atomic_json(ROOT/'artifacts/latest-local-checkpoint.json',dict(passed=True,step=step,
        checkpoint=str(folder.relative_to(ROOT)),complete_sha256=sha(folder/'complete.json'),
        local_actual_readbacks=2*WORLD,archive_pending=False,actual_archive_ack_sha256=sha(folder/'archive-complete.json'),time=time.time()))
    removed=prune_superseded(folder,step)
    atomic_json(folder/'retention-finalized.json',dict(passed=True,latest_local_step=step,
        removed_superseded_local_steps=removed,all_remote_archives_retained=True,time=time.time()))
