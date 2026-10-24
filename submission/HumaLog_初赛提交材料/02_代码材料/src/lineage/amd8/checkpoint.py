"""Per-rank full FP32/Adam and prefetched-loader checkpoints with readback."""
import gc
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from policy import ROOT, WORLD, atomic_json, sha


def local(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def persistent_buffers(model):
    modules = dict(model.named_modules())
    return {name: value for name, value in model.named_buffers()
            if name.rsplit('.', 1)[-1] not in modules[
                name.rsplit('.', 1)[0] if '.' in name else '']._non_persistent_buffers_set}


def pack(value):
    if isinstance(value, torch.Tensor):
        return dict(tensor=local(value.detach()).cpu(), dtensor=hasattr(value, 'placements'),
                    global_shape=list(value.shape))
    return value


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def rng_state():
    algorithm, keys, position, gaussian, cached = np.random.get_state()
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                python=random.getstate(), numpy=dict(algorithm=algorithm,
                keys=torch.from_numpy(keys.copy()), position=position,
                gaussian=gaussian, cached=cached))


def set_rng(value):
    torch.set_rng_state(value['cpu'])
    torch.cuda.set_rng_state(value['cuda'])
    random.setstate(value['python'])
    state = value['numpy']
    np.random.set_state((state['algorithm'], state['keys'].numpy(), state['position'],
                         state['gaussian'], state['cached']))


def request(operation,relative,expected=None,**extra):
    """No account credential lives here: the local broker owns all SSH transport."""
    request_id=(operation+'_'+str(extra.get('consumer',''))+'_'+relative.replace('/','_').replace('.','_'))
    inbox=ROOT/'artifacts/archive-requests';acks=ROOT/'artifacts/archive-acks'
    inbox.mkdir(exist_ok=True);acks.mkdir(exist_ok=True)
    value=dict(id=request_id,operation=operation,relative=relative,expected=expected,**extra)
    target=inbox/(request_id+'.json')
    if target.exists():assert json.loads(target.read_text())==value
    else:atomic_json(target,value)
    deadline=time.monotonic()+8*3600
    ack=acks/target.name
    while not ack.exists():
        if (ROOT/'user-stop.json').exists():raise RuntimeError('Explicit user stop while archiving; preserve spool')
        if time.monotonic()>deadline:raise TimeoutError('Archive broker did not acknowledge; preserve checkpoint spool')
        time.sleep(2)
    result=json.loads(ack.read_text());assert result['passed'],result
    assert result['id']==request_id
    return result


def release_spool(path):
    path=Path(path)
    resolved=path.resolve()
    retained=Path('/var/tmp/lingbotvla-sft-amd8-20261017-checkpoints')
    assert (resolved.is_relative_to((ROOT/'spool').resolve()) or resolved.is_relative_to(retained)) and path.name.endswith('.pt')
    path.unlink()


def compress_state(raw,path):
    """Lossless storage only: verify the decompressed original byte SHA."""
    original=dict(bytes=raw.stat().st_size,sha256=sha(raw))
    temporary=path.with_name(path.name+'.partial')
    with raw.open('rb') as source,temporary.open('wb') as target:
        with gzip.GzipFile(filename='',mode='wb',fileobj=target,compresslevel=1,mtime=0) as stream:
            shutil.copyfileobj(source,stream,4*2**20)
        target.flush();os.fsync(target.fileno())
    with gzip.open(temporary,'rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==original['sha256']
    temporary.replace(path)
    proof=dict(bytes=path.stat().st_size,sha256=sha(path),raw_bytes=original['bytes'],
        raw_sha256=original['sha256'],codec='gzip_lossless_level1',decompressed_byte_sha_equal=True)
    release_spool(raw)
    return proof


def decode_state(path,proof):
    if not proof.get('codec'):return path
    assert proof['codec']=='gzip_lossless_level1' and proof['decompressed_byte_sha_equal']
    raw=path.with_name(path.stem+'.uncompressed.pt');temporary=raw.with_name(raw.name+'.partial')
    with gzip.open(path,'rb') as src,temporary.open('wb') as out:
        shutil.copyfileobj(src,out,4*2**20);out.flush();os.fsync(out.fileno())
    assert temporary.stat().st_size==proof['raw_bytes'] and sha(temporary)==proof['raw_sha256']
    temporary.replace(raw);release_spool(path)
    return raw


def save_one(model, optimizer, step, run, identity, args, loader, generator, folder, spool):
    rank, world = dist.get_rank(), dist.get_world_size()
    params = dict(model.named_parameters())
    names = {id(parameter): name for name, parameter in params.items()}
    loader_state = dict(step=step, identity=identity, loader=loader.state_dict(),
                        generator=generator.get_state(),epoch=loader.sampler.epoch)
    data_path = spool / f'data_rank{rank}.pt'
    write_tensor_file(data_path, loader_state)
    assert equal(loader_state, torch.load(data_path, map_location='cpu', weights_only=True))
    data_receipt=dict(step=step,bytes=data_path.stat().st_size,sha256=sha(data_path),readback_equal=True)
    atomic_json(folder/f'data_rank{rank}.json',data_receipt)
    del loader_state
    payload = dict(step=step, world=world, microbatch=args.batch, accum=args.accum,
        identity=identity, rng=rng_state(), schedule=dict(total=10000, next_update=step + 1),
        parameters={name: pack(value) for name, value in params.items()},
        buffers={name: pack(value) for name, value in persistent_buffers(model).items()},
        optimizer={name: {key: pack(value) for key, value in optimizer.state.get(parameter, {}).items()}
                   for name, parameter in params.items()},
        groups=[{**{key: value for key, value in group.items() if key != 'params'},
                 'params': [names[id(p)] for p in group['params']]} for group in optimizer.param_groups])
    raw_path = spool / f'rank{rank}.uncompressed.pt'
    write_tensor_file(raw_path, payload)
    del payload
    gc.collect()
    restored = torch.load(raw_path, map_location='cpu', weights_only=True, mmap=True)
    for name, parameter in params.items():
        assert torch.equal(restored['parameters'][name]['tensor'], local(parameter.detach()).cpu()), name
        for key, value in optimizer.state.get(parameter, {}).items():
            expected = pack(value)
            assert equal(restored['optimizer'][name][key], expected), (name, key)
    for name, buffer in persistent_buffers(model).items():
        assert torch.equal(restored['buffers'][name]['tensor'], local(buffer.detach()).cpu()), name
    assert equal(restored['rng'], rng_state())
    del restored
    gc.collect()
    path=spool/f'rank{rank}.pt'
    receipt=dict(step=step,readback_equal=True,**compress_state(raw_path,path))
    atomic_json(folder/f'rank{rank}.json', receipt)
    relative=str(folder.relative_to(ROOT))
    for path,proof in ((data_path,data_receipt),(path,receipt)):
        request('upload',relative+'/'+path.name,proof)
        release_spool(path)
    return receipt


def save(model,optimizer,step,run,identity,args,loader,generator):
    rank,world=dist.get_rank(),dist.get_world_size();assert world==WORLD
    folder=Path(run)/f'checkpoint_sft_{step:06d}';folder.mkdir(exist_ok=True)
    assert not (folder/'complete.json').exists(),'Never overwrite a complete checkpoint'
    spool=ROOT/'spool'/folder.relative_to(ROOT);spool.mkdir(parents=True,exist_ok=True)
    # At most two ~9.6GB rank files coexist, leaving room for source weights/data.
    receipt=None;started=time.monotonic()
    for first in range(0,world,2):
        dist.barrier()
        if rank in (first,first+1):
            receipt=save_one(model,optimizer,step,run,identity,args,loader,generator,folder,spool)
        dist.barrier()
    dist.barrier()
    if rank == 0:
        receipts = {f'{prefix}{r}': json.loads((folder / f'{prefix}{r}.json').read_text())
                    for prefix in ('rank', 'data_rank') for r in range(world)}
        assert len(receipts) == 16 and all(r['readback_equal'] and r['step'] == step for r in receipts.values())
        for name in ('norm_stats.json', 'norm_stats.provenance.json', 'robotwin_relative.yaml',
                     'sampling-binding.json', 'source-binding.json','asset-binding.json'):
            shutil.copy2(ROOT / 'artifacts' / name, folder / name)
        for name in ('config.json', 'model_config.json', 'lingbotvla_cli.yaml'):
            shutil.copy2(Path(run) / name, folder / name)
        atomic_json(folder / 'complete.json', dict(step=step, world=world, identity=identity,
            receipts=receipts,retained=True,format='dense_full_fp32_adam_official_prefetch_remote_gzip_v1'))
        files=[p.name for p in folder.iterdir() if p.is_file() and p.name!='complete.json']
        ack=request('finalize',str(folder.relative_to(ROOT)),small_files=sorted(files))
        atomic_json(folder/'archive-complete.json',ack)
    dist.barrier()
    return dict(path=str(folder),archived=True,seconds=time.monotonic()-started,**receipt)


def write_tensor_file(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def restore_one(model, optimizer, folder, spool, complete, identity, args, loader, generator, local_source=None):
    from torch.distributed.tensor import DTensor
    rank = dist.get_rank()
    assert complete['identity'] == identity and complete['world'] == WORLD
    for prefix in ('rank', 'data_rank'):
        path = spool / f'{prefix}{rank}.pt'
        proof = complete['receipts'][f'{prefix}{rank}']
        if local_source is None:
            request('download',str(folder.relative_to(ROOT))+'/'+path.name,proof,consumer=Path(args.run).name)
        else:
            source=local_source/path.name
            assert source.is_file() and not source.is_symlink() and source.stat().st_size==proof['bytes']
            assert sha(source)==proof['sha256'] and not path.exists()
            # Restore from a separate scratch copy; never consume the retained checkpoint.
            shutil.copy2(source,path)
        assert proof['readback_equal'] and sha(path) == proof['sha256']
    raw=decode_state(spool/f'rank{rank}.pt',complete['receipts'][f'rank{rank}'])
    state = torch.load(raw, map_location='cpu', weights_only=True, mmap=True)
    assert state['identity'] == identity
    assert state['schedule'] == dict(total=10000, next_update=state['step'] + 1)
    assert (state['microbatch'], state['accum'], state['world']) == (args.batch, args.accum, WORLD)
    params = dict(model.named_parameters())
    assert state['parameters'].keys() == params.keys()
    def unpack(value, reference):
        if not isinstance(value, dict) or 'tensor' not in value:
            return value
        tensor = value['tensor']
        if value['dtensor']:
            assert list(reference.shape) == value['global_shape']
            return DTensor.from_local(tensor.to(reference.device), reference.device_mesh,
                reference.placements, run_check=False, shape=reference.shape, stride=reference.stride())
        # Adam's CPU step scalars must not retain the mmap storage of the whole
        # downloaded rank file, otherwise NFS cannot release the limited spool.
        return tensor.clone() if tensor.ndim == 0 else tensor.to(reference.device)
    with torch.no_grad():
        for name, parameter in params.items():
            local(parameter).copy_(state['parameters'][name]['tensor'])
            assert torch.equal(local(parameter).cpu(), state['parameters'][name]['tensor']), name
        buffers = persistent_buffers(model)
        assert state['buffers'].keys() == buffers.keys()
        for name, buffer in buffers.items():
            local(buffer).copy_(state['buffers'][name]['tensor'])
            assert torch.equal(local(buffer).cpu(), state['buffers'][name]['tensor']), name
    assert len(optimizer.param_groups) == len(state['groups'])
    for target, source in zip(optimizer.param_groups, state['groups']):
        assert [id(p) for p in target['params']] == [id(params[name]) for name in source['params']]
        target.update({key: value for key, value in source.items() if key != 'params'})
        for name in source['params']:
            optimizer.state[params[name]] = {key: unpack(value, params[name])
                for key, value in state['optimizer'][name].items()}
            for key, value in optimizer.state[params[name]].items():
                assert equal(pack(value), state['optimizer'][name][key]), (name, key)
    data = torch.load(spool / f'data_rank{rank}.pt', map_location='cpu', weights_only=True)
    assert data['identity'] == identity and data['step'] == state['step']
    loader.sampler.set_epoch(data['epoch']);loader.load_state_dict(data['loader'])
    generator.set_state(data['generator'])
    step, saved_rng = state['step'], copy.deepcopy(state['rng'])
    del data, state
    gc.collect()
    release_spool(raw);release_spool(spool/f'data_rank{rank}.pt')
    return step, saved_rng


def restore(model,optimizer,folder,identity,args,loader,generator):
    rank,world=dist.get_rank(),dist.get_world_size();assert world==WORLD
    folder=Path(folder);complete=json.loads((folder/'complete.json').read_text())
    local_source=None
    if complete.get('local_retained'):
        retained=Path('/var/tmp/lingbotvla-sft-amd8-20261017-checkpoints')
        candidate=retained/folder.parent.name/folder.name
        assert complete['local_storage']==str(candidate) and candidate.resolve()==candidate
        if all((candidate/(name+'.pt')).is_file() for name in complete['receipts']):
            local_source=candidate
    if local_source is None:
        assert json.loads((folder/'archive-complete.json').read_text())['passed']
        spool=ROOT/'spool'/folder.relative_to(ROOT)
    else:
        spool=retained/'restore'/Path(args.run).name/folder.name
    spool.mkdir(parents=True,exist_ok=True)
    result=None
    for first in range(0,world,2):
        dist.barrier()
        if rank in (first,first+1):
            result=restore_one(model,optimizer,folder,spool,complete,identity,args,loader,generator,local_source)
        dist.barrier()
    assert result is not None
    atomic_json(ROOT/'artifacts'/f'restore-exact-{Path(args.run).name}-rank{rank}.json',
        dict(passed=True,step=result[0],all_parameters_buffers_adam_exact=True,
             restored_loader_epoch=loader.sampler.epoch,source_checkpoint=str(folder),identity=identity,
             restored_from_retained_local_copy=local_source is not None))
    return result
