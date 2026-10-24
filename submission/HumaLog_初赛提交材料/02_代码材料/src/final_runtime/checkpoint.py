"""Per-rank full FP32/Adam and prefetched-loader checkpoints with readback."""
import gc
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import torch.distributed as dist
from policy import ROOT, atomic_json, sha


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


def save(model, optimizer, step, run, identity, args, loader, generator):
    rank, world = dist.get_rank(), dist.get_world_size()
    folder = Path(run) / f'checkpoint_sft_{step:06d}'
    folder.mkdir(exist_ok=True)
    assert not (folder / 'complete.json').exists(), 'Never overwrite a complete checkpoint'
    receipt=None
    for owner in range(4):
        dist.barrier()
        if rank == owner:
            params = dict(model.named_parameters())
            names = {id(parameter): name for name, parameter in params.items()}
            loader_state = dict(step=step, identity=identity, loader=loader.state_dict(),
                                generator=generator.get_state())
            data_path = folder / f'data_rank{rank}.pt'
            write_tensor_file(data_path, loader_state)
            assert equal(loader_state, torch.load(data_path, map_location='cpu', weights_only=True))
            atomic_json(data_path.with_suffix('.json'), dict(step=step, bytes=data_path.stat().st_size,
                sha256=sha(data_path), readback_equal=True))
            del loader_state
            payload = dict(step=step, world=world, microbatch=args.batch, accum=args.accum,
                identity=identity, rng=rng_state(), schedule=dict(total=16000, phase_start=10000, stage_updates=6000, rewarm_updates=300, next_update=step+1),
                parameters={name: pack(value) for name, value in params.items()},
                buffers={name: pack(value) for name, value in persistent_buffers(model).items()},
                optimizer={name: {key: pack(value) for key, value in optimizer.state.get(parameter, {}).items()}
                           for name, parameter in params.items()},
                groups=[{**{key: value for key, value in group.items() if key != 'params'},
                         'params': [names[id(p)] for p in group['params']]} for group in optimizer.param_groups])
            path = folder / f'rank{rank}.pt'
            write_tensor_file(path, payload)
            del payload
            gc.collect()
            restored = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
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
            receipt = dict(step=step, bytes=path.stat().st_size, sha256=sha(path), readback_equal=True)
            atomic_json(path.with_suffix('.json'), receipt)
        dist.barrier()
    dist.barrier()
    if rank == 0:
        receipts = {f'{prefix}{r}': json.loads((folder / f'{prefix}{r}.json').read_text())
                    for prefix in ('rank', 'data_rank') for r in range(world)}
        assert len(receipts) == 8 and all(r['readback_equal'] and r['step'] == step for r in receipts.values())
        from policy import ROOT
        for name in ('norm_stats.json', 'norm_stats.provenance.json', 'robotwin_relative.yaml',
                     'task_weights.csv', 'source-binding.json', 'data-filter-plan.json', 'source-migration-binding.json'):
            shutil.copy2(ROOT / 'artifacts' / name, folder / name)
        shutil.copy2(ROOT/'artifacts/geometryfix11000_requested/aug6000-recipe.json',folder/'aug6000-recipe.json')
        shutil.copy2(ROOT/'artifacts/geometryfix11000_requested/source-binding.json',folder/'geometryfix-source-binding.json')
        for name in ('config.json', 'model_config.json', 'lingbotvla_cli.yaml'):
            shutil.copy2(Path(run) / name, folder / name)
        shutil.copy2(ROOT/'foreground/manifest.reviewed.json',folder/'foreground-manifest.json')
        shutil.copy2(ROOT/'foreground/semantic-policy-review.json',folder/'foreground-semantic-policy-review.json')
        atomic_json(folder / 'complete.json', dict(step=step, world=world, identity=identity,
            receipts=receipts, retained=True, format='dense_full_fp32_adam_prefetch_v1'))
    dist.barrier()
    return dict(path=str(folder), **receipt)


def write_tensor_file(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def restore(model, optimizer, folder, identity, args, loader, generator):
    import copy
    from torch.distributed.tensor import DTensor
    from policy import PARENT_CHECKPOINT,PARENT_COMPLETE_SHA,PEAKS
    rank=dist.get_rank();folder=Path(folder).resolve()
    parent=folder==PARENT_CHECKPOINT.resolve()
    complete=json.loads((folder/'complete.json').read_text())
    if parent:
        assert sha(folder/'complete.json')==PARENT_COMPLETE_SHA
        assert complete['step']==11000 and complete['world']==4
        binding=json.loads((ROOT/'artifacts/geometryfix11000_requested/source-binding.json').read_text())
        assert complete['identity']==binding['old_identity']
        for name in ('manifest.json','norm_stats.json','robotwin_relative.yaml','data-filter-plan.json','input_inventory.json'):
            assert complete['identity'][name]==identity[name],name
    else:assert complete['identity']==identity and complete['world']==4
    for prefix in ('rank','data_rank'):
        path=folder/f'{prefix}{rank}.pt';receipt=json.loads(path.with_suffix('.json').read_text())
        assert receipt==complete['receipts'][f'{prefix}{rank}'] and receipt['readback_equal']
        assert path.stat().st_size==receipt['bytes'] and sha(path)==receipt['sha256']
    state=torch.load(folder/f'rank{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
    assert state['identity']==complete['identity'] and state['world']==4
    if parent:
        assert state['step']==11000 and (state['microbatch'],state['accum'])==(16,3)
        assert state['schedule']==dict(total=16000,phase_start=10000,stage_updates=6000,rewarm_updates=300,next_update=11001)
    else:
        assert state['schedule']==dict(total=16000,phase_start=10000,stage_updates=6000,rewarm_updates=300,next_update=state['step']+1)
        assert (state['microbatch'],state['accum'])==(args.batch,args.accum)
    params=dict(model.named_parameters());assert params.keys()==state['parameters'].keys()
    assert len(params)==1672 and sum(p.numel() for p in params.values())==6375906359
    def unpack(value,reference):
        if not isinstance(value,dict) or 'tensor' not in value:return value
        tensor=value['tensor']
        if value['dtensor']:
            assert list(reference.shape)==value['global_shape']
            return DTensor.from_local(tensor.to(reference.device),reference.device_mesh,reference.placements,
                run_check=False,shape=reference.shape,stride=reference.stride())
        return tensor.clone() if tensor.ndim==0 else tensor.to(reference.device)
    with torch.no_grad():
        for name,p in params.items():
            saved=state['parameters'][name]
            assert list(p.shape)==saved['global_shape'] and local(p).shape==saved['tensor'].shape
            assert saved['tensor'].dtype==torch.float32 and p.dtype==torch.float32
            local(p).copy_(saved['tensor']);assert torch.equal(local(p).cpu(),saved['tensor']),name
        buffers=persistent_buffers(model);assert buffers.keys()==state['buffers'].keys() and len(buffers)==36
        for name,b in buffers.items():
            saved=state['buffers'][name];local(b).copy_(saved['tensor'])
            assert torch.equal(local(b).cpu(),saved['tensor']),name
    assert len(optimizer.param_groups)==len(state['groups'])==4
    for target,source in zip(optimizer.param_groups,state['groups']):
        assert [id(p) for p in target['params']]==[id(params[n]) for n in source['params']]
        target.update({k:v for k,v in source.items() if k!='params'})
        for name in source['params']:
            parameter=params[name]
            optimizer.state[parameter]={k:unpack(v,parameter) for k,v in state['optimizer'][name].items()}
            for key,value in optimizer.state[parameter].items():
                assert equal(pack(value),state['optimizer'][name][key]),(name,key)
            if parent:assert float(optimizer.state[parameter]['step'])==11000
        if parent:assert target['peak_lr']==PEAKS[target['label']]  # Scalars and Adam moments inherited verbatim.
    data=torch.load(folder/f'data_rank{rank}.pt',map_location='cpu',weights_only=True)
    assert data['identity']==complete['identity'] and data['step']==state['step']
    loader.load_state_dict(data['loader']);generator.set_state(data['generator'])
    step,saved_rng=state['step'],copy.deepcopy(state['rng'])
    del data,state;gc.collect()
    atomic_json(Path(ROOT)/'runs'/args.run/f'restore_rank{rank}.json',dict(passed=True,step=step,
        parent_boundary=parent,all_parameters_buffers_Adam_values_exact=True,parameter_tensors=1672,
        persistent_buffers=36,source_complete_sha256=sha(folder/'complete.json'),
        old_official_loader_cursor_and_generator_restored=True,world=4,
        group_changes_only_declared_lr_schedule_metadata=False,geometry_only_data_contract_transition=parent))
    return step,saved_rng

def fingerprint(model,optimizer):
    from images import tensor_sha
    params=dict(model.named_parameters())
    return dict(rng=rng_fingerprint(),parameters={n:tensor_sha(local(p.detach())) for n,p in params.items()},
        buffers={n:tensor_sha(local(b.detach())) for n,b in persistent_buffers(model).items()},
        optimizer={n:{k:(tensor_sha(local(v)) if isinstance(v,torch.Tensor) else v) for k,v in optimizer.state[p].items()} for n,p in params.items()},
        groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups])

def rng_fingerprint():
    import hashlib
    from images import tensor_sha
    r=rng_state();n=r['numpy']
    return dict(cpu=tensor_sha(r['cpu']),cuda=tensor_sha(r['cuda']),
        python=hashlib.sha256(repr(r['python']).encode()).hexdigest(),
        numpy=dict(algorithm=n['algorithm'],keys=tensor_sha(n['keys']),position=n['position'],
            gaussian=n['gaussian'],cached=n['cached']))


