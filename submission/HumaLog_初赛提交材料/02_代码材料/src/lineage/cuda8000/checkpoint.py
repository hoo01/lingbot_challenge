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
                identity=identity, rng=rng_state(), schedule=dict(total=10000, next_update=step + 1),
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
        for name in ('config.json', 'model_config.json', 'lingbotvla_cli.yaml'):
            shutil.copy2(Path(run) / name, folder / name)
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
    from torch.distributed.tensor import DTensor
    rank = dist.get_rank()
    folder = Path(folder)
    complete = json.loads((folder / 'complete.json').read_text())
    if folder == ROOT / 'incoming/migrated8000':
        return restore_migration(model, optimizer, folder, identity, args)
    assert complete['identity'] == identity and complete['world'] == 4
    for prefix in ('rank', 'data_rank'):
        path = folder / f'{prefix}{rank}.pt'
        proof = json.loads(path.with_suffix('.json').read_text())
        assert proof['readback_equal'] and sha(path) == proof['sha256']
    state = torch.load(folder / f'rank{rank}.pt', map_location='cpu', weights_only=True, mmap=True)
    assert state['identity'] == identity
    assert state['schedule'] == dict(total=10000, next_update=state['step'] + 1)
    assert (state['microbatch'], state['accum'], state['world']) == (args.batch, args.accum, 4)
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
    data = torch.load(folder / f'data_rank{rank}.pt', map_location='cpu', weights_only=True)
    assert data['identity'] == identity and data['step'] == state['step']
    loader.load_state_dict(data['loader'])
    generator.set_state(data['generator'])
    step, saved_rng = state['step'], state['rng']
    del data, state
    gc.collect()
    return step, saved_rng

def fingerprint(model,optimizer):
    from images import tensor_sha
    params=dict(model.named_parameters())
    return dict(parameters={n:tensor_sha(local(p.detach())) for n,p in params.items()},
        buffers={n:tensor_sha(local(b.detach())) for n,b in persistent_buffers(model).items()},
        optimizer={n:{k:(tensor_sha(local(v)) if isinstance(v,torch.Tensor) else v) for k,v in optimizer.state[p].items()} for n,p in params.items()},
        groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups])


def restore_migration(model,optimizer,folder,identity,args):
    from torch.distributed.tensor import DTensor
    rank=dist.get_rank();proof=json.loads((folder/'migration.json').read_text())
    assert proof['passed'] and proof['all_parameters_buffers_adam_exact'] and (args.batch,args.accum)==(16,4)
    path=folder/f'rank{rank}.pt';receipt=proof['receipts'][f'rank{rank}']
    assert path.stat().st_size==receipt['bytes'] and sha(path)==receipt['sha256']
    s=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert s['step']==8000 and s['world']==4 and s['schedule']==dict(total=10000,next_update=8001)
    params=dict(model.named_parameters());assert params.keys()==s['parameters'].keys()
    with torch.no_grad():
        for n,p in params.items():
            v=s['parameters'][n];assert list(p.shape)==v['global_shape'] and local(p).shape==v['tensor'].shape
            local(p).copy_(v['tensor']);assert torch.equal(local(p).cpu(),v['tensor']),n
        buffers=persistent_buffers(model);assert buffers.keys()==s['buffers'].keys()
        for n,b in buffers.items():
            v=s['buffers'][n];local(b).copy_(v['tensor']);assert torch.equal(local(b).cpu(),v['tensor']),n
    assert len(optimizer.param_groups)==len(s['groups'])
    for target,source in zip(optimizer.param_groups,s['groups']):
        assert [id(p) for p in target['params']]==[id(params[n]) for n in source['params']]
        target.update({k:v for k,v in source.items() if k!='params'})
        for n in source['params']:
            p=params[n];state={}
            for k,v in s['optimizer'][n].items():
                tensor=v['tensor']
                if v['dtensor']:
                    value=DTensor.from_local(tensor.to(p.device),p.device_mesh,p.placements,run_check=False,shape=p.shape,stride=p.stride())
                else:value=tensor.clone() if tensor.ndim==0 else tensor.to(p.device)
                assert equal(pack(value),v),(n,k)
                state[k]=value
            assert float(state['step'])==8000
            optimizer.state[p]=state
    del s;gc.collect()
    atomic_json(ROOT/'artifacts'/f'migration-restore-{Path(args.run).name}-rank{rank}.json',
        dict(passed=True,step=8000,all_parameters_buffers_adam_exact=True,parameter_tensors=len(params),
            new_sampler_rng_phase=True,source_complete_sha256=proof['source_complete_sha256'],identity=identity))
    return 8000,None
