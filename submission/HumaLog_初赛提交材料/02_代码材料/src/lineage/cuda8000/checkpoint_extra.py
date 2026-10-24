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
