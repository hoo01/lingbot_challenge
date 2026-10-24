"""CPU-only, exact FP32 model/Adam eight-to-four reshard and model export."""
import copy,gc,hashlib,json,os,time
from pathlib import Path
import torch
from safetensors.torch import save_file
ROOT=Path('/mnt/raid1/lingbotvla_sft_amd8000_to10000_20261020')
EXPECTED='2e71c096ecfbef0c92adb9e811e8dbfb9a53b5c98e2e8233073bb0aad8b45619'

def digest(t):
    return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def file_sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic(p,v):
    t=p.with_name(p.name+'.new');t.write_text(json.dumps(v));t.replace(p)

def main():
    torch.set_num_threads(4)
    src=ROOT/'incoming/source8000';dst=ROOT/'incoming/migrated8000';hf=ROOT/'incoming/model8000'
    assert not (dst/'migration.json').exists() and not (ROOT/'user-stop.json').exists()
    gate=json.loads((ROOT/'artifacts/input-gate.json').read_text());assert gate['passed']
    assert all(file_sha(ROOT/'code'/n)==h for n,h in gate['code'].items())
    assert file_sha(ROOT/'artifacts/data-filter-plan.json')==gate['data_filter_sha256']
    assert file_sha(src/'complete.json')==EXPECTED
    original=json.loads((src/'complete.json').read_text())
    assert original['step']==8000 and original['world']==8
    assert json.loads((ROOT/'artifacts/source-stage-complete.json').read_text())['actual_sha_readback_count']==16
    states=[]
    for rank in range(8):
        p=src/f'rank{rank}.pt';proof=original['receipts'][f'rank{rank}']
        assert proof['readback_equal'] and p.stat().st_size==proof['bytes'] and file_sha(p)==proof['sha256']
        s=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        assert s['step']==8000 and s['world']==8 and s['schedule']==dict(total=10000,next_update=8001)
        assert (s['microbatch'],s['accum'])==(16,2)
        states.append(s)
    for s in states[1:]:
        assert s['identity']==states[0]['identity'] and s['groups']==states[0]['groups']
        for key in ('parameters','optimizer','buffers'):assert s[key].keys()==states[0][key].keys()
    assert len(states[0]['parameters'])==1672 and len(states[0]['buffers'])==36
    dst.mkdir(exist_ok=True);hf.mkdir(exist_ok=True)
    targets=[dict(step=8000,world=4,microbatch=16,accum=4,identity=copy.deepcopy(states[0]['identity']),
        rng=None,schedule=dict(total=10000,next_update=8001),parameters={},buffers={},optimizer={},
        groups=copy.deepcopy(states[0]['groups']),migration_source_complete_sha256=EXPECTED,
        migration_rng_policy='new_explicit_CUDA_rank_seed_and_filtered_sampler_phase') for _ in range(4)]
    proofs={};model_hashes={};index={};chunk={};chunk_bytes=0;part=0;numel=0;total_model_bytes=0
    def full(values,label):
        first=values[0]
        if not isinstance(first,dict) or 'tensor' not in first:
            assert all(v==first for v in values);return first
        assert all((v['dtensor'],v['global_shape'],v['tensor'].dtype)==(first['dtensor'],first['global_shape'],first['tensor'].dtype) for v in values)
        if first['dtensor']:
            assert len(first['global_shape'])>0
            tensor=torch.cat([v['tensor'] for v in values],dim=0)
            assert list(tensor.shape)==first['global_shape'],label
        else:
            tensor=first['tensor'].clone()
            assert all(torch.equal(v['tensor'],tensor) for v in values),label
        assert torch.isfinite(tensor).all(),label
        return {'tensor':tensor,'dtensor':first['dtensor'],'global_shape':first['global_shape']}
    def split(value):
        if not isinstance(value,dict) or 'tensor' not in value:return [copy.deepcopy(value) for _ in range(4)]
        tensor=value['tensor']
        if value['dtensor']:
            # Match FSDP2 Shard(0): ceil-sized chunks followed by empty ranks.
            size=(tensor.shape[0]+3)//4
            pieces=[tensor[min(r*size,tensor.shape[0]):min((r+1)*size,tensor.shape[0])].clone() for r in range(4)]
            assert torch.equal(torch.cat(pieces,dim=0),tensor)
        else:pieces=[tensor.clone() for _ in range(4)]
        return [dict(tensor=p,dtensor=value['dtensor'],global_shape=value['global_shape']) for p in pieces]
    def flush():
        nonlocal chunk,chunk_bytes,part
        if not chunk:return
        part+=1;name=f'model-part{part:04d}.safetensors';save_file(chunk,str(hf/name))
        for key in chunk:index[key]=name
        chunk={};chunk_bytes=0;gc.collect()
    for section in ('parameters','buffers'):
        for name in states[0][section]:
            value=full([s[section][name] for s in states],section+'/'+name);tensor=value['tensor']
            assert tensor.dtype==torch.float32
            h=digest(tensor);model_hashes[name]=h;total_model_bytes+=tensor.numel()*tensor.element_size()
            if section=='parameters':numel+=tensor.numel()
            for r,local in enumerate(split(value)):targets[r][section][name]=local
            if chunk_bytes+tensor.numel()*tensor.element_size()>2*2**30:flush()
            chunk[name]=tensor;chunk_bytes+=tensor.numel()*tensor.element_size()
            if section=='parameters':
                keys=states[0]['optimizer'][name].keys()
                assert set(keys)=={'step','exp_avg','exp_avg_sq'}
                values={key:full([s['optimizer'][name][key] for s in states],'optimizer/'+name+'/'+key) for key in keys}
                assert float(values['step']['tensor'])==8000
                for key in ('exp_avg','exp_avg_sq'):
                    assert values[key]['global_shape']==value['global_shape'] and values[key]['tensor'].dtype==torch.float32
                splits={key:split(v) for key,v in values.items()}
                for r in range(4):targets[r]['optimizer'][name]={key:splits[key][r] for key in keys}
                proofs[name]={'parameter':h,**{key:digest(v['tensor']) for key,v in values.items()}}
                del values,splits
        flush()
    assert numel==6375906359 and len(index)==1708
    atomic(hf/'model.safetensors.index.json',dict(metadata={'total_size':total_model_bytes},weight_map=index))
    config=json.loads((src/'model_config.json').read_text());config['tokenizer_path']='/mnt/raid1/lingbotvla_smoke_20261009/models/Qwen3-VL-4B-Instruct'
    atomic(hf/'config.json',config)
    atomic(hf/'runtime-parameter-audit.json',dict(source_checkpoint_step=8000,trained_parameter_sha256={n:model_hashes[n] for n in targets[0]['parameters']}))
    atomic(hf/'export_audit.json',dict(passed=True,parameter_elements=numel,parameter_tensors=1672,persistent_buffers=36,source_complete_sha256=EXPECTED))
    receipts={}
    for rank in range(4):
        state=targets[rank];p=dst/f'rank{rank}.pt';assert not p.exists()
        torch.save(state,p);check=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        def equal(a,b):
            if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        assert equal(check,state)
        receipts[f'rank{rank}']=dict(step=8000,bytes=p.stat().st_size,sha256=file_sha(p),readback_equal=True)
        atomic(dst/f'rank{rank}.json',receipts[f'rank{rank}'])
        targets[rank]=None;del state,check;gc.collect()
        atomic(ROOT/'artifacts/reshard-progress.json',dict(phase='writing',ranks_complete=rank+1,time=time.time()))
    atomic(dst/'complete.json',dict(step=8000,world=4,receipts=receipts,source_complete_sha256=EXPECTED))
    atomic(dst/'migration.json',dict(passed=True,step=8000,from_world=8,to_world=4,all_parameters_buffers_adam_exact=True,
        parameter_elements=numel,parameter_tensors=1672,persistent_buffers=36,optimizer_tensor_groups=1672,
        new_rng_and_filtered_sampler_phase=True,source_complete_sha256=EXPECTED,parameter_adam_full_tensor_sha256=proofs,
        model_tensor_sha256=model_hashes,receipts=receipts,time=time.time()))
    assert not torch.cuda.is_initialized()
    atomic(ROOT/'artifacts/reshard-completed.json',dict(passed=True,step=8000,parameter_elements=numel,no_gpu_initialized=True,time=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic(ROOT/'artifacts/reshard-error.json',dict(error=repr(error),time=time.time()))
        raise
