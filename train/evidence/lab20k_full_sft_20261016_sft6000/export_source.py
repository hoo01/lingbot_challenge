"""CPU-only exact reconstruction of the saved dense SFT FSDP shards.

Runs in a dedicated evaluation folder, not inside training code or a checkpoint.
The original optimizer/data files are read-only and never transferred.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
os.nice(10)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def reconstruct(parts):
    import torch
    first = parts[0]
    assert len(parts) == 4
    assert all({k:v for k,v in p.items() if k!='tensor'} ==
               {k:v for k,v in first.items() if k!='tensor'} for p in parts)
    values = [p['tensor'] for p in parts]
    shape = first['global_shape']
    assert all(t.dtype == torch.float32 for t in values)
    if not first['dtensor'] or all(list(t.shape) == shape for t in values):
        result = values[0].clone()
        assert all(torch.equal(result, t) for t in values[1:]), 'Replicated values differ'
    else:
        assert shape and all(list(t.shape[1:]) == shape[1:] for t in values)
        # Actual FSDP2 full-shard placement in this run is Shard(0).
        shard_size = (shape[0] + 3) // 4
        expected = [max(0, min(shard_size, shape[0]-rank*shard_size)) for rank in range(4)]
        assert [t.shape[0] for t in values] == expected, 'Unexpected shard geometry'
        result = torch.cat(values, dim=0)
    assert list(result.shape) == shape and torch.isfinite(result).all()
    return result.contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--complete-sha', required=True)
    a = p.parse_args()
    source, out = Path(a.source), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out/'extract.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'packet_complete.json').exists():
            data = json.loads((out/'packet_complete.json').read_text())
            assert data['source_complete_sha256'] == a.complete_sha
            print(json.dumps({'already_complete':True,'files':len(data['files'])}), flush=True)
            return
        # Partial exports require explicit diagnosis, never a blind re-extraction.
        assert not list(out.glob('model-*.safetensors')), 'Inspect preserved partial extraction first'
        def status(phase, **extra):
            save(out/'extract-status.json', dict(time=time.time(),phase=phase,gpu_used=False,**extra))
        status('verifying_source')
        assert sha(source/'complete.json') == a.complete_sha
        proof = json.loads((source/'complete.json').read_text())
        identity = proof['identity']
        assert proof['step']==6000 and proof['world']==4 and len(proof['receipts'])==8
        assert identity['dense_full_sft'] and identity['source_step']==20000
        assert identity['norm_stats.json']=='b24e6a7ccf44fa3ecc9dbbe060cf9b0937cf3307bc2a68232bc98c4578c22f6d'
        assert all(v['readback_equal'] and v['step']==6000 for v in proof['receipts'].values())
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        torch.set_num_threads(2)
        states=[]
        for rank in range(4):
            path=source/f'rank{rank}.pt'
            assert path.stat().st_size==proof['receipts'][f'rank{rank}']['bytes']
            assert sha(path)==proof['receipts'][f'rank{rank}']['sha256']
            state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            assert state['step']==6000 and state['identity']==identity and state['world']==4
            states.append(state)
            status('source_rank_verified', rank=rank)
        parameters=list(states[0]['parameters']);buffers=list(states[0]['buffers'])
        assert len(parameters)==1672 and len(buffers)==36
        assert not any('lora_' in n or '_orig_mod.' in n for n in parameters)
        assert all(list(s['parameters'])==parameters and list(s['buffers'])==buffers for s in states)
        hashes={}; buffer_hashes={}; index={}; chunk={}; chunk_bytes=0; number=0; numel=0
        files=[]
        def flush():
            nonlocal chunk,chunk_bytes,number
            if not chunk:return
            number+=1;name=f'model-{number:05d}.safetensors'
            tmp=out/(name+'.partial');save_file(chunk,str(tmp),metadata={'format':'pt'})
            tmp.replace(out/name)
            with safe_open(out/name,framework='pt',device='cpu') as readback:
                assert set(readback.keys())==set(chunk)
                for key in readback.keys():
                    actual=readback.get_tensor(key)
                    assert torch.equal(actual,chunk[key]),key
            for key in chunk:index[key]=name
            files.append(dict(name=name,bytes=(out/name).stat().st_size,sha256=sha(out/name)))
            chunk={};chunk_bytes=0;gc.collect()
            status('extracting_dense_parameters', files=number,tensors=len(index))
        for section,names in [('parameters',parameters),('buffers',buffers)]:
            for name in names:
                value=reconstruct([s[section][name] for s in states])
                size=value.numel()*value.element_size()
                if chunk_bytes+size>1_900_000_000:flush()
                digest=hashlib.sha256(value.numpy().tobytes()).hexdigest()
                (hashes if section=='parameters' else buffer_hashes)[name]=digest
                if section=='parameters':numel+=value.numel()
                chunk[name]=value;chunk_bytes+=size
        flush()
        assert numel==6375906359 and len(index)==1708
        del states,chunk,value;gc.collect()
        save(out/'model.safetensors.index.json', {'metadata':{'total_size':sum(x['bytes'] for x in files)},'weight_map':index})
        shutil.copy2(source/'model_config.json',out/'config.json')
        for name in ('lingbotvla_cli.yaml','norm_stats.json','norm_stats.provenance.json',
                     'robotwin_relative.yaml','source-binding.json','complete.json'):
            shutil.copy2(source/name,out/name)
        save(out/'runtime-construction.json',{'nonpersistent_buffer_initialization':'native','checkpoint_parameter_dtype':'float32'})
        audit=dict(kind='dense_full_sft_FP32_values_all_1672_parameter_tensors',
            source_checkpoint_step=6000,parent_initial_step=20000,source_checkpoint=str(source),
            source_checkpoint_identity=identity,source_complete_sha256=a.complete_sha,
            trained_parameter_sha256=hashes,persistent_buffer_sha256=buffer_hashes,
            parameter_elements=numel,parameter_tensors=1672,persistent_buffers=36,
            dtype='float32',no_lora_merge_required=True,all_export_readback_equal=True)
        save(out/'runtime-parameter-audit.json',audit)
        for path in sorted(out.iterdir()):
            if path.suffix in ('.json','.yaml') and path.name not in ('extract-status.json','packet_complete.json'):
                files.append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path)))
        assert sha(source/'complete.json')==a.complete_sha
        result=dict(passed=True,step=6000,parent_initial_step=20000,source=str(source),
            source_complete_sha256=a.complete_sha,identity=identity,files=files,
            parameter_elements=numel,parameter_tensors=1672,persistent_buffers=36,
            runtime_parameter_audit_sha256=sha(out/'runtime-parameter-audit.json'),
            gpu_used=False,training_modified=False,optimizer_transferred=False,time=time.time())
        save(out/'packet_complete.json',result)
        status('completed',files=len(files),bytes=sum(x['bytes'] for x in files))
        print(json.dumps({'passed':True,'files':len(files),'parameter_elements':numel}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'phase':'needs_diagnosis','error':repr(error)}),flush=True)
        raise
