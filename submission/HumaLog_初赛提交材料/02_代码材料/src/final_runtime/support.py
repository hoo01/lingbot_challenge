"""Thin raw-HDF5/LoRA runner around the unchanged pinned official LingBot modules.

All data is authorized clean training data. This is a bounded development run,
not a validation or task-success evaluation. Checkpoints need the pinned base.
"""
import argparse
import bisect
from collections import Counter, defaultdict
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import yaml

ASSETS = Path('/mnt/raid1/lingbotvla_smoke_20261009')
ROOT = Path('/mnt/raid1/lingbotvla_longtrain_20261010')
REPO = ASSETS/'code/lingbot-vla-v2'
RANK = int(os.environ.get('RANK',0))
WORLD = int(os.environ.get('WORLD_SIZE',1))
RUN = None

def identity():
    files={n:hashlib.sha256((ROOT/'artifacts'/n).read_bytes()).hexdigest() for n in
        ('manifest.json','norm_stats.json','norm_stats.provenance.json','robotwin_relative.yaml')}
    files['train.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    files['base_revision']='11c703bf6a5c1f45b3b69168482da11fdbba53d7'
    files['upstream_revision']='be969b8fd117fb70550c5d4bf4bc328211b5b1b6'
    return files

def emit(event, **values):
    row = dict(event=event,rank=RANK,time=time.time(),**values)
    print(json.dumps(row),flush=True)
    with (RUN/f'events_rank{RANK}.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
        f.flush()

def local(x):
    return x.to_local() if hasattr(x,'to_local') else x

def category(name):
    if '.lora_' in name: return 'language_lora'
    if '.qwenvl.' in name and '.visual.' in name: return 'vision'
    if '.qwen_expert.' in name: return 'action_expert'
    return 'projections_alignment'

def config_for_run(args):
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    cli = yaml.safe_load((REPO/'configs/vla/robotwin/robotwin.yaml').read_text())
    cli['model'].update(model_path=str(ASSETS/'models/lingbot-vla-v2-6b'),
                        tokenizer_path=str(ASSETS/'models/Qwen3-VL-4B-Instruct'))
    cli['train'].update(freeze_vision_encoder=False,train_expert_only=False,
        skip_auxiliary_loss=False,use_compile=False,attention_implementation='eager',
        vit_attn_implementation='sdpa',precompute_grid_thw=False,
        enable_gradient_checkpointing=True,enable_full_shard=True,enable_fp32=True,
        action_fp32=True,loss_type='fm',optimizer='muon',use_moe_expert_lr=False,
        muon_exclude_name_patterns=['lora_'],muon_momentum=0.95,muon_nesterov=True,
        muon_ns_steps=5,muon_adjust_lr_fn='match_rms_adamw',lr=3e-5,lr_min=3e-6,
        lr_warmup_ratio=0.05,lr_start=3e-6,weight_decay=0.01,
        micro_batch_size=args.batch,global_batch_size=WORLD*args.batch*args.accum,
        output_dir=str(RUN),max_steps=args.schedule_steps)
    for key in ('sequence_wise_loss_coeff','router_z_loss_coeff'):
        cli['train'][key]=float(cli['train'][key])
    align = cli['train']['align_params']
    align['depth'].update(moge_path=str(ASSETS/'models/moge-2-vitb-normal/model.pt'),
        morgbd_path=str(ASSETS/'models/teachers/depth/model.pt'))
    align['video'].update(ckpt_path=str(ASSETS/'models/teachers/dino_video/teacher_step_10000.pth'),
        config_path=str(ASSETS/'models/teachers/dino_video/config.yaml'))
    cli['data'].update(use_future_image=True,img_size=256,
        dataset_format='official_raw_hdf5',train_path=str(ROOT/'artifacts/manifest.json'),
        norm_stats_file=str(ROOT/'artifacts/norm_stats.json'),
        robot_config_root=str(ROOT/'artifacts'),robot_config_path=str(ROOT/'artifacts/robotwin_relative.yaml'),
        joints=[str(x) for x in cli['data']['joints']],
        norm_type=[str({'arm.position':'meanstd'}),str({'end.position':'meanstd'}),
                   str({'effector.position':'minmax'})])
    config=LingbotVLAV2Config(**{**cli['model'],**cli['train']})
    config.skip_auxiliary_loss=False
    config.use_cache=False
    if RANK==0:
        saved_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        receipt=dict(cli=cli,args=saved_args,base_revision='11c703bf6a5c1f45b3b69168482da11fdbba53d7',
            upstream_revision='be969b8fd117fb70550c5d4bf4bc328211b5b1b6',
            norm_sha256=hashlib.sha256((ROOT/'artifacts/norm_stats.json').read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            group_lrs={'vision':3e-6,'language_lora':1e-4,'other':3e-5},identity=identity(),
            experiment='clean2500_fullvision_action_lora_dualteacher_per_horizon',
            precision='FP32 master parameters; official FSDP per-module mixed compute')
        (RUN/'config.json').write_text(json.dumps(receipt,indent=2))
        (RUN/'train_source.py').write_bytes(Path(__file__).read_bytes())
        (RUN/'lingbotvla_cli.yaml').write_text(yaml.safe_dump(cli,sort_keys=False))
    return config,cli

class Clean2500(Dataset):
    def __init__(self,config,cli):
        from lingbotvla.models import build_processor
        from lingbotvla.data.vla_data.utils import FeatureTransform
        self.records=json.loads((ROOT/'artifacts/manifest.json').read_text())['records']
        assert len(self.records)==2500 and all(r['split']=='train' for r in self.records)
        self.ends=np.cumsum([r['frames'] for r in self.records]).tolist()
        self.verified=set()
        self.transform=FeatureTransform(str(ROOT/'artifacts/robotwin_relative.yaml'),
            SimpleNamespace(**cli['data']),config,build_processor(config.tokenizer_path),
            norm_stats_path=str(ROOT/'artifacts/norm_stats.json'),chunk_size=50,
            image_augment=True,use_depth_align=True,use_future_image=True)
    def __len__(self): return self.ends[-1]
    def __getitem__(self,draw):
        from lingbot_competition.data import read_joints,action_chunk,CAMERAS,decode_rgb
        from torchvision.transforms.v2 import Resize
        # A stateless draw makes data, instruction and augmentation reproducible on resume.
        rng=np.random.default_rng(20261010+draw)
        torch.manual_seed(20261010+draw)
        random.seed(20261010+draw)
        np.random.seed((20261010+draw) % 2**32)
        index=int(rng.integers(self.ends[-1]))
        ep=bisect.bisect_right(self.ends,index)
        t=index-(self.ends[ep-1] if ep else 0)
        record=self.records[ep]
        path=ASSETS/record['path']
        if ep not in self.verified:
            deadline=time.monotonic()+1800
            last_identity=None
            while True:
                identity=(path.stat().st_size,path.stat().st_mtime_ns) if path.exists() else None
                if identity is not None and identity!=last_identity:
                    with path.open('rb') as f:
                        if hashlib.file_digest(f,'sha256').hexdigest()==record['sha256']:
                            self.verified.add(ep)
                            break
                    last_identity=identity
                assert time.monotonic()<deadline, f'Data copy timed out: {path}'
                time.sleep(1)
        with h5py.File(path,'r') as h5:
            q=read_joints(h5)
            action,pad=action_chunk(q,t,50)
            future=min(t+49,len(q)-1)
            item={'observation.state':torch.from_numpy(q[t].copy()),
                  'action':torch.from_numpy(action),'action_is_pad':torch.from_numpy(pad),
                  'task':record['instructions'][int(rng.integers(len(record['instructions'])))]}
            for camera,key in CAMERAS.items():
                pair=[]
                for frame in (t,future):
                    im=torch.from_numpy(decode_rgb(h5[f'observation/{camera}/rgb'][frame])).permute(2,0,1).float()
                    pair.append(Resize((256,256),antialias=True)(im))
                item[key]=torch.stack(pair)
        batch=self.transform.apply(item)
        batch['joint_mask'] &= ~batch['action_is_pad'].bool().unsqueeze(-1)
        # Test the actual official inverse, not a duplicate implementation.
        restored=self.transform.unapply({k:batch[k].clone() for k in
            ('actions','state','action_joint_mask','state_joint_mask')})['action']
        assert torch.allclose(restored.float(),torch.from_numpy(action),atol=2e-5,rtol=1e-5)
        batch={k:v for k,v in batch.items() if isinstance(v,torch.Tensor)}
        batch['_meta']=json.dumps(dict(task=record['task'],episode=record['episode'],frame=t,draw=draw))
        return batch

def build_student(config):
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.loader import CustomizedModelingLoader
    from peft import LoraConfig,inject_adapter_in_model
    apply_lingbot_qwen3_vl_patch();apply_lingbot_qwen2_patch()
    model=CustomizedModelingLoader(LingbotVlaV2Policy).load_model(
        {'config':config,'torch_dtype':torch.float32},weights_path=str(ASSETS/'models/lingbot-vla-v2-6b'),
        init_device='cuda',post_training=True,adanorm_time=True)
    vlm=model.model.qwenvl_with_expert.qwenvl
    vlm.requires_grad_(False)
    targets=[n for n,m in vlm.named_modules() if isinstance(m,torch.nn.Linear)
             and n.startswith('model.language_model.layers.')
             and n.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}]
    assert len(targets)==252
    inject_adapter_in_model(LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,target_modules=targets,bias='none'),vlm)
    counts=Counter()
    for name,p in model.named_parameters():
        enabled='.qwenvl.' not in name or '.visual.' in name or '.lora_' in name
        p.requires_grad_(enabled)
        counts[category(name) if enabled else 'frozen_language']+=p.numel()
    assert counts['vision']==415347712
    emit('parameters',counts=dict(counts),allocated_gib=torch.cuda.memory_allocated()/2**30)
    return model

def make_optimizer(student,cli):
    from lingbotvla.optim.optimizer import build_muon_optimizer
    optimizer=build_muon_optimizer(student,SimpleNamespace(**cli['train']),lr=3e-5,weight_decay=0.01)
    names={id(p):n for n,p in student.named_parameters()}
    seen=set(); report=[]
    for inner in optimizer.optimizers:
        split=[]
        for group in inner.param_groups:
            buckets=defaultdict(list)
            for p in group['params']:
                name=names[id(p)];cat=category(name)
                assert id(p) not in seen and p.requires_grad
                seen.add(id(p))
                if cat=='language_lora': assert type(inner).__name__=='AdamW'
                buckets[3e-6 if cat=='vision' else 1e-4 if cat=='language_lora' else 3e-5].append(p)
            for lr,params in buckets.items():
                split.append({**group,'params':params,'lr':lr})
                report.append(dict(optimizer=type(inner).__name__,lr=lr,
                    parameters=sum(p.numel() for p in params),tensors=len(params)))
        inner.param_groups=split
    assert seen=={id(p) for p in student.parameters() if p.requires_grad}
    emit('optimizer_groups',groups=report)
    return optimizer

def targets_for_batch(align,batch,teachers=None):
    from lingbotvla.models.vla.vision_models.module_utils import build_depth_model,build_video_model,get_depth_target,get_video_target
    if teachers is None:
        # Teacher construction must not consume the restored student's RNG stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            moge,depth=build_depth_model(align);video=build_video_model(align['video'])
        teachers=(moge,depth,video)
        assert all(not p.requires_grad for m in teachers for p in m.parameters())
    current=batch.pop('pil_images');future=batch.pop('future_pil_images')
    bundles=[]
    # Limit teacher activation memory independently of the student's micro-batch.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i in range(len(current)):
            cur=current[i:i+1];fut=future[i:i+1]
            dt,_=get_depth_target('MoRGBD',teachers[:2],cur)
            fdt,_=get_depth_target('MoRGBD',teachers[:2],fut)
            vt=get_video_target(teachers[2],cur,fut,align['video'])
            bundles.append(dict(depth_targets=dt,future_depth_targets=fdt,
                future_video_targets=vt['patch'],future_video_cls_targets=vt.get('cls'),
                future_video_current_patch=vt['current_patch']))
    targets={k:torch.cat([x[k] for x in bundles]) if bundles[0][k] is not None else None for k in bundles[0]}
    assert all(x is None or (not x.requires_grad and torch.isfinite(x).all()) for x in targets.values())
    return teachers,targets

def audit_clip(model):
    norms={k:torch.zeros((),device='cuda') for k in ('vision','language_lora','action_expert','projections_alignment')}
    missing=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None
        elif p.grad is None:
            missing.append(name)
        else:
            norms[category(name)]+=torch.linalg.vector_norm(local(p.grad).float()).square()
    for v in norms.values():
        dist.all_reduce(v)
        assert torch.isfinite(v) and v>0
    assert not missing,missing
    total=sum(norms.values()).sqrt()
    scale=(1/(total+1e-6)).clamp(max=1.)
    for p in model.parameters():
        if p.grad is not None: local(p.grad).mul_(scale)
    return {k:float(v.sqrt()) for k,v in norms.items()}

def pack(x):
    if isinstance(x,torch.Tensor):
        return dict(tensor=local(x.detach()).cpu(),dtensor=hasattr(x,'placements'),global_shape=list(x.shape),
                    shard_dims=[getattr(p,'dim',None) for p in getattr(x,'placements',())])
    return x

def persistent_buffers(model):
    modules=dict(model.named_modules())
    return {n:b for n,b in model.named_buffers()
            if n.rsplit('.',1)[-1] not in modules[n.rsplit('.',1)[0] if '.' in n else '']._non_persistent_buffers_set}

def save_checkpoint(model,optimizer,scheduler,step,args):
    folder=RUN/f'checkpoint_{step:06d}';folder.mkdir(exist_ok=True)
    assert not (folder/'complete.json').exists(), 'Never overwrite a completed checkpoint'
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    names={id(p):n for n,p in params.items()}
    payload=dict(step=step,world=WORLD,batch=args.batch,accum=args.accum,schedule_steps=args.schedule_steps,
         identity=identity(),
         norm_sha256=hashlib.sha256((ROOT/'artifacts/norm_stats.json').read_bytes()).hexdigest(),
        parameters={n:pack(p) for n,p in params.items()},
        buffers={n:pack(b) for n,b in persistent_buffers(model).items()},
        optimizer={n:{k:pack(v) for k,v in optimizer.state.get(p,{}).items()} for n,p in params.items()},
        groups=[[{**{k:v for k,v in g.items() if k!='params'},'params':[names[id(p)] for p in g['params']]}
                 for g in opt.param_groups] for opt in optimizer.optimizers],
        scheduler=scheduler.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
    path=folder/f'rank{RANK}.pt'
    with path.with_suffix('.partial').open('wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    path.with_suffix('.partial').replace(path)
    del payload;gc.collect()
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    restored=torch.load(path,map_location='cpu',weights_only=True)
    for n,p in params.items(): assert torch.equal(restored['parameters'][n]['tensor'],local(p.detach()).cpu()),n
    for n,b in persistent_buffers(model).items(): assert torch.equal(restored['buffers'][n]['tensor'],local(b.detach()).cpu()),n
    for n,p in params.items():
        for k,v in optimizer.state.get(p,{}).items():
            stored=restored['optimizer'][n][k]
            assert torch.equal(stored['tensor'],local(v.detach()).cpu()) if isinstance(v,torch.Tensor) else stored==v
    del restored;gc.collect()
    receipt=dict(step=step,path=str(path),bytes=path.stat().st_size,sha256=digest,readback_equal=True)
    (folder/f'rank{RANK}.json').write_text(json.dumps(receipt,indent=2))
    dist.barrier()
    if RANK==0:
        import shutil
        for name in ('manifest.json','norm_stats.json','norm_stats.provenance.json','robotwin_relative.yaml'):
            shutil.copy2(ROOT/'artifacts'/name,folder/name)
        for name in ('config.json','lingbotvla_cli.yaml','train_source.py'):
            shutil.copy2(RUN/name,folder/name)
        (folder/'complete.json').write_text(json.dumps(dict(step=step,world=WORLD,identity=identity()),indent=2))
    emit('checkpoint_verified',**receipt)
    return receipt

def restore_checkpoint(model,optimizer,scheduler,folder,args):
    from torch.distributed.tensor import DTensor
    assert (folder/'complete.json').exists()
    path=folder/f'rank{RANK}.pt'
    receipt=json.loads((folder/f'rank{RANK}.json').read_text())
    with path.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256'],f'Checkpoint hash mismatch: {path}'
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['identity']==identity(), 'Checkpoint data/config/code/base identity mismatch'
    assert (state['world'],state['batch'],state['accum'],state['schedule_steps'])==(WORLD,args.batch,args.accum,args.schedule_steps)
    assert state['norm_sha256']==hashlib.sha256((ROOT/'artifacts/norm_stats.json').read_bytes()).hexdigest()
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    def unpack(v,p):
        if not isinstance(v,dict) or 'tensor' not in v: return v
        tensor=v['tensor'].to(p.device) if v['dtensor'] or v['tensor'].ndim else v['tensor']
        if v['dtensor']:
            assert list(p.shape)==v['global_shape']
            return DTensor.from_local(tensor,p.device_mesh,p.placements,run_check=False,shape=p.shape,stride=p.stride())
        return tensor
    with torch.no_grad():
        for n,p in params.items(): local(p).copy_(state['parameters'][n]['tensor'])
        for n,b in persistent_buffers(model).items(): local(b).copy_(state['buffers'][n]['tensor'])
    for opt,groups in zip(optimizer.optimizers,state['groups']):
        assert len(opt.param_groups)==len(groups)
        for target,source in zip(opt.param_groups,groups):
            assert [id(p) for p in target['params']]==[id(params[n]) for n in source['params']]
            target.update({k:v for k,v in source.items() if k!='params'})
            for n in source['params']:
                p=params[n];opt.state[p]={k:unpack(v,p) for k,v in state['optimizer'][n].items()}
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    step=state['step'];del state;gc.collect()
    emit('checkpoint_restored',step=step,path=str(folder))
    return step

def main():
    global RUN
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--phase',choices=['data','train'],default='train')
    parser.add_argument('--steps',type=int,default=15000)
    parser.add_argument('--schedule-steps',type=int,default=15000)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--save-every',type=int,default=500)
    parser.add_argument('--accum',type=int,default=1)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--no-save',action='store_true')
    args=parser.parse_args()
    assert args.steps<=args.schedule_steps and args.batch>0 and args.accum>0
    RUN=ROOT/'runs'/args.run;RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'config.json').exists():
        assert args.resume is not None, 'Existing run requires an explicit checkpoint'
        prior=json.loads((RUN/'config.json').read_text())
        assert prior['identity']==identity()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    torch.set_num_threads(6);torch.manual_seed(20261010)
    config,cli=config_for_run(args)
    dataset=Clean2500(config,cli)
    if args.phase=='data':
        for draw in [0,1,2,10,100,1000,10000]:
            batch=dataset[draw]
            emit('data_roundtrip',metadata=json.loads(batch['_meta']),shapes={k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)})
        emit('data_passed',episodes=len(dataset.records),windows=len(dataset))
        return
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.optim import build_lr_scheduler
    init_parallel_state(dp_size=WORLD,dp_mode='fsdp2')
    student=build_student(config)
    student=build_parallelize_model(student,enable_full_shard=True,enable_mixed_precision=True,
        enable_fp32=True,enable_gradient_checkpointing=True,init_device='cuda',
        basic_modules=student._no_split_modules,fsdp_llm_blocks=False,vlm_fsdp=True,enable_reentrant=False)
    student.train()
    optimizer=make_optimizer(student,cli)
    # Match the official trainer, including zero-coefficient global load monitoring.
    if cli['train'].get('use_moe',False):
        from lingbotvla.models.vla.lingbot_vla.moe_load_balance import build_moe_load_balance_hook
        coeff=float(cli['train'].get('bias_update_speed',0.001))
        interval=int(cli['train'].get('bias_update_interval',1))
        assert interval==1, 'Windowed hook state requires an extended resume format.'
        optimizer.register_step_pre_hook(build_moe_load_balance_hook(student,coeff=coeff,
            bias_centering=cli['train'].get('bias_centering',False),update_interval=interval))
        emit('moe_load_balance_hook',coefficient=coeff,interval=interval)
    scheduler=build_lr_scheduler(optimizer,train_steps=args.schedule_steps,lr=3e-5,lr_min=3e-6,
        lr_warmup_ratio=0.05,lr_start=3e-6,lr_decay_style='cosine')
    start_step=restore_checkpoint(student,optimizer,scheduler,args.resume,args) if args.resume else 0
    effective=WORLD*args.batch*args.accum
    draw_ids=[base+RANK*args.batch+j
              for base in range(start_step*effective,args.steps*effective,WORLD*args.batch)
              for j in range(args.batch)]
    loader=DataLoader(dataset,batch_size=args.batch,sampler=draw_ids,num_workers=1,
        multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2,pin_memory=True)
    iterator=iter(loader)
    teachers=None;metrics=[];receipt=None
    for step in range(start_step,args.steps):
        optimizer.zero_grad(set_to_none=True)
        dist.barrier();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        loss_sums=np.zeros(6);metas=[]
        for micro in range(args.accum):
            source=next(iterator);metas.extend(json.loads(x) for x in source.pop('_meta'))
            batch={k:v.cuda(non_blocking=True) for k,v in source.items()}
            teachers,targets=targets_for_batch(cli['train']['align_params'],batch,teachers)
            outputs=student(**batch,**targets)
            assert torch.isfinite(outputs[0])
            assert all(torch.is_tensor(v) and torch.isfinite(v) and v>=0 for v in outputs[2:5])
            (outputs[0]/args.accum).backward()
            loss_sums+=np.array([float(v.detach()) if torch.is_tensor(v) else float(v) for v in outputs[:6]])/args.accum
            del outputs,targets,batch,source
        norms=audit_clip(student)
        sentinels={}
        for name,p in student.named_parameters():
            cat=category(name)
            if p.requires_grad and cat not in sentinels and local(p.grad).numel() and local(p.grad).abs().max()>0:
                index=local(p.grad).abs().reshape(-1).argmax()
                sentinels[cat]=(p,index,local(p.detach()).reshape(-1)[index].clone())
        assert len(sentinels)==4
        optimizer.step();scheduler.step()
        updates={k:float((local(p.detach()).reshape(-1)[idx]-old).abs()) for k,(p,idx,old) in sentinels.items()}
        assert all(np.isfinite(v) for v in updates.values()),updates
        if step<3: assert all(v>0 for v in updates.values()),updates
        assert all(p.grad is None for m in teachers for p in m.parameters())
        torch.cuda.synchronize()
        averaged=torch.tensor(loss_sums,device='cuda',dtype=torch.float64)
        dist.all_reduce(averaged);averaged/=WORLD
        row=dict(step=step+1,seconds=time.monotonic()-start,losses=averaged.tolist(),
            learning_rates=[float(g['lr']) for g in optimizer.param_groups],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,gradient_norms=norms,
            updates=updates,samples=metas)
        emit('step',**row);metrics.append(row)
        if len(metrics)>100: metrics.pop(0)
        if not args.no_save and (step+1)%args.save_every==0:
            optimizer.zero_grad(set_to_none=True)
            receipt=save_checkpoint(student,optimizer,scheduler,step+1,args)
    optimizer.zero_grad(set_to_none=True)
    if not args.no_save and args.steps%args.save_every:
        receipt=save_checkpoint(student,optimizer,scheduler,args.steps,args)
    (RUN/f'result_rank{RANK}.json').write_text(json.dumps(dict(status='passed',start_step=start_step,
        final_step=args.steps,batch=args.batch,effective_batch=effective,metrics=metrics,checkpoint=receipt),indent=2))
    emit('complete',steps=args.steps)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':
    main()
