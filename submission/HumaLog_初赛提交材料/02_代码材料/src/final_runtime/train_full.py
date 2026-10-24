"""Dense Lab20k warm start, four-rank FP32-master full SFT with bounded gates."""
import argparse
from collections import Counter
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
import yaml
import checkpoint as ckpt
from data import OfficialClean
from stage_sampler import build_loader,next_batch
from aug6000.reader import AugmentedOfficialClean
from aug6000.foreground import ForegroundCache
from aug6000.policy import set_learning_rates,FIRST_DRAW
from aug6000.integration import forward_batch
from images import tensor_sha
from policy import (ROOT, SOURCE, UPSTREAM, ASSETS, STEPS, SAVE_EVERY, GLOBAL_BATCH,
                    SEED, PEAKS, NORM_SHA, START, category,
                    sha, atomic_json)
from recipe import make_recipe

RANK = int(os.environ.get('RANK', 0))
RUN = None


def emit(event, **values):
    row = dict(event=event, rank=RANK, time=time.time(), **values)
    print(json.dumps(row), flush=True)
    if RUN:
        with (RUN / f'events_rank{RANK}.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    return row


def identity(args):
    files=('manifest.json','norm_stats.json','norm_stats.provenance.json','robotwin_relative.yaml',
           'input_inventory.json','data-filter-plan.json','source-migration-binding.json')
    result={name:sha(ROOT/'artifacts'/name) for name in files}
    result['code']={name:sha(ROOT/'train_geometryfix11000_requested'/name) for name in ('policy.py','recipe.py','images.py','data.py',
        'checkpoint.py','train_full.py','support.py','base_recipe.py','filtering.py','official_sampler.py','teacher_targets.py')}
    result.update(source_step=8000,parent_initial_step=20000,dense_full_sft=True,seed=SEED,total_steps=STEPS,
        microbatch=args.batch,accum=args.accum,global_batch=GLOBAL_BATCH,learning_rates=PEAKS,
        optimizer='AdamW_inherited_state',original_warmup=500,phase_start=8000,
        new_sampler_rng_phase=True,source_complete_sha256='2e71c096ecfbef0c92adb9e811e8dbfb9a53b5c98e2e8233073bb0aad8b45619')
    result['code']={str(p.relative_to(ROOT/'train_geometryfix11000_requested')):sha(p) for p in sorted((ROOT/'train_geometryfix11000_requested').rglob('*.py')) if '__pycache__' not in p.parts}
    result.update(source_step=10000,phase_start=10000,total_steps=16000,new_stage_updates=6000,global_batch=192,microbatch=16,accum=3,original_warmup=500,new_phase_rewarm=300,final_lr_ratio=.3,new_sampler_rng_phase=False,foreground_manifest=sha(ROOT/'foreground/manifest.reviewed.json'),recipe=sha(ROOT/'artifacts/geometryfix11000_requested/aug6000-recipe.json'),source_complete_sha256=__import__('policy').PARENT_COMPLETE_SHA)
    result.update(source_step=11000,geometry_fix_boundary=11000,geometry_global_probability=.45,
        geometry_contract='primary_affine_global45_scale09_11_translate_lt6percent_v1',
        source_binding=sha(ROOT/'artifacts/geometryfix11000_requested/source-binding.json'))
    assert result['norm_stats.json']==NORM_SHA
    return result


def build_student(config):
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from lingbotvla.models.loader import CustomizedModelingLoader
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    # Construct on meta, stream one source shard at a time directly to this GPU.
    # Nonpersistent buffers retain their native constructor values.
    model = CustomizedModelingLoader(LingbotVlaV2Policy).load_model(
        {'config': config, 'torch_dtype': torch.float32}, weights_path=str(SOURCE),
        init_device='cuda', empty_init=True, post_training=True, adanorm_time=True)
    index = json.loads((SOURCE / 'model.safetensors.index.json').read_text())['weight_map']
    expected = json.loads((SOURCE / 'runtime-parameter-audit.json').read_text())['trained_parameter_sha256']
    params = dict(model.named_parameters())
    buffers = ckpt.persistent_buffers(model)
    assert len(params) == len(expected) == 1672
    assert set(index) == set(params) | set(buffers) and len(buffers) == 36
    assert sum(p.numel() for p in params.values()) == 6375906359
    loaded = set()
    actual_hashes = {}
    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    for shard in sorted(set(index.values())):
        with safe_open(SOURCE / shard, framework='pt', device='cpu') as source:
            for name in source.keys():
                assert name in index and index[name] == shard and name not in loaded
                value = source.get_tensor(name)
                assert value.dtype == torch.float32 and torch.isfinite(value).all(), name
                set_module_tensor_to_device(model, name, device, value=value, dtype=torch.float32)
                actual = model.get_parameter(name) if name in params else model.get_buffer(name)
                digest = tensor_sha(actual)
                assert digest == (expected[name] if name in params else tensor_sha(value)), name
                actual_hashes[name] = digest
                loaded.add(name)
                del value, actual
        gc.collect()
    assert loaded == set(index)
    model.to(device)
    model.requires_grad_(True)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters())
    assert not any('.lora_' in name for name, _ in model.named_parameters())
    proof = dict(passed=True, source_step=8000, source=str(SOURCE),
        parameter_elements=6375906359, parameter_tensors=1672, persistent_buffers=36,
        all_actual_parameter_values_equal_source=True, actual_tensor_sha256=actual_hashes,
        master_dtype='float32', new_lora_factors=0, local_rank=int(os.environ['LOCAL_RANK']),
        actual_cuda_device=torch.cuda.current_device(), device_total_gib=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30,
        device_uuid=str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), 'uuid', '')))
    atomic_json(RUN / f'initial_values_rank{RANK}.json', proof)
    emit('source_loaded_exact', parameters=1672, buffers=36)
    return model


def make_optimizer(model):
    groups = []
    seen = set()
    for label, peak in PEAKS.items():
        parameters = [p for name, p in model.named_parameters() if category(name) == label]
        assert parameters and all(p.requires_grad for p in parameters)
        assert not any(id(p) in seen for p in parameters)
        seen.update(map(id, parameters))
        groups.append(dict(params=parameters, peak_lr=peak, lr=peak / 500, label=label))
    assert seen == {id(p) for p in model.parameters()}
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8,
                                  weight_decay=0., foreach=False, fused=False)
    from lingbotvla.models.vla.lingbot_vla.moe_load_balance import build_moe_load_balance_hook
    optimizer.register_step_pre_hook(build_moe_load_balance_hook(
        model, coeff=0., bias_centering=False, update_interval=1))
    emit('optimizer_groups', groups=[dict(label=g['label'], peak_lr=g['peak_lr'],
        parameters=sum(p.numel() for p in g['params']), tensors=len(g['params'])) for g in groups],
        optimizer='AdamW', expert_lr_multiplier=1., weight_decay=0.)
    return optimizer


def gradient_audit(model):
    from torch.distributed.tensor import DTensor, Shard
    labels = [*PEAKS, 'language_q', 'language_k', 'language_v', 'language_o', 'language_mlp']
    sums = torch.zeros(len(labels), device='cuda', dtype=torch.float64)
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        value = torch.linalg.vector_norm(ckpt.local(parameter.grad).float()).double().square()
        if not isinstance(parameter.grad, DTensor) or not any(isinstance(p, Shard) for p in parameter.grad.placements):
            value = value / 4
        group = category(name)
        sums[labels.index(group)] += value
        if group == 'language':
            extra = next((f'language_{short}' for short in ('q', 'k', 'v', 'o')
                          if name.endswith(f'.{short}_proj.weight')), None)
            if '.mlp.' in name:
                extra = 'language_mlp'
            if extra:
                sums[labels.index(extra)] += value
    assert not missing, missing
    dist.all_reduce(sums)
    assert torch.isfinite(sums).all() and (sums > 0).all(), dict(zip(labels, sums.tolist()))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
    norm = float(norm.full_tensor() if hasattr(norm, 'full_tensor') else norm)
    assert np.isfinite(norm)
    return dict(zip(labels, sums.sqrt().tolist())), norm


def update_sentinels(model):
    sentinels = {}
    for name, parameter in model.named_parameters():
        group = category(name)
        if group == 'language':
            if '.self_attn.' in name and name.endswith(('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight')):
                group = 'language_' + name.rsplit('.', 2)[-2].split('_')[0]
            elif '.mlp.' in name:
                group = 'language_mlp'
        if group in sentinels or parameter.grad is None:
            continue
        grad = ckpt.local(parameter.grad).reshape(-1)
        if not grad.numel():
            continue
        indices = torch.topk(grad.abs(), min(16, grad.numel())).indices
        sentinels[group] = (parameter, indices, ckpt.local(parameter.detach()).reshape(-1)[indices].clone())
    return sentinels


def batch_fingerprint(batch):
    return {key: tensor_sha(value) for key, value in sorted(batch.items()) if isinstance(value, torch.Tensor)}


def accounting(values):
    numbers = [float(v.detach()) for v in values]
    assert all(np.isfinite(v) for v in numbers)
    epsilon = max(torch.finfo(v.dtype).eps for v in values)
    bound = (.5 * epsilon + 4 * torch.finfo(torch.float32).eps) * (
        abs(numbers[0]) + sum(abs(v) for v in numbers[1:]))
    assert abs(numbers[0] - sum(numbers[1:])) <= bound, (numbers, bound)


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--role', choices=['benchmark', 'resume-check', 'formal'], required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--accum', type=int, required=True)
    parser.add_argument('--end-step', type=int, default=STEPS)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--engineering-saves', default='')
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.resume is not None
    assert (args.batch,args.accum)==(16,3) and args.batch*args.accum*4==GLOBAL_BATCH
    assert 1 <= args.end_step <= STEPS
    if args.role == 'formal':
        assert args.end_step == STEPS and not args.engineering_saves
        ready = json.loads((ROOT / 'artifacts/geometryfix11000_requested/ready-for-formal.json').read_text())
        assert ready['passed'] and ready['microbatch'] == args.batch and ready['accum'] == args.accum
    RUN = ROOT / 'runs' / args.run
    RUN.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    run_identity = identity(args)
    if RANK == 0 and (RUN / 'config.json').exists():
        assert args.resume and json.loads((RUN / 'config.json').read_text())['identity'] == run_identity
    if args.role == 'formal':
        assert ready['identity'] == run_identity
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    cli = make_recipe(RUN, args.batch, args.accum)
    config = LingbotVLAV2Config(**{**cli['model'], **cli['train']})
    config.skip_auxiliary_loss = False
    config.use_cache = False
    if RANK == 0:
        atomic_json(RUN / 'config.json', dict(identity=run_identity, args=vars(args) | {
            'resume': str(args.resume) if args.resume else None}, cli=cli,
            actual_stage_steps=6000, source_step=11000, phase_source_step=10000, save_every=SAVE_EVERY,
            trainable_scope='all_student_parameters', teacher_scope='frozen',
            scheduler='phase10000_300rewarm_5700cosine_to30percent_v1'))
        (RUN / 'lingbotvla_cli.yaml').write_text(yaml.safe_dump(cli, sort_keys=False))
        atomic_json(RUN / 'model_config.json', config.to_dict())
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    assert dist.get_world_size() == 4
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    init_parallel_state(dp_size=4, dp_mode='fsdp2')
    model = build_student(config)
    model = build_parallelize_model(model, enable_full_shard=True,
        enable_mixed_precision=True, enable_fp32=True, enable_gradient_checkpointing=True,
        init_device='cuda', basic_modules=model._no_split_modules,
        fsdp_llm_blocks=False, vlm_fsdp=True, enable_reentrant=False)
    model.train()
    layers = model.model.qwenvl_with_expert.qwenvl.model.language_model.layers
    assert len(layers) == 36 and all(hasattr(layer, 'unshard') for layer in layers)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters())
    optimizer = make_optimizer(model)
    from policy import PARENT_CHECKPOINT
    parent_boundary=args.resume is not None and args.resume.resolve()==PARENT_CHECKPOINT.resolve()
    cache=ForegroundCache(ROOT/'foreground/manifest.reviewed.json',run_identity['foreground_manifest'])
    dataset=AugmentedOfficialClean(OfficialClean(config,cli,audit_images=True),cache,run_identity['recipe'],allow_parent_boundary_restore=parent_boundary)
    loader,sampler,generator=build_loader(dataset,RANK,allow_parent_boundary=parent_boundary)
    start, saved_rng = (ckpt.restore(model, optimizer, args.resume, run_identity, args,
        loader, generator) if args.resume else (0, None))
    iterator = iter(loader)
    # Dedicated loader generator and forked teacher initialization do not consume model RNG.
    if saved_rng:
        ckpt.set_rng(saved_rng)
        emit('restored_exact', step=start, rng_sha256=tensor_sha(torch.cuda.get_rng_state()))
    import teacher_targets as support
    teachers = None
    metrics = []
    engineering = {int(v) for v in args.engineering_saves.split(',') if v}
    for completed in range(start, args.end_step):
        stop = torch.tensor(int(RANK == 0 and (ROOT / 'user-stop.json').exists()), device='cuda')
        dist.broadcast(stop, src=0)
        if int(stop):
            emit('user_stopped', completed_step=completed)
            break
        update = completed + 1
        optimizer.zero_grad(set_to_none=True)
        set_learning_rates(optimizer,update-10000)
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses, metadata, fingerprints, sample_losses = np.zeros(7), [], [], []
        rng_before = tensor_sha(torch.cuda.get_rng_state())
        for micro in range(args.accum):
            source, iterator = next_batch(loader, sampler, iterator)
            metadata.extend(json.loads(value) for value in source.pop('_meta'))
            if args.role != 'formal' or update <= 11003:
                fingerprints.append(batch_fingerprint(source))
            batch = {name: value.cuda(non_blocking=True) for name, value in source.items()}
            outputs,teachers=forward_batch(model,batch,cli['train']['align_params'],support.targets_for_batch,teachers)
            targets=None
            sample_losses.extend(outputs[6]['batch_mean_losses'].detach().float().cpu().tolist())
            values = [*outputs[:6], outputs[6]['router_z_loss']]
            accounting(values)
            (outputs[0] / args.accum).backward()
            losses += np.asarray([float(value.detach()) for value in values]) / args.accum
            del outputs, values, targets, batch, source
        expected_draws = [FIRST_DRAW+(completed-10000)*GLOBAL_BATCH+micro*4*args.batch+RANK*args.batch+j
                          for micro in range(args.accum) for j in range(args.batch)]
        assert [m['draw'] for m in metadata] == expected_draws
        norms, norm = gradient_audit(model)
        sentinels = update_sentinels(model) if update <= 11003 or update % 100 == 0 else {}
        optimizer.step()
        changes = {name: float((ckpt.local(p.detach()).reshape(-1)[indices] - old).abs().max())
                   for name, (p, indices, old) in sentinels.items()}
        assert all(np.isfinite(value) for value in changes.values())
        assert all(not p.requires_grad and p.grad is None for teacher in teachers for p in teacher.parameters())
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        values = torch.tensor(losses, device='cuda', dtype=torch.float64)
        dist.all_reduce(values)
        values /= 4
        row = dict(step=update, seconds=seconds, losses=values.tolist(),
            loss_labels=['total', 'flow_matching', 'depth', 'future_depth', 'video', 'sequence_wise', 'router_z'],
            learning_rates={g['label']: g['lr'] for g in optimizer.param_groups},
            gradient_norms=norms, gradient_norm_before_clip=norm, updates=changes,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            samples=metadata, sample_action_losses=sample_losses, input_sha256=fingerprints, cuda_rng_before=rng_before,
            cuda_rng_after=tensor_sha(torch.cuda.get_rng_state()), global_batch=GLOBAL_BATCH)
        if args.role != 'formal' or update <= 11003:
            row['all_rng_after']=ckpt.rng_fingerprint()
        emit('step', **row)
        if update == 11003 and args.role != 'formal':
            atomic_json(RUN / f'state11003_rank{RANK}.json', ckpt.fingerprint(model, optimizer))
        atomic_json(RUN / f'progress_rank{RANK}.json', row)
        metrics.append(row)
        metrics = metrics[-20:]
        if update in engineering or (args.role == 'formal' and update % SAVE_EVERY == 0):
            optimizer.zero_grad(set_to_none=True)
            saved = ckpt.save(model, optimizer, update, RUN, run_identity, args, loader, generator)
            emit('checkpoint_verified', **saved)
    else:
        atomic_json(RUN / f'result_rank{RANK}.json', dict(status='passed', final_step=args.end_step,
            requested_stage_steps=STEPS, microbatch=args.batch, accum=args.accum, global_batch=GLOBAL_BATCH,
            role=args.role, metrics=metrics, identity=run_identity))
        emit('complete', final_step=args.end_step, role=args.role)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
