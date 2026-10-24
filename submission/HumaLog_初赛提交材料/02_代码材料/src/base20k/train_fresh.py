"""Fresh official-base training with declared q/v LoRA and raw-HDF5 adapters.

Official model, optimizer, scheduler, MoE and FeatureTransform stay pinned.
The old runner is imported only for already verified serialization/teacher
helpers; its model builder, optimizer grouping and random frame reader never run.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
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
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
import yaml

import train as io
from fresh_recipe import (BASE_REVISION, UPSTREAM_REVISION, NORM_SHA256,
                          LORA_RANK, LORA_ALPHA, frame_location,
                          is_language_target, make_recipe)
from raw255_augmentation import install_raw255_augmentation

ASSETS = Path(os.environ.get('HUMALOG_ASSETS', '/mnt/raid1/lingbotvla_smoke_20261009'))
ROOT = Path(os.environ.get('HUMALOG_TRAIN_ROOT', '/mnt/raid1/lingbotvla_qv256_20261012'))
REPO = ASSETS / 'code/lingbot-vla-v2'
RANK = int(os.environ.get('RANK', 0))
WORLD = int(os.environ.get('WORLD_SIZE', 1))
RUN = None
SEED = 42


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity():
    files = {name: sha(ROOT / 'artifacts' / name) for name in (
        'manifest.json', 'norm_stats.json', 'norm_stats.provenance.json',
        'robotwin_relative.yaml', 'input_inventory.json')}
    files.update({name: sha(Path(__file__).parent / name) for name in (
        'train_fresh.py', 'fresh_recipe.py', 'raw255_augmentation.py', 'train.py')})
    files.update(base_revision=BASE_REVISION, upstream_revision=UPSTREAM_REVISION,
                 initialization='official_base_only_no_lab_weights', seed=SEED,
                 sampler='StatefulDistributedSampler_shuffle_drop_last_v1',
                 language_lora_rank=LORA_RANK, language_lora_alpha=LORA_ALPHA)
    assert files['norm_stats.json'] == NORM_SHA256
    return files


@contextmanager
def sample_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            yield np.random.default_rng(seed)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class IndexedClean2500(io.Clean2500):
    """The sampler's index selects the exact frame; it never draws a replacement."""
    def __init__(self, config, cli):
        from lingbotvla.data.vla_data import transform
        assert install_raw255_augmentation(transform) or hasattr(transform, '_raw255_augmentation_original')
        super().__init__(config, cli)
        self.epoch = 0
        inventory = json.loads((ROOT / 'artifacts/input_inventory.json').read_text())
        self.file_inventory = {r['path']: r for r in inventory['clean_episodes']}
        assert len(self.file_inventory) == 2500

    def state_dict(self):
        return {'epoch': self.epoch}

    def load_state_dict(self, state):
        self.epoch = int(state['epoch'])

    def __getitem__(self, index):
        index = int(index)
        ep, t = frame_location(self.ends, index)
        with sample_rng(SEED + self.epoch * len(self) + index) as rng:
            return self.read_frame(ep, t, index, rng)

    def read_frame(self, ep, t, index, rng):
        # Spawned workers import fresh modules; a patch in the parent does not
        # cross that boundary. Install it where each process actually applies it.
        from lingbotvla.data.vla_data import transform
        install_raw255_augmentation(transform)
        assert hasattr(transform, '_raw255_augmentation_original')
        from lingbot_competition.data import read_joints, action_chunk, CAMERAS, decode_rgb
        from torchvision.transforms.v2 import Resize
        record = self.records[ep]
        path = ASSETS / record['path']
        verified = self.file_inventory[str(path)]
        stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (verified['bytes'], verified['mtime_ns']), str(path)
        assert verified['sha256'] == record['sha256']
        with h5py.File(path, 'r') as h5:
            q = read_joints(h5)
            assert t < len(q) - 1
            action, pad = action_chunk(q, t, 50)
            future = min(t + 49, len(q) - 1)
            item = {'observation.state': torch.from_numpy(q[t].copy()),
                'action': torch.from_numpy(action), 'action_is_pad': torch.from_numpy(pad),
                'task': record['instructions'][int(rng.integers(len(record['instructions'])))]}
            for camera, key in CAMERAS.items():
                images = []
                for frame in (t, future):
                    image = torch.from_numpy(decode_rgb(h5[f'observation/{camera}/rgb'][frame]))
                    image = image.permute(2, 0, 1).float()
                    images.append(Resize((256, 256), antialias=True)(image))
                item[key] = torch.stack(images)
        batch = self.transform.apply(item)
        batch['joint_mask'] &= ~batch['action_is_pad'].bool().unsqueeze(-1)
        restored = self.transform.unapply({k: batch[k].clone() for k in
            ('actions', 'state', 'action_joint_mask', 'state_joint_mask')})['action']
        assert torch.allclose(restored.float(), torch.from_numpy(action), atol=2e-5, rtol=1e-5)
        for key in ('pil_images', 'future_pil_images'):
            pixels = batch[key]
            # These are raw teacher RGB, not yet teacher-normalized RGB.
            # The unchanged official teacher entry divides by 255 exactly once.
            assert torch.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 255
        result = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        result['_meta'] = json.dumps(dict(task=record['task'], episode=record['episode'],
            frame=t, index=index, epoch=self.epoch,
            raw255_adapter_installed=True,
            student_range=[float(batch['images'].min()), float(batch['images'].max())],
            teacher_current_unit_mean=float((batch['pil_images'] / 255).mean()),
            teacher_future_unit_mean=float((batch['future_pil_images'] / 255).mean())))
        return result


def build_student(config):
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.loader import CustomizedModelingLoader
    from peft import LoraConfig, inject_adapter_in_model
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    model = CustomizedModelingLoader(LingbotVlaV2Policy).load_model(
        {'config': config, 'torch_dtype': torch.float32},
        weights_path=str(ASSETS / 'models/lingbot-vla-v2-6b'),
        init_device='cuda', post_training=True, adanorm_time=True)
    vlm = model.model.qwenvl_with_expert.qwenvl
    vlm.requires_grad_(False)
    targets = [name for name, module in vlm.named_modules()
               if isinstance(module, torch.nn.Linear) and is_language_target(name)]
    assert len(targets) == 72, targets
    inject_adapter_in_model(LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, target_modules=targets, bias='none'), vlm)
    counts, dtypes = Counter(), Counter()
    for name, parameter in model.named_parameters():
        enabled = '.qwenvl.' not in name or '.visual.' in name or '.lora_' in name
        parameter.requires_grad_(enabled)
        counts[io.category(name) if enabled else 'frozen_language'] += parameter.numel()
        dtypes[f'{"trainable" if enabled else "frozen"}:{parameter.dtype}'] += parameter.numel()
    assert counts['vision'] == 415347712
    assert all(torch.count_nonzero(m.lora_B['default'].weight) == 0
               for m in vlm.modules() if hasattr(m, 'lora_B'))
    io.emit('parameters', counts=dict(counts), actual_dtypes=dict(dtypes),
        lora_targets=targets, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
        initialization='official_base_only_no_lab_weights',
        allocated_gib=torch.cuda.memory_allocated() / 2**30)
    return model


def make_optimizer(student, cli):
    from lingbotvla.optim.optimizer import build_muon_optimizer
    train = cli['train']
    optimizer = build_muon_optimizer(student, SimpleNamespace(**train),
        lr=train['lr'], weight_decay=train['weight_decay'])
    names = {id(p): name for name, p in student.named_parameters()}
    seen, report = set(), []
    expert_lr = train['lr'] * (train['token_num_experts'] / train['token_top_k'])**0.5
    for inner in optimizer.optimizers:
        for group in inner.param_groups:
            categories = Counter()
            for parameter in group['params']:
                name = names[id(parameter)]
                assert id(parameter) not in seen and parameter.requires_grad
                seen.add(id(parameter))
                categories[io.category(name)] += parameter.numel()
                routed = '.mlp.experts.' in name
                expected_lr = expert_lr if routed else train['lr']
                assert abs(group['lr'] - expected_lr) < 1e-12, (name, group['lr'], expected_lr)
            report.append(dict(optimizer=type(inner).__name__, lr=group['lr'],
                weight_decay=group['weight_decay'], categories=dict(categories),
                tensors=len(group['params'])))
    assert seen == {id(p) for p in student.parameters() if p.requires_grad}
    assert any(abs(g['lr'] - expert_lr) < 1e-12 for g in report)
    io.emit('optimizer_groups', groups=report, official_groups_unchanged=True,
        expert_lr_multiplier=(train['token_num_experts'] / train['token_top_k'])**0.5)
    return optimizer


def gradient_audit(student):
    from torch.distributed.tensor import DTensor, Shard
    norms = {name: torch.zeros((), device='cuda') for name in
             ('vision', 'language_lora', 'action_expert', 'projections_alignment')}
    missing = []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
        elif parameter.grad is None:
            missing.append(name)
        else:
            grad = parameter.grad
            squared = torch.linalg.vector_norm(io.local(grad).float()).square()
            # Report replicated tensors once; clipping itself uses the official routine below.
            if not isinstance(grad, DTensor) or not any(isinstance(p, Shard) for p in grad.placements):
                squared = squared / WORLD
            norms[io.category(name)] += squared
    assert not missing, missing
    for value in norms.values():
        dist.all_reduce(value)
        assert torch.isfinite(value) and value > 0
    return {name: float(value.sqrt()) for name, value in norms.items()}


def loss_accounting(values):
    """Check the official sum after FSDP independently rounds returned tensors.

    The pinned FSDP2 root has output_dtype=bfloat16. Its total and six
    components are rounded separately; recomputing that sum is not bit-exact.
    Backpropagation still uses the unchanged official total, values[0].
    """
    numbers = [float(value.detach()) for value in values]
    assert all(np.isfinite(value) for value in numbers), numbers
    eps = max(torch.finfo(value.dtype).eps for value in values)
    total, parts = numbers[0], sum(numbers[1:])
    magnitude = abs(total) + sum(abs(value) for value in numbers[1:])
    tolerance = (0.5 * eps + 4 * torch.finfo(torch.float32).eps) * magnitude
    difference = abs(total - parts)
    receipt = dict(total=total, component_sum=parts, difference=difference,
                   rounding_bound=tolerance, dtypes=[str(value.dtype) for value in values])
    assert difference <= tolerance, receipt
    return receipt


def build_loader(dataset, args):
    sampler = StatefulDistributedSampler(dataset, num_replicas=WORLD, rank=RANK,
                                         shuffle=True, seed=SEED)
    generator = torch.Generator().manual_seed(SEED + RANK)
    loader = StatefulDataLoader(dataset, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, multiprocessing_context='spawn',
        persistent_workers=False, prefetch_factor=2, pin_memory=True,
        drop_last=True, generator=generator)
    return loader, sampler, generator


def save_run(student, optimizer, scheduler, step, args, loader, generator, epoch):
    folder = RUN / f'checkpoint_{step:06d}'
    folder.mkdir(exist_ok=True)
    sidecar = folder / f'data_rank{RANK}.pt'
    temporary = sidecar.with_suffix('.partial')
    state = dict(step=step, epoch=epoch, identity=identity(),
                 loader=loader.state_dict(), generator=generator.get_state())
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(sidecar)
    readback = torch.load(sidecar, map_location='cpu', weights_only=True)
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    assert equal(state, readback), 'DataLoader state readback mismatch'
    (folder / f'data_rank{RANK}.json').write_text(json.dumps(dict(
        sha256=sha(sidecar), bytes=sidecar.stat().st_size, step=step,
        epoch=epoch, readback_equal=True), indent=2))
    # The shared writer emits complete.json only after all ranks finish readback.
    return io.save_checkpoint(student, optimizer, scheduler, step, args)


def restore_run(student, optimizer, scheduler, args, dataset, loader, sampler, generator):
    folder = args.resume
    sidecar = folder / f'data_rank{RANK}.pt'
    proof = json.loads(sidecar.with_suffix('.json').read_text())
    assert proof['sha256'] == sha(sidecar) and proof['readback_equal']
    state = torch.load(sidecar, map_location='cpu', weights_only=True)
    assert state['identity'] == identity()
    epoch = state['epoch']
    dataset.epoch = epoch
    sampler.set_epoch(epoch)
    loader.load_state_dict(state['loader'])
    generator.set_state(state['generator'])
    step = io.restore_checkpoint(student, optimizer, scheduler, folder, args)
    assert step == state['step']
    return step, epoch


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--phase', choices=['data', 'train'], default='train')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--schedule-steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=2000)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    assert args.steps <= args.schedule_steps and args.batch > 0 and args.accum > 0
    RUN = ROOT / 'runs' / args.run
    RUN.mkdir(parents=True, exist_ok=True)
    io.ROOT, io.RUN, io.ASSETS, io.REPO = ROOT, RUN, ASSETS, REPO
    io.identity = identity
    run_identity = identity()
    if (RUN / 'config.json').exists():
        assert args.resume is not None
        assert json.loads((RUN / 'config.json').read_text())['identity'] == run_identity
    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if args.phase == 'train':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        assert WORLD == 4
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    cli = make_recipe(REPO, ASSETS, ROOT, RUN, args.batch, args.accum, args.schedule_steps)
    config = LingbotVLAV2Config(**{**cli['model'], **cli['train']})
    config.skip_auxiliary_loss = False
    config.use_cache = False
    if RANK == 0:
        receipt = dict(cli=cli, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            identity=run_identity, base_revision=BASE_REVISION, upstream_revision=UPSTREAM_REVISION,
            initialization='official_base_only_no_lab_weights', norm_sha256=NORM_SHA256,
            lora={'targets': ['language.q_proj', 'language.v_proj'], 'r': LORA_RANK,
                  'alpha': LORA_ALPHA, 'dropout': 0.0, 'old_adapters_loaded': False},
            sampling='official StatefulDistributedSampler; indexed raw HDF5; shuffle; drop_last=True',
            augmentation='official paired color augmentation with explicit raw [0,255] boundary repair',
            precision={'base_load_dtype': 'float32', 'frozen_param_dtype': 'float32',
                'trainable_param_dtype': 'float32', 'student_compute': 'official module mixed precision',
                'teacher_autocast_dtype': 'bfloat16', 'training_checkpoint_dtype': 'float32'},
            declared_differences=['raw HDF5 adapter', 'relative per-horizon MeanStd / absolute gripper MinMax',
                'language q/v LoRA r256 alpha512', 'L2 flow matching retained',
                'FSDP full shard and gradient checkpointing', 'SDPA vision and no outer model compile for this runtime'])
        (RUN / 'config.json').write_text(json.dumps(receipt, indent=2))
        (RUN / 'train_source.py').write_bytes(Path(__file__).read_bytes())
        (RUN / 'lingbotvla_cli.yaml').write_text(yaml.safe_dump(cli, sort_keys=False))
    dataset = IndexedClean2500(config, cli)
    if args.phase == 'data':
        for index in (0, 1, 138, 139, 10000, len(dataset) - 1):
            batch = dataset[index]
            io.emit('data_roundtrip', metadata=json.loads(batch['_meta']),
                shapes={k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)})
        io.emit('data_passed', episodes=len(dataset.records), windows=len(dataset))
        return
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.optim import build_lr_scheduler
    init_parallel_state(dp_size=WORLD, dp_mode='fsdp2')
    student = build_student(config)
    student = build_parallelize_model(student, enable_full_shard=True,
        enable_mixed_precision=True, enable_fp32=True, enable_gradient_checkpointing=True,
        init_device='cuda', basic_modules=student._no_split_modules,
        fsdp_llm_blocks=False, vlm_fsdp=True, enable_reentrant=False)
    student.train()
    optimizer = make_optimizer(student, cli)
    from lingbotvla.models.vla.lingbot_vla.moe_load_balance import build_moe_load_balance_hook
    optimizer.register_step_pre_hook(build_moe_load_balance_hook(student,
        coeff=cli['train']['bias_update_speed'], bias_centering=False, update_interval=1))
    io.emit('moe_load_balance_hook', coefficient=0, interval=1)
    train = cli['train']
    scheduler = build_lr_scheduler(optimizer, train_steps=args.schedule_steps,
        lr=train['lr'], lr_min=train['lr_min'], lr_warmup_ratio=train['lr_warmup_ratio'],
        lr_start=train['lr_start'], lr_decay_style=train['lr_decay_style'],
        lr_decay_ratio=train['lr_decay_ratio'])
    loader, sampler, generator = build_loader(dataset, args)
    start_step, epoch = (restore_run(student, optimizer, scheduler, args,
        dataset, loader, sampler, generator) if args.resume else (0, 0))
    dataset.epoch = epoch
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    teachers, metrics, receipt = None, [], None
    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses, metadata = np.zeros(7), []
        for _ in range(args.accum):
            try:
                source = next(iterator)
            except StopIteration:
                epoch += 1
                dataset.epoch = epoch
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                source = next(iterator)
                io.emit('epoch_started', epoch=epoch, step=step)
            metadata.extend(json.loads(x) for x in source.pop('_meta'))
            batch = {k: v.cuda(non_blocking=True) for k, v in source.items()}
            teachers, targets = io.targets_for_batch(train['align_params'], batch, teachers)
            outputs = student(**batch, **targets)
            values = [*outputs[:6], outputs[6]['router_z_loss']]
            assert all(torch.isfinite(v).all() for v in values)
            accounting = loss_accounting(values)
            if step < 3:
                io.emit('loss_accounting', step=step + 1, **accounting)
            (outputs[0] / args.accum).backward()
            losses += np.array([float(v.detach()) for v in values]) / args.accum
            del outputs, targets, batch, source
        norms = gradient_audit(student)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), train['max_grad_norm'], foreach=True)
        grad_norm = float(grad_norm.full_tensor() if hasattr(grad_norm, 'full_tensor') else grad_norm)
        assert np.isfinite(grad_norm)
        sentinels = {}
        for name, parameter in student.named_parameters():
            cat = io.category(name)
            grad = io.local(parameter.grad) if parameter.grad is not None else None
            if parameter.requires_grad and cat not in sentinels and grad is not None and grad.numel() and grad.abs().max() > 0:
                index = grad.abs().reshape(-1).argmax()
                sentinels[cat] = (parameter, index, io.local(parameter.detach()).reshape(-1)[index].clone())
        assert len(sentinels) == 4
        optimizer.step()
        scheduler.step()
        updates = {cat: float((io.local(p.detach()).reshape(-1)[idx] - old).abs())
                   for cat, (p, idx, old) in sentinels.items()}
        assert all(np.isfinite(v) for v in updates.values())
        if step < 3:
            assert all(v > 0 for v in updates.values()), updates
        assert all(p.grad is None for teacher in teachers for p in teacher.parameters())
        torch.cuda.synchronize()
        averaged = torch.tensor(losses, device='cuda', dtype=torch.float64)
        dist.all_reduce(averaged)
        averaged /= WORLD
        row = dict(step=step + 1, epoch=epoch, seconds=time.monotonic() - started,
            losses=averaged.tolist(), loss_labels=['total', 'flow_matching', 'depth', 'future_depth',
                'video_current_and_future', 'sequence_wise', 'router_z'],
            learning_rates=[float(g['lr']) for g in optimizer.param_groups],
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            gradient_norms=norms, gradient_norm_before_clip=grad_norm,
            updates=updates, samples=metadata)
        io.emit('step', **row)
        metrics.append(row)
        metrics = metrics[-100:]
        if not args.no_save and (step + 1) % args.save_every == 0:
            optimizer.zero_grad(set_to_none=True)
            receipt = save_run(student, optimizer, scheduler, step + 1, args,
                               loader, generator, epoch)
    optimizer.zero_grad(set_to_none=True)
    if not args.no_save and args.steps % args.save_every:
        receipt = save_run(student, optimizer, scheduler, args.steps, args,
                           loader, generator, epoch)
    (RUN / f'result_rank{RANK}.json').write_text(json.dumps(dict(status='passed',
        start_step=start_step, final_step=args.steps, batch=args.batch,
        effective_batch=WORLD * args.batch * args.accum, metrics=metrics, checkpoint=receipt), indent=2))
    io.emit('complete', steps=args.steps)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
