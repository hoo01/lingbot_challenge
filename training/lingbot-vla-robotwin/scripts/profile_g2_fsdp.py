"""Bounded real-clean-data GPU memory profiler for the G2 FSDP preflight.

This is deliberately not the resumable training entrypoint: no checkpoint,
export, SwanLab run, or formal training metrics are written. All small runtime
assets are staged beneath /data/tmp; base weights and datasets are read locally.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import yaml

from lrvla.data import load_manifest
from lrvla.training import BCTrainingModule, move_batch, prepare_runtime_files
from lrvla.training_data import build_clean_datasets, epoch_dataloader
from lrvla.training_distributed import DistributedContext
from lrvla.training_models import ACTION_ROOT, configure_trainable
from lrvla.training_runtime import build_policy, enable_action_checkpointing


def emit(context: DistributedContext, event: str, **fields) -> None:
    torch.cuda.synchronize(context.device)
    print(json.dumps({
        "event": event,
        "rank": context.rank,
        "allocated_gib": round(torch.cuda.memory_allocated(context.device) / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(context.device) / 2**30, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(context.device) / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(context.device) / 2**30, 3),
        **fields,
    }, sort_keys=True), flush=True)


def profile(context: DistributedContext, root: Path, args: argparse.Namespace) -> None:
    config_path = root / args.config
    config = yaml.safe_load(config_path.read_text())
    config.update(mode="expert", expert_last_n_layers=-1, train_projections=True)
    manifest_path = root / config["manifest"]
    norm_path = root / config["norm_stats"]
    manifest = load_manifest(manifest_path)
    architecture = yaml.safe_load((root / config["architecture_yaml"]).read_text())
    architecture["train"]["chunk_size"] = int(config["chunk_size"])
    prepared = None
    if context.is_main:
        prepared = prepare_runtime_files(
            config, architecture, manifest_path, norm_path,
            Path(args.scratch) / f"batch_{args.batch_size}_accum_{args.accumulation}",
        )
    data_config, robot_config_name, _ = context.broadcast_object(prepared)
    dist.barrier(device_ids=[context.local_rank])

    policy = build_policy(
        root, architecture, root / config["base_checkpoint"],
        root / config["qwen_path"], device=context.device, load_weights=True,
    )
    selection = configure_trainable(policy, config)
    expert = policy.get_submodule(ACTION_ROOT)
    if len(expert.layers) != 36 or len(selection["trainable_names"]) != 875:
        raise AssertionError("Unexpected G2 Expert selection")
    enable_action_checkpointing(policy)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        root / config["qwen_path"], padding_side="right", local_files_only=True,
    )
    dataset, _, splits = build_clean_datasets(
        manifest, data_config, policy.config, processor, Path(robot_config_name),
        int(config["val_episodes"]), action_chunk_tail=config["action_chunk_tail"],
    )
    if len(splits) != 50 or any(len(value["train"]) != 45 for value in splits.values()):
        raise AssertionError("Expected the E0 clean 50-task train split")
    loader = iter(epoch_dataloader(
        dataset, args.batch_size, int(config["seed"]), num_workers=args.num_workers,
        rank=context.rank, world_size=context.world_size,
    ))
    frozen = [parameter for parameter in policy.parameters() if not parameter.requires_grad]
    wrapped = FSDP(
        BCTrainingModule(policy),
        auto_wrap_policy=ModuleWrapPolicy({type(expert.layers[0])}),
        ignored_states=frozen,
        use_orig_params=True,
        device_id=context.device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
    )
    if sum(isinstance(layer, FSDP) for layer in expert.layers) != 36:
        raise AssertionError("Not all Expert layers were individually sharded")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in wrapped.parameters() if parameter.requires_grad),
        lr=float(config["lr"]), weight_decay=float(config["weight_decay"]), foreach=False,
    )
    wrapped.train()
    emit(context, "ready", batch_size=args.batch_size, accumulation=args.accumulation,
         global_batch=args.batch_size * args.accumulation * context.world_size,
         dataset_samples=len(dataset), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
         norm_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
         num_workers=args.num_workers, expert_layers=36)
    dist.barrier(device_ids=[context.local_rank])
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(context.device)
        start = time.monotonic()
        load_seconds = 0.0
        local_valid = 0.0
        local_loss = 0.0
        for micro in range(args.accumulation):
            loaded = time.monotonic()
            batch = next(loader)
            load_seconds += time.monotonic() - loaded
            if batch["actions"].shape != (args.batch_size, 50, 55):
                raise AssertionError(f"Unexpected clean action shape: {tuple(batch['actions'].shape)}")
            local_valid += float(batch["joint_mask"].float().sum())
            sync = wrapped.no_sync() if micro < args.accumulation - 1 else nullcontext()
            with sync:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = wrapped(move_batch(batch, context.device), reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite G2 real-data loss")
                loss.backward()
            local_loss += float(loss.detach())
            del batch, loss
        global_valid = context.sum_scalar(local_valid)
        global_loss = context.sum_scalar(local_loss)
        if global_valid <= 0:
            raise AssertionError("No valid clean action coordinates")
        # Probe only: this scaling mirrors DDP's rank averaging. Full FSDP
        # gradient/loss parity still requires a separate trainer audit.
        for parameter in wrapped.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                parameter.grad.div_(global_valid / context.world_size)
        grad_norm = float(wrapped.clip_grad_norm_(float(config["max_grad_norm"])))
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError("Nonfinite G2 gradient norm")
        optimizer.step()
        torch.cuda.synchronize(context.device)
        emit(context, "profile_step", step=step, batch_size=args.batch_size,
             accumulation=args.accumulation, global_batch=args.batch_size * args.accumulation * context.world_size,
             num_workers=args.num_workers,
             loss_mean=global_loss / global_valid, gradient_norm=grad_norm,
             step_seconds=round(time.monotonic() - start, 3),
             data_load_seconds=round(load_seconds, 3))
        dist.barrier(device_ids=[context.local_rank])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_expert_ddp.yaml")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scratch", default="/data/tmp/g2_fsdp_profile_20261012")
    args = parser.parse_args()
    if (not 1 <= args.batch_size <= 32 or not 1 <= args.accumulation <= 4
            or not 1 <= args.steps <= 8 or not 0 <= args.num_workers <= 8):
        parser.error("Preflight bounds: batch 1–32, accumulation 1–4, steps 1–8, workers 0–8")
    context = DistributedContext.from_environment("cuda")
    if context.world_size != 2:
        raise ValueError("This G2 profiler requires exactly two ranks")
    try:
        profile(context, Path(__file__).resolve().parents[1], args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
