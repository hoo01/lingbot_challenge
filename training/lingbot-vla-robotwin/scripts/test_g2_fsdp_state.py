#!/usr/bin/env python3
"""Two-rank toy FSDP save/resume equivalence test, without competition assets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from lrvla.training_checkpoint import resume_fsdp_checkpoint, save_fsdp_checkpoint
from lrvla.training_distributed import DistributedContext


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(16, 16).requires_grad_(False)
        self.trainable = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))

    def forward(self, value):
        return self.trainable(self.frozen(value))


def snapshot(model, context):
    with FSDP.summon_full_params(model, recurse=True, writeback=False,
                                 rank0_only=True, offload_to_cpu=True):
        if context.is_main:
            return {name: parameter.detach().cpu().clone()
                    for name, parameter in model.module.named_parameters() if parameter.requires_grad}
    return None


def step(model, optimizer, scheduler, device):
    optimizer.zero_grad(set_to_none=True)
    value = torch.arange(32 * 16, device=device, dtype=torch.float32).reshape(32, 16) / 512
    loss = model(value).square().mean()
    loss.backward()
    model.clip_grad_norm_(1.0)
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["save", "resume"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = DistributedContext.from_environment("cuda")
    if context.world_size != 2:
        raise ValueError("Toy test requires two ranks")
    try:
        torch.manual_seed(11)
        model = Toy().to(context.device)
        frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
        wrapped = FSDP(model, ignored_states=frozen, use_orig_params=True,
                       device_id=context.device, sharding_strategy=ShardingStrategy.FULL_SHARD)
        optimizer = torch.optim.AdamW((p for p in wrapped.parameters() if p.requires_grad),
                                      lr=0.01, foreach=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: 1 - index / 10)
        metadata = {"base_identity": {"sha256": "toy"}, "manifest_content_sha256": "toy",
                    "selection": {"mode": "toy"}, "norm_sha256": "toy", "model_geometry": {},
                    "qwen_config_sha256": "toy", "loss_normalization": "toy", "data_sampling": {},
                    "distributed": {"world_size": 2, "strategy": "fsdp_full_shard"}}
        checkpoint = args.output / "global_step_1"
        if args.phase == "save":
            loss1 = step(wrapped, optimizer, scheduler, context.device)
            save_fsdp_checkpoint(wrapped, optimizer, scheduler, checkpoint,
                                 {"global_step": 1, "epoch": 0, "batch_offset": 1}, metadata, context)
            first = snapshot(wrapped, context)
            loss2 = step(wrapped, optimizer, scheduler, context.device)
            second = snapshot(wrapped, context)
            if context.is_main:
                torch.save({"first": first, "second": second, "loss1": loss1, "loss2": loss2},
                           args.output / "expected.pt")
                print(f"toy_save_ok loss1={loss1:.9f} loss2={loss2:.9f}", flush=True)
        else:
            state = resume_fsdp_checkpoint(wrapped, optimizer, scheduler, checkpoint,
                                           expected_metadata=metadata, context=context)
            assert state == {"global_step": 1, "epoch": 0, "batch_offset": 1}
            first = snapshot(wrapped, context)
            loss2 = step(wrapped, optimizer, scheduler, context.device)
            second = snapshot(wrapped, context)
            if context.is_main:
                expected = torch.load(args.output / "expected.pt", map_location="cpu", weights_only=True)
                for name in first:
                    torch.testing.assert_close(first[name], expected["first"][name], rtol=0, atol=0)
                    torch.testing.assert_close(second[name], expected["second"][name], rtol=0, atol=0)
                assert loss2 == expected["loss2"]
                print(f"toy_resume_exact_ok loss2={loss2:.9f}", flush=True)
        context.barrier()
    finally:
        context.close()


if __name__ == "__main__":
    main()
