"""Two-rank FSDP feasibility probe for G2; never trains or saves the 6B policy.

Run from the training repository root with ``torchrun --nproc_per_node=2``.
The toy mode checks distributed forward/backward/optimizer mechanics. The full
mode loads the real frozen base and checks full-expert selection and FSDP
initialization, but deliberately does not claim training/resume/export support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import yaml

from lrvla.training_distributed import DistributedContext
from lrvla.training_models import ACTION_ROOT, configure_trainable, memory_report
from lrvla.training_runtime import build_policy, enable_action_checkpointing
from lrvla.training import BCTrainingModule


def report(context: DistributedContext, event: str, **values) -> None:
    torch.cuda.synchronize(context.device)
    record = {
        "event": event,
        "rank": context.rank,
        "allocated_gib": round(torch.cuda.memory_allocated(context.device) / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(context.device) / 2**30, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(context.device) / 2**30, 3),
        **values,
    }
    print(json.dumps(record, sort_keys=True), flush=True)


def toy_probe(context: DistributedContext) -> None:
    torch.manual_seed(17)
    module = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    wrapped = FSDP(module, device_id=context.device, sharding_strategy=ShardingStrategy.FULL_SHARD)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    value = torch.randn(8, 64, device=context.device)
    loss = wrapped(value).square().mean()
    loss.backward()
    gradient_norm = float(wrapped.clip_grad_norm_(1.0))
    optimizer.step()
    if not torch.isfinite(loss) or not torch.isfinite(torch.tensor(gradient_norm)):
        raise FloatingPointError("Nonfinite toy FSDP result")
    report(context, "toy_step", loss=float(loss.detach()), gradient_norm=gradient_norm)


def _fixture_batch(fixture_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    fixtures = torch.load(fixture_path, map_location="cpu", weights_only=True)
    observation = fixtures[0]["observation"]
    noise = fixtures[0]["noise"]
    if noise.shape != (1, 50, 55):
        raise AssertionError(f"Unexpected audit noise shape: {tuple(noise.shape)}")
    keys = ("images", "img_masks", "lang_tokens", "lang_masks", "image_grid_thw", "state")
    batch = {key: observation[key].unsqueeze(0).to(device) for key in keys}
    # This is a non-optimizing plumbing test, not a supervised training sample.
    batch["actions"] = torch.zeros_like(noise, device=device)
    batch["joint_mask"] = observation["action_joint_mask"].to(device)[None, None, :].expand(1, 50, 55)
    return batch


def full_probe(
    context: DistributedContext,
    root: Path,
    config_file: Path,
    *,
    fixture_path: Path,
    backward: bool,
    optimizer_step: bool,
) -> None:
    config = yaml.safe_load(config_file.read_text())
    config["mode"] = "expert"
    config["expert_last_n_layers"] = -1
    config["train_projections"] = True
    architecture = yaml.safe_load((root / config["architecture_yaml"]).read_text())
    architecture["train"]["chunk_size"] = int(config["chunk_size"])
    policy = build_policy(
        root,
        architecture,
        root / config["base_checkpoint"],
        root / config["qwen_path"],
        device=context.device,
        load_weights=True,
    )
    selection = configure_trainable(policy, config)
    memory = memory_report(policy)
    expert = policy.get_submodule(ACTION_ROOT)
    if len(expert.layers) != 36 or selection["mode"] != "expert":
        raise AssertionError("Unexpected G2 selection or architecture")
    if any("qwenvl." in name for name in selection["trainable_names"]):
        raise AssertionError("VLM must remain frozen")
    report(
        context,
        "before_fsdp",
        expert_layers=len(expert.layers),
        trainable_tensors=len(selection["trainable_names"]),
        trainable_parameters=memory["trainable_parameters"],
        ddp_static_lower_bound_gib=round(memory["adamw_training_lower_bound_gib"], 3),
    )
    if backward:
        enable_action_checkpointing(policy)
    # FSDP1 flattens each managed handle into one dtype. The model intentionally
    # keeps frozen weights BF16 and trainables FP32, so only shard trainables.
    frozen_parameters = [p for p in policy.parameters() if not p.requires_grad]
    wrapped = FSDP(
        BCTrainingModule(policy),
        auto_wrap_policy=ModuleWrapPolicy({type(expert.layers[0])}),
        ignored_states=frozen_parameters,
        use_orig_params=True,
        device_id=context.device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
    )
    wrapped_layers = sum(isinstance(layer, FSDP) for layer in expert.layers)
    if wrapped_layers != 36:
        raise AssertionError(f"Expected 36 individually sharded expert layers, got {wrapped_layers}")
    if sum(p.numel() for p in wrapped.parameters() if p.requires_grad) <= 0:
        raise AssertionError("No local trainable shard")
    report(
        context,
        "after_fsdp",
        wrapped_expert_layers=wrapped_layers,
        local_trainable_elements=sum(p.numel() for p in wrapped.parameters() if p.requires_grad),
    )
    dist.barrier(device_ids=[context.local_rank])
    if backward:
        batch = _fixture_batch(fixture_path, context.device)
        wrapped.train()
        torch.cuda.reset_peak_memory_stats(context.device)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = wrapped(batch, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite G2 FSDP probe loss")
        gathered_loss = [torch.zeros_like(loss.detach()) for _ in range(context.world_size)]
        dist.all_gather(gathered_loss, loss.detach())
        if not torch.allclose(gathered_loss[0], gathered_loss[1], atol=1e-5, rtol=1e-5):
            raise AssertionError("Identical fixture/seed gave different losses across ranks")
        loss.backward()
        gradients = [p.grad for p in wrapped.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
            raise FloatingPointError("Missing or nonfinite G2 FSDP gradients")
        report(context, "full_backward_no_step", loss=float(loss.detach()), gradient_tensors=len(gradients))
        if optimizer_step:
            optimizer = torch.optim.AdamW((p for p in wrapped.parameters() if p.requires_grad),
                                          lr=1e-7, foreach=False)
            gradient_norm = float(wrapped.clip_grad_norm_(1.0))
            if not torch.isfinite(torch.tensor(gradient_norm)):
                raise FloatingPointError("Nonfinite global FSDP gradient norm")
            optimizer.step()
            report(context, "full_optimizer_step", gradient_norm=gradient_norm,
                   optimizer_state_tensors=sum(len(state) for state in optimizer.state.values()))
        dist.barrier(device_ids=[context.local_rank])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "full", "full-backward", "full-optimizer"), required=True)
    parser.add_argument("--config", default="configs/train_expert_ddp.yaml")
    parser.add_argument(
        "--fixture",
        default=("runs/e0_jiali_20261008_004327/"
                 "action-audit-20261008T145826Z-79ydnrm1/fixture.pt"),
        help="Existing audit fixture used only for the non-optimizing/synthetic-action probe",
    )
    args = parser.parse_args()
    context = DistributedContext.from_environment("cuda")
    if context.world_size != 2:
        raise ValueError("This G2 feasibility probe requires exactly two GPU ranks")
    try:
        if args.mode == "toy":
            toy_probe(context)
        else:
            root = Path(__file__).resolve().parents[1]
            full_probe(context, root, root / args.config, fixture_path=root / args.fixture,
                       backward=args.mode in {"full-backward", "full-optimizer"},
                       optimizer_step=args.mode == "full-optimizer")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
