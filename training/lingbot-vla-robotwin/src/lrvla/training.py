"""Single-node, clean-only supervised post-training of LingBot-VLA v2."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import yaml

from .training_checkpoint import (capture_rng_state, export_hf_checkpoint,
                                  prune_fsdp_checkpoints, resume_fsdp_checkpoint, resume_training_checkpoint,
                                  save_fsdp_checkpoint, save_training_checkpoint)
from .training_data import build_clean_datasets, epoch_dataloader, model_collate
from .training_distributed import DistributedContext, distributed_metadata
from .training_models import configure_trainable, flow_matching_bc, memory_report, normalize_bc_gradients
from .training_identity import base_checkpoint_identity, manifest_content_sha256, repository_identity
from .training_runtime import add_vendor_path, build_policy, enable_action_checkpointing
from .training_schedule import schedule_multiplier


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BCTrainingModule(nn.Module):
    """Give DDP a real forward boundary while keeping export/checkpoint names stable."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    @property
    def model(self):
        """Expose the official VLA path expected by upstream FSDP2 grouping."""
        return self.policy.model

    def forward(self, batch, reduction="sum"):
        if reduction == "sample":
            return self.policy.sample_actions(
                batch["images"], batch["img_masks"], batch["lang_tokens"],
                batch["lang_masks"], batch["state"], noise=batch["noise"],
                image_grid_thw=batch["image_grid_thw"])
        return flow_matching_bc(self.policy, batch, reduction=reduction)


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def validate(model, loader, device, max_batches=8, loss_fn=flow_matching_bc):
    model.eval()
    numerator, valid_count = 0.0, 0.0
    gpu_ids = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=gpu_ids), torch.no_grad():
        torch.manual_seed(12345)
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                enabled=torch.device(device).type == "cuda"):
                loss = loss_fn(model, move_batch(batch, device))
            count = float(batch["joint_mask"].float().sum())
            numerator += float(loss) * count
            valid_count += count
    model.train()
    return numerator / valid_count if valid_count else None


def prepare_runtime_files(config, architecture, manifest_path, norm_path, output):
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    provenance_path = norm_path.with_suffix(".provenance.json")
    if not provenance_path.is_file():
        raise ValueError("Norm statistics need a clean-data provenance sidecar; run scripts/prepare_data.py norm.")
    provenance = json.loads(provenance_path.read_text())
    content_digest = manifest_content_sha256(json.loads(manifest_path.read_text()))
    matches_manifest = (provenance.get("manifest_sha256") == manifest_digest or
                        provenance.get("manifest_content_sha256") == content_digest)
    if provenance.get("clean_only") is not True or not matches_manifest:
        raise ValueError("Norm provenance does not match this clean-only manifest.")
    if provenance.get("norm_stats_sha256") != hashlib.sha256(norm_path.read_bytes()).hexdigest():
        raise ValueError("Norm statistics changed after clean-only provenance was created.")
    holdout = int(config.get("val_episodes", 5))
    if provenance.get("val_episode_ids") != list(range(50 - holdout, 50)):
        raise ValueError("Norm train/validation episode split differs from the training config.")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    saved_norm = assets / "norm_stats.json"
    if norm_path.resolve() != saved_norm.resolve():
        shutil.copy2(norm_path, saved_norm)
        shutil.copy2(provenance_path, saved_norm.with_suffix(".provenance.json"))
    saved_manifest = assets / "clean_manifest.json"
    if manifest_path.resolve() != saved_manifest.resolve():
        shutil.copy2(manifest_path, saved_manifest)
    data = dict(architecture["data"])
    data.update(img_size=int(config.get("img_size", 256)), norm_stats_file=str(saved_norm),
                image_augment=bool(config.get("image_augment", False)), video_backend="pyav", use_future_image=False,
                prompt_type="global", data_name="multi")
    for key in ("joints", "norm_type"):
        data[key] = [repr(item) if isinstance(item, dict) else item for item in data[key]]
    robot_dir = output / "configs" / "robot_configs"
    robot_dir.mkdir(parents=True, exist_ok=True)
    vendor = PROJECT_ROOT / "vendor" / "lingbot-vla-v2"
    robot_config = yaml.safe_load((vendor / "configs" / "robot_configs" / "robotwin.yaml").read_text())
    robot_config["norm_stats"] = "assets/norm_stats.json"
    robot_path = robot_dir / "robotwin.yaml"
    robot_path.write_text(yaml.safe_dump(robot_config, sort_keys=False))
    data["robot_config_root"] = str(robot_dir)
    data["train_path"] = str(saved_manifest)
    return data, robot_path, manifest_digest


def portable_inference_cli(cli):
    """Normalize deployment paths against the run directory, independent of cwd."""
    from copy import deepcopy
    result = deepcopy(cli)
    result["model"]["tokenizer_path"] = "assets/qwen3_vl"
    result["data"].update(norm_stats_file="assets/norm_stats.json", robot_config_root="configs/robot_configs",
                           train_path="assets/clean_manifest.json")
    return result


def _train(config, context, *, dry_run=False):
    training_started = time.monotonic()
    strategy = config.get("distributed_strategy", "ddp")
    if strategy not in {"ddp", "upstream_fsdp2"}:
        raise ValueError("distributed_strategy must be ddp or upstream_fsdp2.")
    if strategy == "upstream_fsdp2":
        if context.world_size != 2:
            raise ValueError("G2 FSDP currently requires exactly two CUDA ranks.")
        if config.get("export", True) and config.get("export_dtype") != "float32":
            raise ValueError("G2 full HF export must use float32, matching E1 delivery.")
    action_chunk_tail = config.get("action_chunk_tail", "mask")
    if action_chunk_tail not in {"mask", "drop"}:
        raise ValueError("action_chunk_tail must be 'mask' or 'drop'.")
    for key, default in (("batch_size", 1), ("gradient_accumulation", 16), ("max_steps", 1000),
                         ("save_every", 100), ("eval_every", 100)):
        if int(config.get(key, default)) <= 0:
            raise ValueError(f"{key} must be positive.")
    if not 0 <= int(config.get("val_episodes", 5)) < 50:
        raise ValueError("val_episodes must be between 0 and 49.")
    torch.manual_seed(int(config.get("seed", 42)))
    random.seed(int(config.get("seed", 42)))
    import numpy as np
    np.random.seed(int(config.get("seed", 42)))
    device = context.device
    if device.type != "cuda" and not dry_run:
        raise ValueError("The full 6B model training entrypoint needs CUDA; use --smoke for CPU validation.")
    base, qwen = resolve_path(config["base_checkpoint"]), resolve_path(config["qwen_path"])
    if "robotwin" in base.name.lower():
        raise ValueError("Use robbyant/lingbot-vla-v2-6b base weights; official RoboTwin post-trained initialization is forbidden.")
    if not (qwen / "config.json").is_file():
        raise FileNotFoundError(f"Qwen3-VL processor/config assets missing: {qwen}")
    architecture = yaml.safe_load(resolve_path(config["architecture_yaml"]).read_text())
    architecture["train"]["chunk_size"] = int(config.get("chunk_size", 50))
    if not dry_run:
        manifest_path, norm_path = resolve_path(config["manifest"]), resolve_path(config["norm_stats"])
        from .data import load_manifest
        manifest = load_manifest(manifest_path)
        if not manifest.get("complete_50_tasks") and not config.get("allow_subset", False):
            raise ValueError("Competition co-training requires all 50 clean tasks. Use --allow-subset only to debug the pipeline.")
        output = resolve_path(config["output_dir"])
        prepared = None
        if context.is_main:
            data_config, robot_config, manifest_digest = prepare_runtime_files(
                config, architecture, manifest_path, norm_path, output)
            print("Verifying frozen base shard SHA-256 identity before model loading...", flush=True)
            prepared = (data_config, str(robot_config), manifest_digest, base_checkpoint_identity(base))
        data_config, robot_config_value, manifest_digest, base_identity = context.broadcast_object(prepared)
        robot_config = Path(robot_config_value)
        context.barrier()
    policy = build_policy(PROJECT_ROOT, architecture, base, qwen, device=device, load_weights=not dry_run)
    selection = configure_trainable(policy, config)
    report = memory_report(policy)
    concise_selection = {k: v for k, v in selection.items() if k not in {"trainable_names", "lora_modules"}}
    concise_selection["lora_module_count"] = len(selection["lora_modules"])
    if context.is_main:
        print(json.dumps({"selection": concise_selection,
                          "memory": report, "dry_run": dry_run,
                          "distributed": {"world_size": context.world_size}}, ensure_ascii=False), flush=True)
    if dry_run:
        return report
    total_gpu_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
    if strategy == "ddp" and report["adamw_training_lower_bound_gib"] > total_gpu_gib * 0.92:
        raise RuntimeError(f"Weight/gradient/AdamW lower bound {report['adamw_training_lower_bound_gib']:.2f} GiB exceeds safe space on {total_gpu_gib:.2f} GiB GPU. Use LoRA or fewer expert_last_n_layers, or move to a larger server GPU.")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(qwen, padding_side="right", local_files_only=True)
    datasets, validation, splits = build_clean_datasets(manifest, data_config, policy.config, processor,
                                                       robot_config, int(config.get("val_episodes", 5)),
                                                       action_chunk_tail=action_chunk_tail)
    if context.is_main:
        print(json.dumps({"training_samples": len(datasets),
                          "validation_samples": len(validation) if validation is not None else 0,
                          "action_chunk_tail": action_chunk_tail}, ensure_ascii=False), flush=True)
    if config.get("gradient_checkpointing", True):
        enable_action_checkpointing(policy)
    train_values = {**architecture["train"], **policy.config.to_dict()}
    train_values.update(output_dir=str(output), train_expert_only=True, freeze_vision_encoder=True,
                        attention_implementation="eager", vit_attn_implementation="sdpa", use_compile=False)
    cli = {"model": {**architecture["model"], "model_path": str(base), "tokenizer_path": str(qwen)},
           "train": train_values, "data": data_config}
    # Qwen processor/config assets are small and contain no duplicate VLM weights.
    qwen_assets = output / "assets" / "qwen3_vl"
    if context.is_main:
        processor.save_pretrained(qwen_assets)
        shutil.copy2(qwen / "config.json", qwen_assets / "config.json")
        saved_cli = portable_inference_cli(cli)
        saved_cli["train"]["tokenizer_path"] = "assets/qwen3_vl"
        (output / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(saved_cli, sort_keys=False))
        (output / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (output / "episode_splits.json").write_text(json.dumps(splits, indent=2))
    context.barrier()
    batch_size, accumulation = int(config.get("batch_size", 1)), int(config.get("gradient_accumulation", 16))
    topology = distributed_metadata(context, batch_size, accumulation)
    if strategy == "upstream_fsdp2":
        topology["strategy"] = strategy
    metadata = {"base_checkpoint": str(base), "base_identity": base_identity,
                "manifest_sha256": manifest_digest, "manifest_content_sha256": manifest_content_sha256(manifest),
                "selection": selection,
                "training_method": "clean_only_behavior_cloning", "competition_all_50_tasks": bool(manifest.get("complete_50_tasks")),
                "loss_normalization": "valid_action_coordinate_mean_v1",
                "data_sampling": {"action_chunk_tail": action_chunk_tail, "validation_action_chunk_tail": "mask",
                                  "val_episodes": int(config.get("val_episodes", 5)),
                                  "seed": int(config.get("seed", 42)), "batch_size": int(config.get("batch_size", 1)),
                                  "gradient_accumulation": int(config.get("gradient_accumulation", 16))},
                "distributed": topology,
                "source": repository_identity(PROJECT_ROOT),
                "memory": report, "norm_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                "qwen_config_sha256": hashlib.sha256((qwen / "config.json").read_bytes()).hexdigest(),
                "model_geometry": {"chunk_size": policy.config.chunk_size, "max_action_dim": policy.config.max_action_dim,
                                   "max_state_dim": policy.config.max_state_dim, "img_size": data_config["img_size"]}}
    if context.is_main:
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    training_model = BCTrainingModule(policy)
    if strategy == "upstream_fsdp2":
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy
        from lingbotvla.distributed.parallel_state import init_parallel_state
        from lingbotvla.distributed.torch_parallelize import build_parallelize_model
        from .training_models import ACTION_ROOT

        init_parallel_state(dp_size=context.world_size, dp_mode="fsdp2", device_type="cuda")
        expert = policy.get_submodule(ACTION_ROOT)
        frozen = {parameter for parameter in policy.parameters() if not parameter.requires_grad}
        # Upstream's enable_mixed_precision=True calls model.float() before sharding,
        # which would upcast the frozen BF16 VLM and violate the E0 precision rule.
        # Keep BF16 frozen/FP32 trainable masters and use its FSDP2 BF16 forward.
        training_model = build_parallelize_model(
            training_model,
            enable_full_shard=True,
            enable_mixed_precision=False,
            enable_fp32=False,
            enable_gradient_checkpointing=False,
            basic_modules=[type(expert.layers[0]).__name__],
            fsdp_llm_blocks=False,
            vlm_fsdp=True,
            init_device="cuda",
            fsdp_kwargs={
                "ignored_params": frozen,
                "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                   reduce_dtype=torch.float32,
                                                   output_dtype=torch.bfloat16),
            },
        )
        wrapped_layers = sum(hasattr(layer, "set_requires_gradient_sync") for layer in expert.layers)
        if wrapped_layers != len(expert.layers):
            raise AssertionError(f"Only {wrapped_layers}/{len(expert.layers)} Expert layers were FSDP-wrapped")
    elif context.distributed:
        training_model = DistributedDataParallel(
            training_model, device_ids=[context.local_rank], output_device=context.local_rank,
            broadcast_buffers=False, find_unused_parameters=False)
    params = [p for p in training_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config.get("lr", 1e-4)),
                                  weight_decay=float(config.get("weight_decay", 0.01)), foreach=False)
    max_steps = int(config.get("max_steps", 1000))
    stop_after_step = int(config.get("stop_after_step", max_steps))
    if not 1 <= stop_after_step <= max_steps:
        raise ValueError("stop_after_step must be between 1 and max_steps")
    warmup = min(int(config.get("warmup_steps", 100)), max_steps)
    schedule_mode = config.get("lr_schedule", "cosine")
    schedule_constant = config.get("lr_constant_multiplier")
    schedule_minimum = float(config.get("lr_minimum_multiplier", 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: schedule_multiplier(
            step,
            max_steps,
            warmup,
            minimum=schedule_minimum,
            mode=schedule_mode,
            constant=schedule_constant,
        ),
    )
    state = {"global_step": 0, "epoch": 0, "batch_offset": 0}
    if config.get("resume"):
        if strategy == "upstream_fsdp2":
            state = resume_fsdp_checkpoint(
                training_model, optimizer, scheduler, resolve_path(config["resume"]),
                expected_metadata=metadata, context=context)
        else:
            state = resume_training_checkpoint(policy, optimizer, scheduler, resolve_path(config["resume"]),
                                               expected_metadata=metadata, rank=context.rank, device=device)
        expected_resume_lr = config.get("expected_resume_lr")
        if expected_resume_lr is not None:
            actual_lrs = [group["lr"] for group in optimizer.param_groups]
            if any(not math.isclose(lr, float(expected_resume_lr), rel_tol=1e-9, abs_tol=1e-12)
                   for lr in actual_lrs):
                raise ValueError(
                    f"Resume learning rate {actual_lrs} differs from expected {expected_resume_lr}"
                )
    else:
        rank_seed = int(config.get("seed", 42)) + context.rank * 100003
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)
        random.seed(rank_seed)
        np.random.seed(rank_seed % 2**32)

    swanlab_module = None
    swanlab_run = None
    swanlab_config = config.get("swanlab", {})
    if context.is_main and swanlab_config.get("enabled", False):
        import swanlab as swanlab_module

        run_id_path = output / "swanlab_run_id.txt"
        run_id = swanlab_config.get("run_id")
        if config.get("resume") and not run_id and run_id_path.is_file():
            run_id = run_id_path.read_text().strip()

        tracked_config = json.loads(json.dumps(config, default=str))
        tracked_config.update({
            "manifest_sha256": manifest_digest,
            "manifest_content_sha256": manifest_content_sha256(manifest),
            "norm_sha256": metadata["norm_sha256"],
            "source_commit": metadata["source"]["commit"],
            "source_dirty": metadata["source"]["dirty"],
            "world_size": topology["world_size"],
            "global_batch_size": topology["global_batch_size"],
            "trainable_parameters": report["trainable_parameters"],
            "total_parameters": report["total_parameters"],
        })
        init_kwargs = {
            "project": swanlab_config.get("project", "lingbot_cha"),
            "workspace": swanlab_config.get("workspace", "Elowynnn"),
            "experiment_name": swanlab_config.get("experiment_name") or output.name,
            "description": swanlab_config.get("description", "LingBot-VLA clean-only post-training"),
            "public": bool(swanlab_config.get("public", False)),
            "config": tracked_config,
        }
        if run_id:
            init_kwargs.update(id=run_id, resume="must")
        swanlab_run = swanlab_module.init(**init_kwargs)
        run_id_path.write_text(swanlab_run.id + "\n")
        print(json.dumps({"swanlab_run_id": swanlab_run.id}), flush=True)

    if max_steps <= 0 or batch_size <= 0 or accumulation <= 0:
        raise ValueError("max_steps, batch_size and gradient_accumulation must be positive.")
    workers = int(config.get("num_workers", 0))
    if context.is_main and workers > 0 and config.get("resume") and config.get("image_augment", False):
        print("Resume preserves consumed batches and main RNG; worker augmentation RNG replay is not guaranteed.", flush=True)

    def iterator():
        return iter(epoch_dataloader(datasets, batch_size, int(config.get("seed", 42)),
                                    state["epoch"], state["batch_offset"], workers,
                                    rank=context.rank, world_size=context.world_size))

    batches = iterator()
    validation_loader = (DataLoader(validation, batch_size=batch_size, collate_fn=model_collate, num_workers=0)
                         if validation and (context.is_main or strategy == "upstream_fsdp2") else None)
    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output / "metrics.jsonl"
    initial_global_step = state["global_step"]
    setup_seconds = context.max_scalar(time.monotonic() - training_started)
    while state["global_step"] < stop_after_step:
        start = time.monotonic()
        loss_numerator, valid_count, data_load_seconds = 0.0, 0.0, 0.0
        for micro_step in range(accumulation):
            load_start = time.monotonic()
            try:
                batch = next(batches)
            except StopIteration:
                state["epoch"] += 1
                state["batch_offset"] = 0
                batches = iterator()
                batch = next(batches)
            data_load_seconds += time.monotonic() - load_start
            state["batch_offset"] += 1
            if strategy == "upstream_fsdp2":
                training_model.set_requires_gradient_sync(micro_step == accumulation - 1)
            synchronization = (training_model.no_sync()
                               if strategy == "ddp" and context.distributed and micro_step < accumulation - 1
                               else nullcontext())
            with synchronization:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = training_model(move_batch(batch, device), reduction="sum")
                if not context.all_true(bool(torch.isfinite(loss).item())):
                    raise FloatingPointError(
                        f"Nonfinite BC loss on at least one rank at step {state['global_step']}")
                loss.backward()
            loss_numerator += float(loss.detach())
            valid_count += float(batch["joint_mask"].float().sum())
        global_loss_numerator = context.sum_scalar(loss_numerator)
        global_valid_count = context.sum_scalar(valid_count)
        # DDP averages gradients across ranks; divide by global_count/world_size
        # to recover the exact global summed-loss / global-coordinate gradient.
        normalize_bc_gradients(params, global_valid_count / context.world_size)
        grad = (torch.nn.utils.clip_grad_norm_(params, float(config.get("max_grad_norm", 1)), foreach=True)
                if strategy == "upstream_fsdp2" else
                torch.nn.utils.clip_grad_norm_(params, float(config.get("max_grad_norm", 1))))
        if strategy == "upstream_fsdp2":
            grad = grad.full_tensor()
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite gradient norm; optimizer step aborted.")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["global_step"] += 1
        step = state["global_step"]
        torch.cuda.synchronize(device)
        elapsed = context.max_scalar(time.monotonic() - start)
        peak_gpu = context.max_scalar(torch.cuda.max_memory_allocated(device) / 2**30)
        record = {"step": step, "loss": global_loss_numerator / global_valid_count, "grad_norm": float(grad),
                  "valid_action_coordinates": global_valid_count,
                  "lr": optimizer.param_groups[0]["lr"], "step_seconds": elapsed,
                  "peak_gpu_gib": peak_gpu, "world_size": context.world_size,
                  "global_batch_size": topology["global_batch_size"]}
        if step == initial_global_step + 1:
            record["setup_seconds"] = setup_seconds
        if strategy == "upstream_fsdp2":
            record["distributed_strategy"] = strategy
            record["num_workers_per_rank"] = workers
            record["data_load_seconds"] = context.max_scalar(data_load_seconds)
            record["peak_gpu_reserved_gib"] = context.max_scalar(
                torch.cuda.max_memory_reserved(device) / 2**30)
        if validation and step % int(config.get("eval_every", 100)) == 0:
            validation_start = time.monotonic()
            context.barrier()
            if strategy == "upstream_fsdp2":
                validation_loss = validate(
                    training_model, validation_loader, device, int(config.get("eval_batches", 8)),
                    loss_fn=lambda model, batch: model(batch, reduction="mean"))
                if context.is_main:
                    record["validation_bc_loss"] = validation_loss
            elif context.is_main:
                record["validation_bc_loss"] = validate(
                    policy, validation_loader, device, int(config.get("eval_batches", 8)))
            context.barrier()
            record["validation_seconds"] = context.max_scalar(time.monotonic() - validation_start)
        if config.get("checkpoint", True) and (step % int(config.get("save_every", 100)) == 0 or step == max_steps):
            checkpoint_start = time.monotonic()
            checkpoint_dir = output / "checkpoints" / f"global_step_{step}"
            if strategy == "upstream_fsdp2":
                save_fsdp_checkpoint(training_model, optimizer, scheduler, checkpoint_dir,
                                     dict(state), metadata, context)
                if context.is_main:
                    (output / "latest_checkpoint.txt").write_text(str(checkpoint_dir) + "\n")
                    keep_steps = config.get("checkpoint_keep_steps")
                    if keep_steps is not None:
                        removed = prune_fsdp_checkpoints(output / "checkpoints", step, keep_steps)
                        print(json.dumps({"checkpoint_retention_step": step,
                                          "kept_milestones": keep_steps,
                                          "removed_checkpoints": removed}), flush=True)
            else:
                rank_rng_states = context.gather_objects(capture_rng_state(device))
                if context.is_main:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_training_checkpoint(policy, optimizer, scheduler, checkpoint_dir / "training.pt",
                                             dict(state), metadata, rank_rng_states=rank_rng_states)
                    (output / "latest_checkpoint.txt").write_text(str(checkpoint_dir / "training.pt") + "\n")
            context.barrier()
            record["checkpoint_seconds"] = context.max_scalar(time.monotonic() - checkpoint_start)
        if context.is_main:
            print(json.dumps(record), flush=True)
            with metrics_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
        if context.is_main and swanlab_run is not None:
            swan_metrics = {
                "train/loss": record["loss"],
                "train/grad_norm": record["grad_norm"],
                "train/learning_rate": record["lr"],
                "train/valid_action_coordinates": record["valid_action_coordinates"],
                "performance/step_seconds": record["step_seconds"],
                "performance/peak_gpu_gib": record["peak_gpu_gib"],
            }
            if "validation_bc_loss" in record:
                swan_metrics["validation/bc_loss"] = record["validation_bc_loss"]
                swan_metrics["performance/validation_seconds"] = record["validation_seconds"]
            if "checkpoint_seconds" in record:
                swan_metrics["performance/checkpoint_seconds"] = record["checkpoint_seconds"]
            swanlab_module.log(swan_metrics, step=step)
    if config.get("export", True) and state["global_step"] == max_steps:
        checkpoint_dir = output / "checkpoints" / f"global_step_{state['global_step']}"
        if strategy == "upstream_fsdp2":
            if config.get("action_audit", False):
                from .training_audit import save_action_reference
                save_action_reference(
                    training_model, processor, data_config, robot_config,
                    validation_loader, output / "action_audit_reference.pt", context)
            torch.cuda.empty_cache()
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            full_state = get_model_state_dict(
                training_model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            if context.is_main:
                if not full_state or any(not name.startswith("policy.") for name in full_state):
                    raise RuntimeError("Upstream FSDP2 full state has missing or unexpected policy keys")
                policy_state = {name.removeprefix("policy."): value for name, value in full_state.items()}
                exported = export_hf_checkpoint(policy, checkpoint_dir / "hf_ckpt_fp32",
                                                dtype=torch.float32, processor=processor,
                                                state_dict=policy_state)
                print(json.dumps({"exported_hf_ckpt": str(exported),
                                  "official_cli": str(output / "lingbotvla_cli.yaml")}), flush=True)
        elif context.is_main:
            exported = export_hf_checkpoint(policy, checkpoint_dir / "hf_ckpt",
                                            dtype=getattr(torch, config.get("export_dtype", "bfloat16")), processor=processor)
            print(json.dumps({"exported_hf_ckpt": str(exported), "official_cli": str(output / "lingbotvla_cli.yaml")}), flush=True)
        context.barrier()
    if context.is_main and swanlab_run is not None:
        swanlab_module.finish()
    return state


def train(config, *, dry_run=False):
    context = DistributedContext.from_environment(config.get("device", "cuda"), dry_run=dry_run)
    try:
        return _train(config, context, dry_run=dry_run)
    finally:
        context.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--qwen-path")
    parser.add_argument("--manifest")
    parser.add_argument("--norm-stats")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--expert-last-n-layers", type=int)
    parser.add_argument("--val-episodes", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true", help="Construct on meta and report parameter memory; no 6B weights loaded.")
    parser.add_argument("--allow-subset", action="store_true", help="Debug with fewer than 50 tasks; resulting run is marked as a subset.")
    parser.add_argument("--smoke", action="store_true", help="Train/export/reload a tiny synthetic model, without using competition assets.")
    parser.add_argument("--no-export", action="store_true", help="Skip the full HF export; useful for short integration tests.")
    parser.add_argument("--local-rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke:
        from .training_smoke import run_smoke
        return run_smoke(resolve_path(args.output_dir or "artifacts/training_smoke"), args.device or "cpu", args.max_steps or 30)
    config = yaml.safe_load(resolve_path(args.config).read_text())
    for key, value in vars(args).items():
        if key not in {"config", "dry_run", "smoke", "allow_subset", "no_export", "local_rank"} and value is not None:
            config[key] = value
    if args.allow_subset:
        config["allow_subset"] = True
    if args.no_export:
        config["export"] = False
    return train(config, dry_run=args.dry_run)
