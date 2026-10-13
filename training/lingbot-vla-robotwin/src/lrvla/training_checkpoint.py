"""Stream official weights and export merged, strict-loadable hf_ckpt folders."""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch
import torch.distributed.checkpoint as distributed_checkpoint
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from safetensors import safe_open
from safetensors.torch import save_file

from .training_models import LoRALinear


def run_relative_path(run, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(run) / path).resolve()


def load_safetensors_streamed(model, checkpoint, dtype=torch.bfloat16, device="cpu"):
    files = sorted(Path(checkpoint).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors shards in {checkpoint}; pass the nested hf_ckpt directory.")
    expected = model.state_dict()
    # Verify completeness and geometry from headers before allocating GPU weights.
    inventory = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name not in expected:
                    raise ValueError(f"Unexpected checkpoint key: {name}")
                if name in inventory:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                shape = tuple(reader.get_slice(name).get_shape())
                if shape != tuple(expected[name].shape):
                    raise ValueError(f"Shape mismatch for {name}: checkpoint {shape}, model {tuple(expected[name].shape)}")
                inventory.add(name)
    missing = set(expected) - inventory
    if missing:
        raise ValueError(f"Incomplete checkpoint: missing {len(missing)} tensors, e.g. {sorted(missing)[:8]}. Finish every shard download and preserve the base architecture.")
    seen = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            keys = reader.keys()
            unknown = set(keys) - set(expected)
            if unknown:
                raise ValueError(f"Unexpected checkpoint keys: {sorted(unknown)[:8]}")
            for name in keys:
                if name in seen:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                if tuple(reader.get_slice(name).get_shape()) != tuple(expected[name].shape):
                    raise ValueError(f"Shape mismatch for {name}: checkpoint {reader.get_slice(name).get_shape()}, model {expected[name].shape}")
                value = reader.get_tensor(name)
                if value.is_floating_point():
                    value = value.to(device=device, dtype=dtype)
                else:
                    value = value.to(device=device)
                # Assign one tensor at a time, avoiding a full-model host copy.
                parent_path, _, leaf = name.rpartition(".")
                parent = model.get_submodule(parent_path) if parent_path else model
                if leaf in parent._parameters:
                    old = parent._parameters[leaf]
                    parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=old.requires_grad)
                elif leaf in parent._buffers:
                    parent._buffers[leaf] = value
                else:
                    raise KeyError(f"Checkpoint tensor {name} has no corresponding module slot")
                seen.add(name)
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)[:12]}. Preserve the base checkpoint architecture/config.")
    if any(p.is_meta for p in model.parameters()):
        raise AssertionError("Unloaded meta parameters remain.")
    return {"shards": len(files), "tensors": len(seen)}


def merged_state_items(model, dtype=torch.bfloat16, state_dict=None):
    adapters = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    for name, tensor in (state_dict if state_dict is not None else model.state_dict()).items():
        adapter_name, _, suffix = name.rpartition(".")
        if adapter_name in adapters and suffix in {"lora_A", "lora_B"}:
            continue
        if ".base." in name:
            prefix, tail = name.rsplit(".base.", 1)
            if prefix in adapters:
                value = adapters[prefix].merged_weight() if tail == "weight" else tensor.detach().cpu()
                name = f"{prefix}.{tail}"
            else:
                value = tensor.detach().cpu()
        else:
            value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.to(dtype)
        yield name, value.contiguous()


def export_hf_checkpoint(model, output, *, dtype=torch.bfloat16, max_shard_bytes=512 * 2**20,
                         processor=None, state_dict=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix an export with existing files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    index, total, shard, shard_bytes, shards = {}, 0, {}, 0, []

    def flush():
        nonlocal shard, shard_bytes
        if not shard:
            return
        filename = f"model-{len(shards) + 1:05d}.safetensors"
        save_file(shard, str(output / filename), metadata={"format": "pt"})
        for name in shard:
            index[name] = filename
        shards.append(filename)
        shard, shard_bytes = {}, 0

    for name, value in merged_state_items(model, dtype, state_dict=state_dict):
        size = value.numel() * value.element_size()
        if shard and shard_bytes + size > max_shard_bytes:
            flush()
        shard[name] = value
        shard_bytes += size
        total += size
    flush()
    (output / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": index}, indent=2))
    if hasattr(model.config, "to_json_file"):
        model.config.to_json_file(str(output / "config.json"))
    if processor is not None:
        processor.save_pretrained(output)
    model_type = getattr(model.config, "model_type", "")
    export_format = "synthetic_smoke" if model_type == "synthetic_smoke" else "official_lingbot_vla_v2"
    adapter_count = sum(isinstance(module, LoRALinear) for module in model.modules())
    (output / "export.json").write_text(json.dumps({"format": export_format, "merged_lora": bool(adapter_count),
                                                   "lora_modules": adapter_count, "dtype": str(dtype),
                                                   "tensors": len(index), "shards": len(shards)}, indent=2))
    return output


def capture_rng_state(device=None):
    numpy_rng = np.random.get_state()
    cuda_rng = None
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state(device if device is not None else torch.cuda.current_device())
    return {"torch_rng": torch.get_rng_state(), "cuda_rng": cuda_rng,
            "python_rng": random.getstate(),
            "numpy_rng": (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:])}


def restore_rng_state(state, device=None):
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"], device if device is not None else torch.cuda.current_device())


def _fsdp_checkpoint_options():
    # The frozen BF16 base is identified by SHA and loaded from its original
    # shards on resume. Only trainable FP32 parameters and AdamW state belong
    # in a G2 training checkpoint.
    return StateDictOptions(cpu_offload=True, ignore_frozen_params=True)


def save_fsdp_checkpoint(model, optimizer, scheduler, output, state, metadata, context):
    """Collectively save a recoverable upstream FSDP2 training state."""
    output = Path(output)
    if context.is_main:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite FSDP checkpoint: {output}")
        output.mkdir(parents=True)
    context.barrier()
    model_state, optimizer_state = get_state_dict(model, optimizer, options=_fsdp_checkpoint_options())
    distributed_checkpoint.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=str(output / "distributed"),
    )
    rank_file = output / f"rank_{context.rank}_runtime.pt"
    temporary = rank_file.with_suffix(".tmp")
    torch.save({"scheduler": scheduler.state_dict(), "state": dict(state), "metadata": metadata,
                "rng": capture_rng_state(context.device)}, temporary)
    temporary.replace(rank_file)
    context.barrier()
    if context.is_main:
        (output / "COMPLETED.json").write_text(json.dumps({
            "global_step": state["global_step"], "world_size": context.world_size,
            "format": "lrvla_upstream_fsdp2_dcp_v1", "base_sha256": metadata["base_identity"]["sha256"],
        }, indent=2))
    context.barrier()


def prune_fsdp_checkpoints(checkpoints_root, current_step, keep_steps):
    """Keep milestones and newest completed state; never touch another run."""
    root = Path(checkpoints_root).resolve(strict=True)
    current_step = int(current_step)
    milestones = {int(step) for step in keep_steps}
    if current_step <= 0 or any(step <= 0 for step in milestones):
        raise ValueError("Checkpoint steps must be positive")
    current = root / f"global_step_{current_step}"
    current_marker = current / "COMPLETED.json"
    if not current_marker.is_file():
        raise ValueError(f"Cannot prune before current checkpoint is complete: {current}")
    identity = json.loads(current_marker.read_text())
    if identity.get("format") != "lrvla_upstream_fsdp2_dcp_v1" or identity.get("global_step") != current_step:
        raise ValueError("Current checkpoint marker has invalid identity")
    deletable = []
    for candidate in sorted(root.iterdir()):
        match = re.fullmatch(r"global_step_(\d+)", candidate.name)
        if not match:
            continue
        step = int(match.group(1))
        if step >= current_step or step in milestones:
            continue
        if candidate.is_symlink() or not candidate.is_dir() or candidate.parent.resolve() != root:
            raise ValueError(f"Unsafe checkpoint target: {candidate}")
        marker = candidate / "COMPLETED.json"
        if not marker.is_file():
            continue  # Leave interrupted saves for explicit inspection.
        metadata = json.loads(marker.read_text())
        if (metadata.get("format") != identity["format"] or metadata.get("global_step") != step
                or metadata.get("world_size") != identity.get("world_size")
                or metadata.get("base_sha256") != identity.get("base_sha256")):
            raise ValueError(f"Refusing to prune checkpoint with different identity: {candidate}")
        deletable.append(candidate)
    for candidate in deletable:
        shutil.rmtree(candidate)
    return [candidate.name for candidate in deletable]


def resume_fsdp_checkpoint(model, optimizer, scheduler, checkpoint, *, expected_metadata, context):
    """Collectively restore trainable shards, AdamW, scheduler, sampler and rank RNG."""
    checkpoint = Path(checkpoint)
    marker = checkpoint / "COMPLETED.json"
    if not marker.is_file():
        raise ValueError(f"Incomplete FSDP checkpoint: {checkpoint}")
    complete = json.loads(marker.read_text())
    if complete.get("format") != "lrvla_upstream_fsdp2_dcp_v1" or complete.get("world_size") != context.world_size:
        raise ValueError("FSDP checkpoint format or world size differs")
    runtime = torch.load(checkpoint / f"rank_{context.rank}_runtime.pt", map_location="cpu", weights_only=True)
    saved_metadata = runtime["metadata"]
    for key in ("base_identity", "manifest_content_sha256", "selection", "norm_sha256",
                "model_geometry", "qwen_config_sha256", "loss_normalization", "data_sampling", "distributed"):
        if saved_metadata.get(key) != expected_metadata.get(key):
            raise ValueError(f"FSDP resume mismatch for {key}")
    if complete["global_step"] != runtime["state"]["global_step"]:
        raise ValueError("FSDP checkpoint marker and runtime step differ")
    model_state, optimizer_state = get_state_dict(model, optimizer, options=_fsdp_checkpoint_options())
    payload = {"model": model_state, "optimizer": optimizer_state}
    distributed_checkpoint.load(payload, checkpoint_id=str(checkpoint / "distributed"))
    # Frozen base tensors are intentionally absent; their exact source shards
    # were validated against base_identity before model construction.
    load_options = StateDictOptions(cpu_offload=True, ignore_frozen_params=True, strict=False)
    set_state_dict(model, optimizer, model_state_dict=payload["model"],
                   optim_state_dict=payload["optimizer"], options=load_options)
    scheduler.load_state_dict(runtime["scheduler"])
    restore_rng_state(runtime["rng"], context.device)
    context.barrier()
    return runtime["state"]


def save_training_checkpoint(model, optimizer, scheduler, output, state, metadata, *, rank_rng_states=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    local_rng = capture_rng_state()
    payload = {"trainable_state": weights, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "state": state, "metadata": metadata, **local_rng}
    if rank_rng_states is not None:
        payload["rank_rng_states"] = rank_rng_states
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def resume_training_checkpoint(model, optimizer, scheduler, checkpoint, *, expected_metadata, rank=0, device=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # A run may change max_steps or checkpoint intervals; parameter/data identity must match.
    for key in ("base_identity", "manifest_content_sha256"):
        if key not in payload["metadata"] or key not in expected_metadata:
            raise ValueError(f"Checkpoint lacks content identity {key}; cannot verify the frozen base and clean data for portable resume.")
    for key in ("base_identity", "manifest_content_sha256", "selection", "norm_sha256", "model_geometry", "qwen_config_sha256"):
        if payload["metadata"].get(key) != expected_metadata.get(key):
            raise ValueError(f"Resume mismatch for {key}; use identical frozen base content, audited clean data and trainable selection.")
    for key in ("loss_normalization", "data_sampling"):
        if payload["metadata"].get(key) != expected_metadata.get(key):
            raise ValueError(f"Resume mismatch for {key}; use the same loss normalization and data sampling. "
                             "Checkpoints predating the tail/normalization fix cannot provide an exact resume.")
    saved_distributed = payload["metadata"].get("distributed")
    expected_distributed = expected_metadata.get("distributed")
    legacy_single_gpu = (saved_distributed is None and expected_distributed is not None and
                         expected_distributed.get("world_size") == 1)
    if saved_distributed != expected_distributed and not legacy_single_gpu:
        raise ValueError("Resume mismatch for distributed topology or global batch configuration.")
    parameters = dict(model.named_parameters())
    trainable = {name for name, p in parameters.items() if p.requires_grad}
    if set(payload["trainable_state"]) != trainable:
        raise ValueError("Resume trainable parameter keys differ from the current model.")
    with torch.no_grad():
        for name, value in payload["trainable_state"].items():
            parameters[name].copy_(value)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rank_states = payload.get("rank_rng_states")
    if rank_states is not None:
        if not 0 <= rank < len(rank_states):
            raise ValueError("Checkpoint does not contain RNG state for this distributed rank.")
        restore_rng_state(rank_states[rank], device)
    else:
        # Backward compatibility with verified single-GPU checkpoints.
        legacy_cuda = payload.get("cuda_rng")
        if isinstance(legacy_cuda, list):
            legacy_cuda = legacy_cuda[0] if legacy_cuda else None
        restore_rng_state({"torch_rng": payload["torch_rng"], "cuda_rng": legacy_cuda,
                           "python_rng": payload["python_rng"], "numpy_rng": payload["numpy_rng"]}, device)
    return payload["state"]
