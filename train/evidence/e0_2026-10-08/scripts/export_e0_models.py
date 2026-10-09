#!/usr/bin/env python3
"""Export saved E0 milestones without training; keep FP32 trainable precision.

Uses the reviewed project's build_policy/configure_trainable/export_hf_checkpoint.
Writes checkpoints/global_step_N/hf_ckpt_fp32 through a temporary directory.
An export receipt verifies file structure and exact trained parameter values.
It is NOT an action-consistency audit or a simulation evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/lingbot-vla-robotwin"))
    parser.add_argument("--run", type=Path, default=Path(
        "/data/lingbot-vla-robotwin/runs/e0_jiali_20261008_004327"))
    parser.add_argument("--steps", nargs="+", type=int, default=[5000, 10000, 15000])
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    require(run.is_dir(), f"Run directory missing: {run}")
    require(all(step > 0 for step in args.steps), "Steps must be positive.")
    sys.path.insert(0, str(root / "src"))

    import fcntl
    import random
    import numpy as np
    import torch
    import yaml
    from safetensors import safe_open
    from lrvla.training_checkpoint import export_hf_checkpoint
    from lrvla.training_identity import base_checkpoint_identity
    from lrvla.training_models import configure_trainable
    from lrvla.training_runtime import build_policy

    lock = (run / "e0_fp32_export.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another export is already running for this run.")

    cfg = yaml.safe_load((run / "training_config.yaml").read_text())
    cli = yaml.safe_load((run / "lingbotvla_cli.yaml").read_text())
    metadata = json.loads((run / "run_metadata.json").read_text())
    require(cfg.get("mode") == "expert", "This helper is for the reviewed expert-mode run.")

    def relative_to(parent, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (parent / path).resolve()

    base = relative_to(root, cfg["base_checkpoint"])
    qwen = relative_to(run, cli["model"]["tokenizer_path"])
    norm = relative_to(run, cli["data"]["norm_stats_file"])
    require(sha256(norm) == metadata["norm_sha256"], "Saved training norm hash mismatch.")
    require(sha256(qwen / "config.json") == metadata["qwen_config_sha256"],
            "Saved Qwen config hash mismatch.")

    jobs = []
    for step in dict.fromkeys(args.steps):
        parent = run / "checkpoints" / f"global_step_{step}"
        checkpoint = parent / "training.pt"
        destination = parent / "hf_ckpt_fp32"
        require(checkpoint.is_file(), f"Training checkpoint missing: {checkpoint}")
        if destination.exists():
            require((destination / "export-receipt.json").is_file(),
                    f"Existing output has no receipt; inspect it first: {destination}")
            receipt = json.loads((destination / "export-receipt.json").read_text())
            require(receipt.get("checkpoint_sha256") == sha256(checkpoint)
                    and receipt.get("step") == step
                    and receipt.get("trainable_values_exact") is True
                    and receipt.get("dtype") == "torch.float32"
                    and receipt.get("base_identity_sha256") == metadata["base_identity"]["sha256"],
                    f"Existing receipt differs; inspect it first: {destination}")
            log(f"Step {step}: existing completed export retained: {destination}")
            continue
        jobs.append((step, checkpoint, destination))
    if not jobs:
        log("All requested exports already have matching receipts. No action audit was run.")
        return

    estimated_bytes = int(metadata["memory"]["total_parameters"]) * 4 * len(jobs)
    free_bytes = shutil.disk_usage(run).free
    require(free_bytes > estimated_bytes + 8 * 2**30,
            f"Need about {estimated_bytes / 2**30:.1f} GiB plus 8 GiB free space; "
            f"available: {free_bytes / 2**30:.1f} GiB.")
    log(f"Will export {len(jobs)} milestones as FP32, approximately {estimated_bytes / 2**30:.1f} GiB.")
    log("Checking frozen base shard hashes. This reads the base model once; please wait.")
    require(base_checkpoint_identity(base) == metadata["base_identity"],
            "Base weights differ from the frozen base used during training.")

    require(torch.cuda.is_available(), "CUDA unavailable; activate the lingbot-vla environment.")
    candidates = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    free_gpu, gpu_index = max(candidates)
    require(free_gpu > 16 * 2**30, "No GPU has 16 GiB free. Wait for a training GPU to be free.")
    torch.cuda.set_device(gpu_index)
    device = f"cuda:{gpu_index}"
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    log(f"Loading the original base on {device}; no optimizer or training loop is started.")
    policy = build_policy(root, cli, base, qwen, device=device, weight_dtype=torch.bfloat16)
    selection = configure_trainable(policy, cfg)
    require(selection == metadata["selection"], "Trainable selection differs from the training run.")
    geometry = {key: getattr(policy.config, key)
                for key in ("chunk_size", "max_action_dim", "max_state_dim")}
    geometry["img_size"] = int(cli["data"]["img_size"])
    require(geometry == metadata["model_geometry"], "Model geometry differs from the training run.")
    policy.eval()
    parameters = dict(policy.named_parameters())
    trainable_names = set(selection["trainable_names"])
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(qwen, padding_side="right", local_files_only=True)

    identity_keys = ("base_identity", "selection", "norm_sha256", "qwen_config_sha256",
                     "model_geometry", "manifest_content_sha256", "loss_normalization",
                     "data_sampling", "distributed")
    for step, checkpoint, destination in jobs:
        log(f"Step {step}: reading saved trainable parameters.")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["state"]["global_step"] == step, f"Wrong step in {checkpoint}")
        for key in identity_keys:
            require(key in payload["metadata"] and key in metadata
                    and payload["metadata"][key] == metadata[key],
                    f"Step {step}: metadata mismatch: {key}")
        weights = payload["trainable_state"]
        require(set(weights) == trainable_names, f"Step {step}: trainable tensor keys differ.")
        with torch.no_grad():
            for name, value in weights.items():
                require(value.shape == parameters[name].shape, f"Shape mismatch: {name}")
                require(value.dtype == torch.float32, f"Expected FP32 trained parameter: {name}")
                require(bool(torch.isfinite(value).all()), f"Nonfinite trained parameter: {name}")
                parameters[name].copy_(value)
        del payload
        temporary = Path(tempfile.mkdtemp(prefix="hf_ckpt_fp32_partial_", dir=destination.parent))
        try:
            log(f"Step {step}: writing FP32 shards. Output may be quiet during this step.")
            export_hf_checkpoint(policy, temporary, dtype=torch.float32, processor=processor)
            index = json.loads((temporary / "model.safetensors.index.json").read_text())
            state = policy.state_dict()
            require(set(index["weight_map"]) == set(state), "Export tensor inventory mismatch.")
            grouped = defaultdict(set)
            for name, filename in index["weight_map"].items():
                grouped[filename].add(name)
            log(f"Step {step}: checking shard headers and all {len(weights)} trained tensors.")
            for filename, names in grouped.items():
                with safe_open(temporary / filename, framework="pt", device="cpu") as reader:
                    require(set(reader.keys()) == names, f"Shard/index mismatch: {filename}")
                    for name in names:
                        require(tuple(reader.get_slice(name).get_shape()) == tuple(state[name].shape),
                                f"Export shape mismatch: {name}")
                        if name in weights:
                            value = reader.get_tensor(name)
                            require(value.dtype == torch.float32 and torch.equal(value, weights[name]),
                                    f"Export changed trained parameter values: {name}")
            receipt = {
                "step": step, "dtype": "torch.float32", "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "base_identity_sha256": metadata["base_identity"]["sha256"],
                "norm_sha256": metadata["norm_sha256"],
                "source_commit": metadata.get("source", {}).get("commit"),
                "cli_sha256": sha256(run / "lingbotvla_cli.yaml"),
                "trainable_values_exact": True, "trainable_tensor_count": len(weights),
                "shards": len(grouped), "tensors": len(index["weight_map"]),
                "action_consistency_audit": "not_run",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (temporary / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            require(not destination.exists(), f"Destination appeared during export: {destination}")
            temporary.rename(destination)
        except BaseException:
            log(f"Incomplete export retained for inspection: {temporary}")
            raise
        del weights, state
        gc.collect()
        log(f"STEP {step} EXPORT OK: {destination}")
    log("ALL EXPORTS COMPLETE. Next: fresh-process action audit, transfer, registration, and evaluation.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"EXPORT STOPPED: {type(error).__name__}: {error}")
        raise
