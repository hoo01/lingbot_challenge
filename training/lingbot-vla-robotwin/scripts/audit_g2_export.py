#!/usr/bin/env python3
"""Independently reload full FP32 G2 HF export and compare fixed 50x14 actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml
from safetensors import safe_open

from lrvla.training_audit import load_action_fixture, physical_action
from lrvla.training_checkpoint import run_relative_path
from lrvla.training_runtime import build_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    export = run / "checkpoints" / f"global_step_{args.step}" / "hf_ckpt_fp32"
    reference = torch.load(run / "action_audit_reference.pt", map_location="cpu", weights_only=True)
    cli = yaml.safe_load((run / "lingbotvla_cli.yaml").read_text())
    qwen = run_relative_path(run, cli["model"]["tokenizer_path"])
    data_config = dict(cli["data"])
    for name in ("norm_stats_file", "robot_config_root", "train_path"):
        data_config[name] = str(run_relative_path(run, data_config[name]))
    robot_config = Path(data_config["robot_config_root"]) / "robotwin.yaml"
    index = json.loads((export / "model.safetensors.index.json").read_text())
    shards = sorted(export.glob("*.safetensors"))
    if not shards or not index["weight_map"]:
        raise ValueError("Missing full HF safetensors export")
    tensors = 0
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                dtype = reader.get_slice(name).get_dtype()
                if dtype in {"BF16", "F16", "F64"}:
                    raise ValueError(f"Non-FP32 floating tensor in export: {name} {dtype}")
                tensors += 1
    if tensors != len(index["weight_map"]):
        raise ValueError("Index tensor count differs from shards")
    # The archive stores FP32 tensors; compare at the training/evaluation BF16
    # execution precision used by FSDP's mixed-precision forward.
    policy = build_policy(Path(__file__).resolve().parents[1], cli, export, qwen,
                          device="cuda:0", load_weights=True, weight_dtype=torch.bfloat16)
    # Deployment loads the full FP32 archive at the agreed BF16 inference
    # precision. Do not re-upcast trainables to FP32 as training setup does.
    policy.config.use_cache = True
    policy.model.config.use_cache = True
    policy.model._use_compile_predict_velocity = False
    policy.eval()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(qwen, padding_side="right", local_files_only=True)
    observation, batch = load_action_fixture(reference["fixture"], torch.device("cuda:0"))
    batch = {name: (value.to(torch.bfloat16) if value.is_floating_point() else value)
             for name, value in batch.items()}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        normalized = policy.sample_actions(
            batch["images"], batch["img_masks"], batch["lang_tokens"],
            batch["lang_masks"], batch["state"], noise=batch["noise"],
            image_grid_thw=batch["image_grid_thw"])[0].float().cpu()
    physical = physical_action(normalized, observation, processor, data_config,
                               robot_config, policy.config)
    norm_ref, phys_ref = reference["normalized"], reference["physical"]
    normalized_error = (normalized - norm_ref).abs()
    physical_error = (physical - phys_ref).abs()
    report = {
        "export": str(export), "format": "complete_hf_fp32", "shards": len(shards),
        "tensors": tensors, "storage_dtype": "torch.float32",
        "reload_frozen_dtype": "torch.bfloat16", "reload_trainable_dtype": "torch.bfloat16",
        "execution_autocast": "torch.bfloat16", "floating_input_dtype": "torch.bfloat16",
        "normalized_shape": list(normalized.shape), "physical_shape": list(physical.shape),
        "normalized_max_abs_error": float(normalized_error.max()),
        "physical_max_abs_error": float(physical_error.max()),
        "normalized_allclose_1e6": bool(torch.allclose(normalized, norm_ref, atol=1e-6, rtol=1e-6)),
        "physical_allclose_1e6": bool(torch.allclose(physical, phys_ref, atol=1e-6, rtol=1e-6)),
        "physical_finite": bool(torch.isfinite(physical).all()),
    }
    path = run / "action-audit.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["normalized_allclose_1e6"] or not report["physical_allclose_1e6"]:
        raise AssertionError("G2 FP32 export/reload action mismatch at atol=rtol=1e-6")


if __name__ == "__main__":
    main()
