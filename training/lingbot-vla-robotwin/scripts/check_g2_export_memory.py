#!/usr/bin/env python3
"""Read-only diagnostic: compare resumed FSDP memory tensors with existing HF export."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml
from safetensors import safe_open

from lrvla import training
from lrvla.training_checkpoint import merged_state_items


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/data/tmp/g2_fsdp_state_real_01")
EXPORT = RUN / "checkpoints/global_step_2/hf_ckpt_fp32"
REFERENCE = RUN / "action_audit_reference.pt"
prior_action = torch.load(REFERENCE, map_location="cpu", weights_only=True)["normalized"].clone()


def compare_existing_export(model, output, *, dtype, processor):
    assert Path(output) == EXPORT and dtype == torch.float32
    current_action = torch.load(REFERENCE, map_location="cpu", weights_only=True)["normalized"]
    index = json.loads((EXPORT / "model.safetensors.index.json").read_text())["weight_map"]
    seen = set()
    max_error = 0.0
    mismatches = []
    for name, tensor in merged_state_items(model, dtype):
        if name not in index:
            raise ValueError(f"Missing from export: {name}")
        with safe_open(EXPORT / index[name], framework="pt", device="cpu") as reader:
            archived = reader.get_tensor(name)
        error = (archived - tensor).abs().max().item()
        if error:
            mismatches.append((name, error))
        max_error = max(max_error, error)
        seen.add(name)
    if seen != set(index):
        raise ValueError(f"Export has {len(set(index) - seen)} extra tensors")
    report = {
        "tensors": len(seen), "memory_vs_export_max_abs_error": max_error,
        "nonidentical_tensors": len(mismatches), "first_mismatches": mismatches[:10],
        "repeat_fsdp_action_max_abs_error": float((prior_action - current_action).abs().max()),
    }
    print(json.dumps(report), flush=True)
    (RUN / "export-memory-audit.json").write_text(json.dumps(report, indent=2))
    return EXPORT


training.export_hf_checkpoint = compare_existing_export
config = yaml.safe_load((ROOT / "configs/train_g2_fsdp_state_test.yaml").read_text())
config["resume"] = str(RUN / "checkpoints/global_step_2")
training.train(config)
