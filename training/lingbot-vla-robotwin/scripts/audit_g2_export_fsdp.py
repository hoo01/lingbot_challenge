#!/usr/bin/env python3
"""Diagnostic: load the FP32 archive and reproduce training FSDP inference."""
from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.distributed as dist
import yaml
from torch.distributed._composable.fsdp import MixedPrecisionPolicy

from lrvla.training import BCTrainingModule
from lrvla.training_audit import load_action_fixture, physical_action
from lrvla.training_checkpoint import run_relative_path
from lrvla.training_models import ACTION_ROOT, configure_trainable
from lrvla.training_runtime import build_policy


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--step", type=int, required=True)
parser.add_argument("--no-vlm-fsdp", action="store_true")
args = parser.parse_args()
run = args.run.resolve()
export = run / "checkpoints" / f"global_step_{args.step}" / "hf_ckpt_fp32"
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group(backend="nccl")
device = torch.device(f"cuda:{rank}")
cli = yaml.safe_load((run / "lingbotvla_cli.yaml").read_text())
config = yaml.safe_load((root / "configs/train_g2_fsdp_state_test.yaml").read_text())
qwen = run_relative_path(run, cli["model"]["tokenizer_path"])
policy = build_policy(root, cli, export, qwen, device=str(device),
                      load_weights=True, weight_dtype=torch.bfloat16)
configure_trainable(policy, config)
policy.config.use_cache = True
policy.model.config.use_cache = True
policy.model._use_compile_predict_velocity = False
expert = policy.get_submodule(ACTION_ROOT)
from lingbotvla.distributed.parallel_state import init_parallel_state
from lingbotvla.distributed.torch_parallelize import build_parallelize_model
init_parallel_state(dp_size=dist.get_world_size(), dp_mode="fsdp2", device_type="cuda")
model = build_parallelize_model(
    BCTrainingModule(policy), enable_full_shard=True,
    enable_mixed_precision=False, enable_fp32=False,
    enable_gradient_checkpointing=False,
    basic_modules=[type(expert.layers[0]).__name__],
    fsdp_llm_blocks=False, vlm_fsdp=not args.no_vlm_fsdp, init_device="cuda",
    fsdp_kwargs={"ignored_params": {p for p in policy.parameters() if not p.requires_grad},
                 "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                   reduce_dtype=torch.float32,
                                                   output_dtype=torch.bfloat16)},
)
model.eval()
reference = torch.load(run / "action_audit_reference.pt", map_location="cpu", weights_only=True)
observation, batch = load_action_fixture(reference["fixture"], device)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    normalized = model(batch, reduction="sample")[0].float().cpu()
expected = reference["normalized"]
physical = None
if rank == 0:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(qwen, padding_side="right", local_files_only=True)
    data_config = dict(cli["data"])
    for name in ("norm_stats_file", "robot_config_root", "train_path"):
        data_config[name] = str(run_relative_path(run, data_config[name]))
    robot_config = Path(data_config["robot_config_root"]) / "robotwin.yaml"
    physical = physical_action(normalized, observation, processor, data_config,
                               robot_config, policy.config)
report = {
    "rank": rank, "world_size": dist.get_world_size(), "shape": list(normalized.shape),
    "max_abs_error": float((normalized - expected).abs().max()),
    "allclose_1e6": bool(torch.allclose(normalized, expected, atol=1e-6, rtol=1e-6)),
}
if physical is not None:
    report.update(physical_shape=list(physical.shape),
                  physical_max_abs_error=float((physical - reference["physical"]).abs().max()),
                  physical_allclose_1e6=bool(torch.allclose(
                      physical, reference["physical"], atol=1e-6, rtol=1e-6)),
                  physical_finite=bool(torch.isfinite(physical).all()))
print(json.dumps(report), flush=True)
if rank == 0:
    report_path = run / "export-upstream-fsdp2-reload-audit.json"
    report_path.write_text(json.dumps(report, indent=2))
dist.barrier()
dist.destroy_process_group()
if not report["allclose_1e6"] or (physical is not None and not report["physical_allclose_1e6"]):
    raise AssertionError("Reloaded FP32 archive under training FSDP does not match")
