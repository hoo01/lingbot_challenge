#!/usr/bin/env python3
"""Fail-closed preflight for the H0 Clean-only full Action Expert run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PRECHECK FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve(root: Path, value: str) -> Path:
    expanded = value.replace("${TRAIN_ROOT}", str(root))
    path = Path(os.path.expandvars(expanded)).expanduser()
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--norm-provenance", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int, choices=(2, 8))
    args = parser.parse_args()

    root = args.training_root.resolve()
    config_path = args.config.resolve()
    provenance_path = args.norm_provenance.resolve()
    require(root.is_dir(), f"training root does not exist: {root}")
    require(config_path.is_file(), f"config does not exist: {config_path}")
    require(provenance_path.is_file(), f"norm provenance missing: {provenance_path}")

    cfg = load_yaml(config_path)
    require(cfg["mode"] == "expert", "mode must be expert")
    require(cfg["expert_last_n_layers"] == 36, "must train all 36 expert layers")
    require(cfg.get("action_mode") == "absolute", "action mode must be absolute")
    require(cfg.get("train_projections") is True, "projections must be trainable")
    require(cfg.get("image_augment") is True, "legal training-time image augmentation must be enabled")
    require(cfg.get("chunk_size") == 50, "chunk_size must remain 50")
    require(cfg.get("action_chunk_tail") == "mask", "chunk tail must be masked")
    require(cfg.get("export_dtype") == "float32", "formal export must be float32")

    global_batch = int(cfg["batch_size"]) * args.world_size * int(cfg["gradient_accumulation"])
    require(global_batch == 64, f"global batch is {global_batch}, expected 64")

    manifest_path = resolve(root, cfg["manifest"])
    norm_path = resolve(root, cfg["norm_stats"])
    robot_path = resolve(root, cfg["robot_config"])
    for label, path in (("manifest", manifest_path), ("norm", norm_path), ("robot config", robot_path)):
        require(path.is_file(), f"{label} missing: {path}")

    manifest = load_json(manifest_path)
    require(manifest.get("clean_only") is True, "manifest is not explicitly clean_only=true")
    require(manifest.get("complete_50_tasks") is True, "manifest is not the complete 50-task set")
    datasets = manifest.get("datasets", [])
    require(len(datasets) == 50, f"manifest has {len(datasets)} datasets, expected 50")
    serialized = json.dumps(manifest, ensure_ascii=False).lower()
    require("randomized" not in serialized, "manifest contains a randomized reference")
    require(all("clean" in str(item.get("path", "")).lower() for item in datasets),
            "at least one dataset path is not explicitly clean")

    provenance = load_json(provenance_path)
    require(provenance.get("clean_only") is True, "norm provenance is not clean-only")
    require(provenance.get("train_episode_ids") == list(range(45)), "norm train split must be episodes 0..44")
    require(provenance.get("val_episode_ids") == list(range(45, 50)), "norm val split must be episodes 45..49")
    expected_norm_sha = provenance.get("norm_stats_sha256")
    require(expected_norm_sha == sha256(norm_path), "norm SHA does not match provenance")

    robot = load_yaml(robot_path)
    actions = robot.get("actions", [])
    require(len(actions) == 2, "robot config must contain arm and effector action groups")
    for group in actions:
        spec = next(iter(group.values()))
        require(spec.get("subtract_state") is False, "all H0 actions must be absolute")
        require("relative_type" not in spec, "relative_type is forbidden in H0")

    print("PRECHECK PASSED")
    print(f"config={config_path}")
    print(f"manifest_sha256={sha256(manifest_path)}")
    print(f"norm_sha256={sha256(norm_path)}")
    print(f"robot_config_sha256={sha256(robot_path)}")
    print(f"world_size={args.world_size}")
    print(f"global_batch_size={global_batch}")
    print("required_post-start_check=trainable inventory must include expert layers 0..35")


if __name__ == "__main__":
    main()
