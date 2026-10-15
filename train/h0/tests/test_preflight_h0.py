from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


H0_ROOT = Path(__file__).parents[1]
REPO_ROOT = H0_ROOT.parents[1]
SCRIPT = H0_ROOT / "scripts" / "preflight_h0.py"
H0_CONFIG = H0_ROOT / "configs" / "train_h0_full_expert_2gpu.yaml"
RUNTIME_CONFIG = (
    REPO_ROOT
    / "training"
    / "lingbot-vla-robotwin"
    / "configs"
    / "train_g2_fsdp.yaml"
)
SPEC = importlib.util.spec_from_file_location("preflight_h0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def valid_norm(count: int = 123) -> dict:
    groups = {}
    for name, dimension in {
        "observation.state.arm.position": 12,
        "observation.state.effector.position": 2,
        "action.arm.position": 12,
        "action.effector.position": 2,
    }.items():
        groups[name] = {
            "mean": [0.0] * dimension,
            "std": [1.0] * dimension,
            "q01": [-0.9] * dimension,
            "q02": [-0.8] * dimension,
            "q98": [0.8] * dimension,
            "q99": [0.9] * dimension,
            "min": [-1.0] * dimension,
            "max": [1.0] * dimension,
        }
    return {"norm_stats": groups, "count": count}


class ConfigConsistencyTests(unittest.TestCase):
    def test_h0_recipe_matches_the_runtime_g2_preset(self) -> None:
        h0 = preflight.load_yaml(H0_CONFIG)
        runtime = preflight.load_yaml(RUNTIME_CONFIG)
        for key, expected in runtime.items():
            if key == "swanlab":
                for nested_key, nested_expected in expected.items():
                    if nested_key != "description":
                        self.assertEqual(h0[key][nested_key], nested_expected, nested_key)
            else:
                self.assertEqual(h0[key], expected, key)
        self.assertEqual(h0["min_free_disk_gib"], 120)


class ModelAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "models" / "lingbot-vla-v2-6b"
        self.qwen = self.root / "models" / "Qwen3-VL-4B-Instruct"
        self.architecture = self.root / "vendor" / "architecture.yaml"
        self.base.mkdir(parents=True)
        self.qwen.mkdir(parents=True)
        self.architecture.parent.mkdir(parents=True)
        self.architecture.write_text("model: {}\n", encoding="utf-8")
        (self.base / "config.json").write_text("{}\n", encoding="utf-8")
        (self.base / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
        (self.base / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
        (self.base / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "parameter.a": "model-00001-of-00002.safetensors",
                "parameter.b": "model-00002-of-00002.safetensors",
            }
        }), encoding="utf-8")
        for filename in (
            "config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"
        ):
            (self.qwen / filename).write_text("{}\n", encoding="utf-8")
        self.cfg = {
            "base_checkpoint": "models/lingbot-vla-v2-6b",
            "qwen_path": "models/Qwen3-VL-4B-Instruct",
            "architecture_yaml": "vendor/architecture.yaml",
        }
        self.base_identity = preflight.safetensors_identity(self.base)["sha256"]
        self.qwen_sha = preflight.sha256(self.qwen / "config.json")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_exact_indexed_model_assets(self) -> None:
        result = preflight.validate_model_assets(
            self.root,
            self.cfg,
            expected_base_identity=self.base_identity,
            expected_qwen_config_sha256=self.qwen_sha,
        )
        self.assertEqual(result, (2, self.base_identity, self.qwen_sha))

    def test_rejects_missing_base_shard(self) -> None:
        (self.base / "model-00002-of-00002.safetensors").unlink()
        with self.assertRaisesRegex(SystemExit, "base shard missing or empty"):
            preflight.validate_model_assets(
                self.root,
                self.cfg,
                expected_base_identity=self.base_identity,
                expected_qwen_config_sha256=self.qwen_sha,
            )

    def test_rejects_wrong_base_bytes(self) -> None:
        (self.base / "model-00001-of-00002.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(SystemExit, "base checkpoint identity"):
            preflight.validate_model_assets(
                self.root,
                self.cfg,
                expected_base_identity=self.base_identity,
                expected_qwen_config_sha256=self.qwen_sha,
            )

    def test_rejects_missing_qwen_processor_file(self) -> None:
        (self.qwen / "tokenizer.json").unlink()
        with self.assertRaisesRegex(SystemExit, "Qwen processor file missing or empty"):
            preflight.validate_model_assets(
                self.root,
                self.cfg,
                expected_base_identity=self.base_identity,
                expected_qwen_config_sha256=self.qwen_sha,
            )


class CleanManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        datasets = []
        for index in range(50):
            task = f"task_{index:02d}"
            dataset = self.root / "data" / "lerobot" / f"{task}_clean_50"
            (dataset / "meta").mkdir(parents=True)
            (dataset / "meta" / "info.json").write_text(
                json.dumps({"codebase_version": "v3.0"}), encoding="utf-8"
            )
            parquet = dataset / "data" / "chunk-000" / "file-000.parquet"
            parquet.parent.mkdir(parents=True)
            parquet.touch()
            datasets.append({
                "task": task,
                "path": f"${{TRAIN_ROOT}}/data/lerobot/{task}_clean_50",
                "episodes": 50,
                "episode_ids": list(range(50)),
                "frames": index + 1,
                "format": "v3.0",
                "source": {"setting": "clean", "episodes": 50, "instructions": "seen"},
                "parquet_files": ["data/chunk-000/file-000.parquet"],
            })
        self.manifest = {
            "clean_only": True,
            "complete_50_tasks": True,
            "total_episodes": 2500,
            "datasets": datasets,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_complete_materialized_clean_manifest(self) -> None:
        tasks, frames = preflight.validate_clean_manifest(self.root, self.manifest)
        self.assertEqual(tasks, 50)
        self.assertEqual(frames, sum(range(1, 51)))

    def test_rejects_duplicate_task(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["datasets"][1]["task"] = manifest["datasets"][0]["task"]
        with self.assertRaisesRegex(SystemExit, "task names are not unique"):
            preflight.validate_clean_manifest(self.root, manifest)

    def test_rejects_missing_parquet(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["datasets"][0]["parquet_files"] = ["data/chunk-000/missing.parquet"]
        with self.assertRaisesRegex(SystemExit, "parquet file missing"):
            preflight.validate_clean_manifest(self.root, manifest)


class NormStatsTests(unittest.TestCase):
    def test_accepts_absolute_14d_schema(self) -> None:
        self.assertEqual(preflight.validate_norm_stats(valid_norm(), {"count": 123}), 123)

    def test_rejects_wrong_dimension(self) -> None:
        norm = valid_norm()
        norm["norm_stats"]["action.arm.position"]["q99"].pop()
        with self.assertRaisesRegex(SystemExit, "expected 12 values"):
            preflight.validate_norm_stats(norm, {"count": 123})

    def test_rejects_non_finite_value(self) -> None:
        norm = valid_norm()
        norm["norm_stats"]["action.effector.position"]["mean"][0] = float("nan")
        with self.assertRaisesRegex(SystemExit, "finite numbers"):
            preflight.validate_norm_stats(norm, {"count": 123})

    def test_rejects_provenance_count_mismatch(self) -> None:
        with self.assertRaisesRegex(SystemExit, "count does not match provenance"):
            preflight.validate_norm_stats(valid_norm(), {"count": 122})


class ReceiptTests(unittest.TestCase):
    def test_writes_json_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            preflight.write_json_atomic(path, {"schema_version": 1, "value": "ok"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"schema_version": 1, "value": "ok"},
            )
            self.assertEqual(list(path.parent.glob(".receipt.json.*.tmp")), [])

    def test_rejects_missing_receipt_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "receipt.json"
            with self.assertRaisesRegex(SystemExit, "receipt parent directory does not exist"):
                preflight.write_json_atomic(path, {})


class StorageTests(unittest.TestCase):
    def test_accepts_writable_nonexistent_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs" / "future-run"
            result = preflight.validate_storage(output, 1)
            self.assertEqual(result["output_dir"], str(output.resolve()))
            self.assertGreaterEqual(result["free_bytes"], 2**30)
            self.assertEqual(result["required_free_bytes"], 2**30)
            self.assertFalse(output.exists())

    def test_rejects_output_path_that_is_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not a directory"):
                preflight.validate_storage(output, 1)

    def test_rejects_insufficient_free_space(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "requires at least"):
                preflight.validate_storage(Path(directory) / "run", 10**9)


if __name__ == "__main__":
    unittest.main()
