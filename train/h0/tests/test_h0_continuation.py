from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml


H0_ROOT = Path(__file__).parents[1]
REPO_ROOT = H0_ROOT.parents[1]
PREFLIGHT_PATH = H0_ROOT / "scripts" / "preflight_h0.py"
SCHEDULE_PATH = REPO_ROOT / "training" / "lingbot-vla-robotwin" / "src" / "lrvla" / "training_schedule.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load_module("preflight_h0_continuation", PREFLIGHT_PATH)
schedule = load_module("training_schedule", SCHEDULE_PATH)


class ContinuationScheduleTests(unittest.TestCase):
    def test_h0_a_finishes_at_five_e_minus_six(self) -> None:
        multiplier = schedule.schedule_multiplier(15000, 15000, 50, minimum=0.1)
        self.assertAlmostEqual(multiplier * 5e-5, 5e-6)

    def test_changed_cosine_horizon_would_jump_but_constant_continuation_does_not(self) -> None:
        changed = schedule.schedule_multiplier(15001, 25000, 0, minimum=0.1)
        continued = schedule.schedule_multiplier(
            15001, 25000, 0, minimum=0.1, mode="constant", constant=0.1
        )
        self.assertGreater(changed * 5e-5, 2e-5)
        self.assertAlmostEqual(continued * 5e-5, 5e-6)

    def test_constant_continuation_rejects_warmup(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not use warmup"):
            schedule.schedule_multiplier(15000, 25000, 1, mode="constant", constant=0.1)


class ContinuationConfigTests(unittest.TestCase):
    def test_all_frozen_phase_configs_pass_recipe_validation(self) -> None:
        cases = (
            ("h0-a", "train_h0_full_expert_2gpu.yaml"),
            ("h0-c1", "train_h0_c1_2gpu.yaml"),
            ("h0-c2", "train_h0_c2_2gpu.yaml"),
        )
        for phase, filename in cases:
            with self.subTest(phase=phase):
                config = yaml.safe_load((H0_ROOT / "configs" / filename).read_text(encoding="utf-8"))
                self.assertEqual(preflight.validate_recipe(config, phase, 2), 64)

    def test_c1_parent_must_be_the_frozen_15k_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "global_step_15000"
            checkpoint.mkdir()
            (checkpoint / "COMPLETED.json").write_text(json.dumps({
                "global_step": 15000,
                "world_size": 2,
                "format": "lrvla_upstream_fsdp2_dcp_v1",
                "base_sha256": preflight.EXPECTED_BASE_IDENTITY,
            }), encoding="utf-8")
            export = checkpoint / "hf_ckpt_fp32" / "model.safetensors"
            export.parent.mkdir()
            export.write_bytes(b"export")
            (checkpoint.parent.parent / "action_audit_reference.pt").write_bytes(b"audit")
            (checkpoint.parent.parent / "training_config.yaml").write_text(
                "experiment_phase: h0-a\n", encoding="utf-8"
            )
            result = preflight.validate_parent_checkpoint(
                checkpoint,
                phase="h0-c1",
                world_size=2,
                expected_step=15000,
                expected_parent_phase="h0-a",
            )
            self.assertEqual(result["global_step"], 15000)
            with self.assertRaisesRegex(SystemExit, "expected the frozen stage boundary"):
                preflight.validate_parent_checkpoint(
                    checkpoint,
                    phase="h0-c1",
                    world_size=2,
                    expected_step=14000,
                    expected_parent_phase="h0-a",
                )

    def test_c1_resume_rejects_an_h0_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "run" / "checkpoints" / "global_step_15500"
            checkpoint.mkdir(parents=True)
            (checkpoint / "COMPLETED.json").write_text(json.dumps({
                "global_step": 15500,
                "world_size": 2,
                "format": "lrvla_upstream_fsdp2_dcp_v1",
                "base_sha256": preflight.EXPECTED_BASE_IDENTITY,
            }), encoding="utf-8")
            (checkpoint.parent.parent / "training_config.yaml").write_text(
                "experiment_phase: h0-a\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SystemExit, "expected h0-c1"):
                preflight.validate_parent_checkpoint(
                    checkpoint,
                    phase="h0-c1",
                    world_size=2,
                    expected_parent_phase="h0-c1",
                )


if __name__ == "__main__":
    unittest.main()
