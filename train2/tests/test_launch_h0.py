from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "launch_h0.sh"


def find_bash() -> str | None:
    override = os.environ.get("BASH_EXE")
    if override:
        return override
    if os.name == "nt":
        for candidate in (
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        ):
            if candidate.is_file():
                return str(candidate)
    return shutil.which("bash")


def shell_path(path: Path) -> str:
    resolved = path.resolve().as_posix()
    if os.name == "nt" and len(resolved) >= 3 and resolved[1:3] == ":/":
        return f"/{resolved[0].lower()}/{resolved[3:]}"
    return resolved


@unittest.skipUnless(find_bash(), "Bash is required for launcher integration tests")
class LauncherIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "runtime"
        self.environment = Path(self.temporary.name) / "environment"
        self.fake_bin = Path(self.temporary.name) / "bin"
        (self.root / "scripts").mkdir(parents=True)
        (self.environment / "bin").mkdir(parents=True)
        self.fake_bin.mkdir(parents=True)

        activate = self.root / "scripts" / "activate.sh"
        activate.write_text(
            f'export CONDA_PREFIX="{shell_path(self.environment)}"\n', encoding="utf-8"
        )
        fake_python = self.environment / "bin" / "python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  if [[ \"$1\" == --receipt ]]; then\n"
            "    printf '{}\\n' > \"$2\"\n"
            "    shift 2\n"
            "  else\n"
            "    shift\n"
            "  fi\n"
            "done\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        fake_nvidia_smi = self.fake_bin / "nvidia-smi"
        fake_nvidia_smi.write_text("#!/usr/bin/env bash\nprintf '0\\n1\\n'\n", encoding="utf-8")
        fake_nvidia_smi.chmod(0o755)

        self.git("init", "-q")
        self.git("config", "user.name", "launcher-test")
        self.git("config", "user.email", "launcher-test@example.invalid")
        self.git("add", "scripts/activate.sh")
        self.git("commit", "-q", "-m", "fixture")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def git(self, *arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=self.root, check=True, capture_output=True)

    def run_launcher(self, *arguments: str, **environment: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(environment)
        env["TRAIN_ROOT"] = shell_path(self.root)
        env["PATH"] = os.pathsep.join((str(self.fake_bin), env.get("PATH", "")))
        return subprocess.run(
            [find_bash(), shell_path(SCRIPT), *arguments],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
        )

    def test_check_does_not_require_gpu(self) -> None:
        result = self.run_launcher("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Preflight complete", result.stdout)

    def test_dirty_tracked_source_is_rejected(self) -> None:
        with (self.root / "scripts" / "activate.sh").open("a", encoding="utf-8") as stream:
            stream.write("# dirty\n")
        result = self.run_launcher("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Tracked files are dirty", result.stderr)

    def test_dry_run_does_not_create_output(self) -> None:
        output = Path(self.temporary.name) / "planned-run"
        result = self.run_launcher("--dry-run", H0_OUTPUT_DIR=shell_path(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN PASSED", result.stdout)
        self.assertIn(shell_path(output), result.stdout)
        self.assertFalse(output.exists())

    def test_existing_formal_output_is_rejected(self) -> None:
        output = Path(self.temporary.name) / "existing-run"
        output.mkdir()
        result = self.run_launcher("--launch", H0_OUTPUT_DIR=shell_path(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Formal output already exists", result.stderr)

    def test_fresh_launch_archives_preflight_receipt(self) -> None:
        output = Path(self.temporary.name) / "fresh-run"
        result = self.run_launcher("--launch", H0_OUTPUT_DIR=shell_path(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((output / "preflight_receipt.json").read_text(encoding="utf-8"), "{}\n")
        launch_record = (output / "launch_record.txt").read_text(encoding="utf-8")
        self.assertIn("preflight_receipt=", launch_record)

    def test_resume_adds_receipt_without_overwriting_original(self) -> None:
        run = Path(self.temporary.name) / "resume-run"
        checkpoint = run / "checkpoints" / "global_step_500"
        checkpoint.mkdir(parents=True)
        (checkpoint / "COMPLETED.json").write_text("{}\n", encoding="utf-8")
        original = run / "preflight_receipt.json"
        original.write_text('{"original": true}\n', encoding="utf-8")

        result = self.run_launcher("--resume", shell_path(checkpoint))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(original.read_text(encoding="utf-8"), '{"original": true}\n')
        resume_receipts = list(run.glob("preflight_resume_*.json"))
        self.assertEqual(len(resume_receipts), 1)
        self.assertEqual(resume_receipts[0].read_text(encoding="utf-8"), "{}\n")

    def test_invalid_smoke_step_is_usage_error(self) -> None:
        result = self.run_launcher("--smoke", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
