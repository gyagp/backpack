"""Tests for scripts/test_matrix.py — verifies acceptance criteria."""

import json
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import test_matrix


# --- Fixtures ---

@pytest.fixture
def models_dir(tmp_path):
    """Create a fake ai-models directory with two models."""
    m1 = tmp_path / "llama-7b"
    (m1 / "gguf").mkdir(parents=True)
    (m1 / "gguf" / "model.gguf").write_text("fake")
    (m1 / "onnx").mkdir()
    (m1 / "onnx" / "model.onnx").write_text("fake")

    m2 = tmp_path / "phi-2"
    (m2 / "gguf").mkdir(parents=True)
    (m2 / "gguf" / "phi.gguf").write_text("fake")
    # phi-2 has no onnx

    # non-model file at top level
    (tmp_path / "readme.txt").write_text("ignore me")

    # empty dir (no format subfolders)
    (tmp_path / "empty-model").mkdir()

    return tmp_path


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "output"


# --- AC: scripts/test_matrix.py exists ---

def test_script_exists():
    script = Path(__file__).resolve().parent.parent / "scripts" / "test_matrix.py"
    assert script.is_file(), "scripts/test_matrix.py must exist"


# --- AC: Discovers all models in ai-models directory ---

def test_discover_models_finds_all(models_dir):
    models = test_matrix.discover_models(models_dir)
    names = [m["name"] for m in models]
    assert "llama-7b" in names
    assert "phi-2" in names
    assert len(models) == 2


def test_discover_models_detects_formats(models_dir):
    models = test_matrix.discover_models(models_dir)
    by_name = {m["name"]: m for m in models}
    assert "gguf" in by_name["llama-7b"]["formats"]
    assert "onnx" in by_name["llama-7b"]["formats"]
    assert "gguf" in by_name["phi-2"]["formats"]
    assert "onnx" not in by_name["phi-2"]["formats"]


def test_discover_models_ignores_files_and_empty(models_dir):
    models = test_matrix.discover_models(models_dir)
    names = [m["name"] for m in models]
    assert "readme.txt" not in names
    assert "empty-model" not in names


def test_discover_models_missing_dir(tmp_path):
    models = test_matrix.discover_models(tmp_path / "nonexistent")
    assert models == []


# --- AC: Tests each model×format×backend combination ---

def test_run_inference_missing_exe(tmp_path):
    r = test_matrix.run_inference(
        tmp_path / "no.exe", "model.gguf", "gguf", "d3d12", 10
    )
    assert r["status"] == "skip"


def test_run_inference_timeout(tmp_path):
    wrapper = tmp_path / "wrap.py"
    wrapper.write_text("import sys, time; time.sleep(30)")
    exe = tmp_path / "run_slow.py"
    exe.write_text(f"import subprocess, sys; subprocess.run([sys.executable, r'{wrapper}'])")
    with patch.object(test_matrix, "run_inference", wraps=test_matrix.run_inference):
        r = _run_with_mock_cmd(lambda: "import time; time.sleep(30)", timeout=1)
    assert r["status"] == "timeout"
    assert "timeout" in r["error"].lower()


def _run_with_mock_cmd(script_body_fn=None, timeout=10, exit_code=None):
    """Helper: mock subprocess.run to simulate inference outcomes."""
    import subprocess as sp
    if exit_code is not None:
        result = MagicMock()
        result.returncode = exit_code
        result.stderr = ""
        with patch("test_matrix.subprocess.run", return_value=result), \
             patch.object(Path, "is_file", return_value=True):
            return test_matrix.run_inference(
                Path("fake.exe"), "model.gguf", "gguf", "d3d12", timeout
            )
    body = script_body_fn()
    original_run = sp.run

    def mock_run(cmd, **kwargs):
        new_cmd = [sys.executable, "-c", body]
        return original_run(new_cmd, **kwargs)

    with patch("test_matrix.subprocess.run", side_effect=mock_run), \
         patch.object(Path, "is_file", return_value=True):
        return test_matrix.run_inference(
            Path("fake.exe"), "model.gguf", "gguf", "d3d12", timeout
        )


def test_run_inference_success_mock():
    r = _run_with_mock_cmd(lambda: "print('hello')", timeout=10)
    assert r["status"] == "pass"
    assert r["duration_s"] > 0


def test_run_inference_failure_mock():
    r = _run_with_mock_cmd(exit_code=1)
    assert r["status"] == "fail"


def test_run_inference_timeout_mock():
    r = _run_with_mock_cmd(lambda: "import time; time.sleep(30)", timeout=1)
    assert r["status"] == "timeout"
    assert "timeout" in r["error"].lower()


# --- AC: Outputs results.json with per-combination pass/fail ---

def test_generate_results_json(output_dir):
    output_dir.mkdir()
    results = [
        {"model": "m1", "format": "gguf", "backend": "d3d12", "status": "pass",
         "duration_s": 1.0, "error": None, "model_path": "p"},
        {"model": "m1", "format": "gguf", "backend": "vulkan", "status": "fail",
         "duration_s": 2.0, "error": "crash", "model_path": "p"},
        {"model": "m1", "format": "onnx", "backend": "d3d12", "status": "timeout",
         "duration_s": 300.0, "error": "exceeded", "model_path": "p"},
    ]
    out = output_dir / "results.json"
    test_matrix.generate_results_json(results, out)

    data = json.loads(out.read_text())
    assert "summary" in data
    assert "results" in data
    assert data["summary"]["total"] == 3
    assert data["summary"]["passed"] == 1
    assert data["summary"]["failed"] == 1
    assert data["summary"]["timeout"] == 1
    assert len(data["results"]) == 3


# --- AC: Outputs results.md with formatted table ---

def test_generate_results_md(output_dir):
    output_dir.mkdir()
    models = [{"name": "m1", "path": "/m1", "formats": {"gguf": ["f"], "onnx": ["f"]}}]
    results = [
        {"model": "m1", "format": "gguf", "backend": "d3d12", "status": "pass"},
        {"model": "m1", "format": "gguf", "backend": "vulkan", "status": "fail"},
        {"model": "m1", "format": "onnx", "backend": "d3d12", "status": "timeout"},
        {"model": "m1", "format": "onnx", "backend": "vulkan", "status": "skip"},
    ]
    out = output_dir / "results.md"
    test_matrix.generate_results_md(results, models, out)

    md = out.read_text()
    assert "# Test Matrix Results" in md
    assert "GGUF+D3D12" in md
    assert "ONNX+VULKAN" in md
    assert "PASS" in md
    assert "FAIL" in md
    assert "TIMEOUT" in md
    assert "m1" in md
    # Verify it's a markdown table
    assert "| Model |" in md
    assert "| --- |" in md


# --- AC: Handles timeouts and OOM gracefully ---

def test_timeout_handled_gracefully():
    r = _run_with_mock_cmd(lambda: "import time; time.sleep(999)", timeout=1)
    assert r["status"] == "timeout"
    assert r["error"] is not None
    assert r["duration_s"] >= 1.0


def test_oom_status_in_json(output_dir):
    output_dir.mkdir()
    results = [
        {"model": "m1", "format": "gguf", "backend": "d3d12", "status": "oom",
         "duration_s": 1.0, "error": "out of memory", "model_path": "p"},
    ]
    out = output_dir / "results.json"
    test_matrix.generate_results_json(results, out)
    data = json.loads(out.read_text())
    assert data["summary"]["oom"] == 1


# --- Integration: dry-run mode ---

def test_main_dry_run(models_dir, capsys):
    with patch("sys.argv", ["test_matrix.py",
                            "--models-dir", str(models_dir),
                            "--dry-run"]):
        with pytest.raises(SystemExit) as exc:
            test_matrix.main()
        assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "llama-7b" in out
    assert "phi-2" in out
