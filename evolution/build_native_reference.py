"""Build native ORT/GenAI in isolated workspace directories on webgfx-104."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def ignored_path(workspace: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((workspace.resolve() / "gitignore").resolve()):
        raise ValueError(f"Native build path must be inside workspace gitignore: {resolved}")
    return resolved


@dataclass(frozen=True)
class BuildPaths:
    workspace: Path
    ort: Path
    genai: Path
    logs: Path
    ort_build_override: Path | None = None
    genai_build_override: Path | None = None
    ort_home_override: Path | None = None

    def checked(self) -> "BuildPaths":
        paths = [ignored_path(self.workspace, path) for path in (self.ort, self.genai, self.logs)]
        overrides = [ignored_path(self.workspace, path) if path else None
                     for path in (self.ort_build_override, self.genai_build_override, self.ort_home_override)]
        checked = BuildPaths(self.workspace.resolve(), *paths, *overrides)
        if checked.ort_build.name != "Release":
            raise ValueError("ORT's Release build directory must end in Release")
        return checked

    @property
    def ort_build(self) -> Path:
        return self.ort_build_override or self.ort / "build/native/Release"

    @property
    def genai_build(self) -> Path:
        return self.genai_build_override or self.genai / "build/native/Release"

    @property
    def ort_home(self) -> Path:
        return self.ort_home_override or self.ort / "install/Release"


def generator_for(build_dir: Path) -> str:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return "Visual Studio 17 2022"
    match = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.+)$", cache.read_text(encoding="utf-8"), re.M)
    if not match or match[1].strip() not in {"Visual Studio 17 2022", "Ninja"}:
        raise ValueError(f"Unsupported existing native build generator: {cache}")
    return match[1].strip()


def build_plan(paths: BuildPaths, python: Path, jobs: int) -> list[dict]:
    paths = paths.checked()
    if jobs < 1:
        raise ValueError("Build parallelism must be positive")
    def step(label: str, argv: list[str], cwd: Path) -> dict:
        return {"label": label, "argv": argv, "cwd": str(cwd)}
    genai_generator = generator_for(paths.genai_build)
    genai_platform = ["-A", "x64"] if genai_generator.startswith("Visual Studio") else []
    genai_library = paths.genai_build / "Release" if genai_platform else paths.genai_build
    helper_dir = paths.logs / "helpers"
    return [
        step("ort-submodule", ["git", "-c", "core.longpaths=true", "submodule", "update", "--init",
                               "--recursive", "cmake/external/onnx"], paths.ort),
        step("ort-configure", [str(python), "-B", str(paths.ort / "tools/ci_build/build.py"),
            "--build_dir", str(paths.ort_build.parent), "--config", "Release", "--update",
            "--use_webgpu", "--build_shared_lib", "--skip_submodule_sync", "--compile_no_warning_as_error",
            "--cmake_generator", generator_for(paths.ort_build), "--cmake_extra_defines",
            "onnxruntime_BUILD_UNIT_TESTS=OFF", "onnxruntime_ENABLE_PYTHON=OFF",
            "onnxruntime_ENABLE_DAWN_BACKEND_D3D12=ON", "onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=OFF",
            "DAWN_USE_AGILITY_SDK=OFF", "DAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON"], paths.ort),
        step("ort-build", ["cmake", "--build", str(paths.ort_build), "--config", "Release",
                           "--target", "onnxruntime", "onnxruntime_providers_shared", "--parallel", str(jobs)], paths.ort),
        step("ort-install", ["cmake", "--install", str(paths.ort_build), "--config", "Release",
                             "--prefix", str(paths.ort_home)], paths.ort),
        step("genai-configure", ["cmake", "-S", str(paths.genai), "-B", str(paths.genai_build),
            "-G", genai_generator, *genai_platform, "-DCMAKE_BUILD_TYPE=Release",
            "-DUSE_CUDA=OFF", "-DUSE_DML=OFF", "-DUSE_WEBGPU=ON", "-DENABLE_PYTHON=OFF",
            "-DENABLE_TESTS=OFF", "-DENABLE_TELEMETRY=OFF", "-DENABLE_MODEL_BENCHMARK=ON",
            "-DORT_HOME=" + str(paths.ort_home)], paths.genai),
        step("genai-build", ["cmake", "--build", str(paths.genai_build), "--config", "Release",
                             "--target", "onnxruntime-genai", "model_benchmark", "--parallel", str(jobs)], paths.genai),
        step("reference-helper", ["cl", "/nologo", "/O2", "/std:c++20", "/EHsc", "/MD",
            "/I" + str(paths.workspace / "runtime"), "/I" + str(paths.genai / "src"),
            str(paths.workspace / "benchmarks/genai_state_reference.cpp"),
            str(paths.workspace / "runtime/json_parser.cpp"), "/Fo" + str(helper_dir) + "/",
            "/Fe" + str(helper_dir / "genai_state_reference.exe"), "/link",
            str(genai_library / "onnxruntime-genai.lib")], helper_dir),
    ]


def normalize_ort_install(home: Path) -> None:
    # The GenAI SDK expects DLLs beside import libraries, and headers directly
    # in include/. Preserve the original install layout too.
    for source, target in [(home / "bin", home / "lib"), (home / "include/onnxruntime", home / "include")]:
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
    for path in [home / "lib/onnxruntime.dll", home / "lib/onnxruntime.lib", home / "include/onnxruntime_c_api.h"]:
        if not path.is_file():
            raise RuntimeError(f"ORT installation is incomplete: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--ort-source", type=Path, required=True)
    parser.add_argument("--genai-source", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--ort-build-dir", type=Path)
    parser.add_argument("--genai-build-dir", type=Path)
    parser.add_argument("--ort-home", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname().lower() != "webgfx-104":
        parser.error("this goal only permits native builds on webgfx-104")
    paths = BuildPaths(args.workspace, args.ort_source, args.genai_source, args.log_root,
                       args.ort_build_dir, args.genai_build_dir, args.ort_home).checked()
    paths.logs.mkdir(parents=True, exist_ok=True)
    venv = paths.logs / "venv"
    python = venv / "Scripts/python.exe"
    plan = build_plan(paths, python, args.jobs)
    (paths.logs / "build-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.plan_only:
        print(paths.logs / "build-plan.json")
        return 0
    active = subprocess.check_output(["powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('backpack_llm.exe',"
        "'backpack_state_reference.exe','genai_state_reference.exe','model_benchmark.exe') } | "
        "Select-Object -ExpandProperty ProcessId"], text=True)
    if active.strip():
        raise RuntimeError("Inference is running; defer compilation until its measurements finish")
    # The refresh wrapper prepares isolated worktrees and enforces the Windows
    # subgroup-matrix exclusion before invoking this build driver.
    context = paths.ort / "onnxruntime/core/providers/webgpu/webgpu_context.cc"
    source = context.read_text(encoding="utf-8-sig")
    guarded_request = re.search(r"#if !defined\(_WIN32\)\s*\n\s*wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix,\s*\n#endif", source)
    if "Backpack Windows policy: subgroup-matrix feature intentionally not requested." not in source and not guarded_request:
        raise RuntimeError("The refresh wrapper has not applied the Windows subgroup-matrix exclusion")
    temporary = paths.workspace / "gitignore/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(temporary)
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    sys.path.insert(0, str(paths.workspace))
    import build
    environment = build.invoke_msvc()
    environment.update(VIRTUAL_ENV=str(venv), TMP=str(temporary), TEMP=str(temporary),
                       PIP_CACHE_DIR=str(temporary / "pip"))
    environment["PATH"] = str(venv / "Scripts") + os.pathsep + environment["PATH"]
    with (paths.logs / "python-dependencies.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(paths.ort / "requirements.txt")],
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    for item in plan:
        print("START " + item["label"], flush=True)
        if item["label"] == "reference-helper":
            Path(item["cwd"]).mkdir(parents=True, exist_ok=True)
        with (paths.logs / (item["label"] + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(item["argv"], cwd=item["cwd"], env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if item["label"] == "ort-install":
            normalize_ort_install(paths.ort_home)
    genai_library = paths.genai_build / "Release" if generator_for(paths.genai_build).startswith("Visual Studio") else paths.genai_build
    layout = {"ort_home": str(paths.ort_home), "ort_build": str(paths.ort_build),
              "genai_build": str(paths.genai_build), "genai_library": str(genai_library),
              "reference_helper": str(paths.logs / "helpers/genai_state_reference.exe")}
    (paths.logs / "build-layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    print("Native ORT/GenAI build complete; model conformance remains a separate gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
