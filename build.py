import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
THIRD_PARTY = ROOT_DIR / "third_party"
DAWN_DIR = THIRD_PARTY / "dawn"
DAWN_BUILD = DAWN_DIR / "build"
DAWN_DLL_DIR = DAWN_BUILD
RUNTIME_DIR = ROOT_DIR / "runtime"
RUNTIME_BUILD = ROOT_DIR / "gitignore" / "runtime" / "build"

DAWN_REPO = "https://dawn.googlesource.com/dawn"

def write_step(msg):
    print(f"\n\033[36m=== {msg} ===\033[0m\n")

def write_ok(msg):
    print(f"  \033[32m[OK]\033[0m {msg}")

def write_skip(msg):
    print(f"  \033[33m[SKIP]\033[0m {msg}")

def write_error(msg):
    print(f"  \033[31m[ERROR]\033[0m {msg}")

def run_cmd(cmd, cwd=None, env=None, check=True):
    try:
        result = subprocess.run(cmd, cwd=cwd, env=env, check=check, shell=True)
    except subprocess.CalledProcessError as e:
        if check:
            sys.exit(e.returncode)
        else:
            return False
    return result.returncode == 0

def assert_command(cmd, hint):
    if shutil.which(cmd) is None:
        write_error(f"'{cmd}' not found. {hint}")
        sys.exit(1)

def invoke_deps():
    write_step("Installing Python dependencies")
    run_cmd(f"{sys.executable} -m pip install setuptools wheel cmake ninja")
    run_cmd(f"{sys.executable} -m pip install numpy tiktoken requests")
    write_ok("Python dependencies installed")

def invoke_clone():
    write_step("Cloning third-party repos")

    if (DAWN_DIR / ".git").exists():
        write_skip(f"Dawn already cloned at {DAWN_DIR}")
    else:
        print("  Cloning Dawn...")
        run_cmd(f"git clone {DAWN_REPO} {DAWN_DIR}")
        write_ok("Dawn cloned")

def invoke_msvc():
    write_step("Loading MSVC environment")
    if shutil.which("cl"):
        write_skip("MSVC already loaded (cl.exe found)")
        return os.environ.copy()

    vs_locations = [
        os.environ.get("ProgramFiles", "C:\\Program Files") + r"\Microsoft Visual Studio\2022\Community",
        os.environ.get("ProgramFiles", "C:\\Program Files") + r"\Microsoft Visual Studio\2022\Professional",
        os.environ.get("ProgramFiles", "C:\\Program Files") + r"\Microsoft Visual Studio\2022\Enterprise",
        os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)") + r"\Microsoft Visual Studio\2022\BuildTools"
    ]

    vcvars = None
    for vs in vs_locations:
        candidate = Path(vs) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if candidate.exists():
            vcvars = candidate
            break

    if not vcvars:
        vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            try:
                res = subprocess.check_output(f'"{vswhere}" -latest -property installationPath', shell=True, text=True).strip()
                if res:
                    candidate = Path(res) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    if candidate.exists():
                        vcvars = candidate
            except:
                pass

    if not vcvars:
        write_error("Could not find vcvars64.bat. Install Visual Studio 2022 Build Tools.")
        sys.exit(1)

    def load_manual_msvc_env():
        vs_root = vcvars.parents[3]
        msvc_root = vs_root / "VC" / "Tools" / "MSVC"
        if not msvc_root.exists():
            return None
        versions = sorted([p for p in msvc_root.iterdir() if p.is_dir()], reverse=True)
        if not versions:
            return None
        msvc = versions[0]

        kits_root = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10"
        include_root = kits_root / "Include"
        lib_root = kits_root / "Lib"
        if not include_root.exists() or not lib_root.exists():
            return None
        sdk_versions = sorted([p.name for p in include_root.iterdir() if p.is_dir()], reverse=True)
        sdk = next((v for v in sdk_versions if (lib_root / v).exists()), None)
        if not sdk:
            return None

        new_env = os.environ.copy()
        path_parts = [
            str(msvc / "bin" / "Hostx64" / "x64"),
            str(vs_root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja"),
            str(kits_root / "bin" / sdk / "x64"),
            new_env.get("PATH", ""),
        ]
        include_parts = [
            str(msvc / "include"),
            str(include_root / sdk / "ucrt"),
            str(include_root / sdk / "um"),
            str(include_root / sdk / "shared"),
            str(include_root / sdk / "winrt"),
            str(include_root / sdk / "cppwinrt"),
        ]
        lib_parts = [
            str(msvc / "lib" / "x64"),
            str(lib_root / sdk / "ucrt" / "x64"),
            str(lib_root / sdk / "um" / "x64"),
        ]
        new_env["PATH"] = os.pathsep.join(path_parts)
        new_env["INCLUDE"] = os.pathsep.join(include_parts)
        new_env["LIB"] = os.pathsep.join(lib_parts)
        return new_env

    print(f"  Loading from: {vcvars}")
    try:
        output = subprocess.check_output(
            f'"{vcvars}" >nul 2>&1 && set',
            shell=True,
            text=True,
            timeout=60)
        new_env = os.environ.copy()
        for line in output.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                new_env[k] = v

        # apply to current os.environ
        for k, v in new_env.items():
            os.environ[k] = v

        try:
            cl_ver = subprocess.check_output("cl 2>&1", env=os.environ, shell=True, stderr=subprocess.STDOUT, text=True).splitlines()[0]
        except subprocess.CalledProcessError as e:
            cl_ver = e.output.splitlines()[0] if e.output else "unknown"

        write_ok(f"MSVC loaded: {cl_ver}")
        return new_env
    except Exception as e:
        write_skip(f"vcvars64.bat failed: {e}")
        new_env = load_manual_msvc_env()
        if not new_env:
            write_error("Failed to synthesize MSVC environment")
            sys.exit(1)
        os.environ.update(new_env)
        try:
            cl_ver = subprocess.check_output("cl 2>&1", env=os.environ, shell=True, stderr=subprocess.STDOUT, text=True).splitlines()[0]
        except subprocess.CalledProcessError as ce:
            cl_ver = ce.output.splitlines()[0] if ce.output else "unknown"
        write_ok(f"MSVC loaded from installed paths: {cl_ver}")
        return new_env

def invoke_dawn():
    invoke_msvc()

    if (DAWN_DLL_DIR / "webgpu_dawn.dll").exists() and \
       (DAWN_DLL_DIR / "dxcompiler.dll").exists() and \
       (DAWN_DLL_DIR / "dxil.dll").exists():
        write_skip(f"Dawn DLLs already deployed at {DAWN_DLL_DIR}")
        print("  To rebuild, run: python build.py dawn-clean && python build.py dawn")
        return

    assert_command("cmake", "Install via: pip install cmake")
    assert_command("ninja", "Install via: pip install ninja")
    assert_command("git", "Install Git from https://git-scm.com")

    if not (DAWN_DIR / ".git").exists():
        write_error("Dawn not cloned. Run: python build.py clone")
        sys.exit(1)

    write_step("Configuring Dawn")
    cmake_args = [
        "cmake",
        "-S", str(DAWN_DIR),
        "-B", str(DAWN_BUILD),
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DDAWN_FETCH_DEPENDENCIES=ON",
        "-DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DDAWN_USE_BUILT_DXC=ON",
        "-DDAWN_ENABLE_D3D12=ON",
        "-DDAWN_ENABLE_D3D11=ON",
        "-DDAWN_ENABLE_VULKAN=ON",
        "-DDAWN_ENABLE_NULL=ON",
        "-DDAWN_ENABLE_INSTALL=ON",
        "-DDAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON",
        "-DDAWN_ENABLE_DESKTOP_GL=OFF",
        "-DDAWN_ENABLE_OPENGLES=OFF",
        "-DDAWN_ENABLE_METAL=OFF",
        "-DDAWN_ENABLE_SWIFTSHADER=OFF",
        "-DDAWN_BUILD_SAMPLES=OFF",
        "-DDAWN_BUILD_TESTS=OFF",
        "-DDAWN_BUILD_BENCHMARKS=OFF",
        "-DDAWN_BUILD_NODE_BINDINGS=OFF",
        "-DDAWN_BUILD_PROTOBUF=OFF",
        "-DDAWN_USE_GLFW=OFF",
        "-DDAWN_WERROR=OFF",
        "-DTINT_BUILD_CMD_TOOLS=OFF",
        "-DTINT_BUILD_TESTS=OFF",
        "-DTINT_BUILD_BENCHMARKS=OFF",
        "-DTINT_BUILD_IR_BINARY=OFF",
        "-DTINT_BUILD_TINTD=OFF",
        "-DTINT_ENABLE_INSTALL=OFF",
        # Disable validation for performance
        "-DDAWN_ENABLE_SPIRV_VALIDATION=OFF",
        "-DDAWN_DXC_ENABLE_ASSERTS_IN_NDEBUG=OFF",
        "-DTINT_ENABLE_IR_VALIDATION=OFF"
    ]
    if not run_cmd(" ".join(cmake_args), check=False):
        write_error("Dawn CMake configure failed")
        sys.exit(1)
    write_ok("Dawn configured")

    write_step("Building Dawn (this may take 10-30 minutes)")
    if not run_cmd(f"cmake --build {DAWN_BUILD} --target webgpu_dawn --target dxcompiler", check=False):
        write_error("Dawn build failed")
        sys.exit(1)
    if not run_cmd(f"cmake --build {DAWN_BUILD} --target third_party/copy_dxil_dll", check=False):
        write_error("dxil.dll copy failed")
        sys.exit(1)
    write_ok("Dawn built")

    invoke_dawn_deploy()

def invoke_dawn_deploy():
    write_step("Deploying Dawn DLLs")
    DAWN_DLL_DIR.mkdir(parents=True, exist_ok=True)

    for dll in ["webgpu_dawn.dll", "dxcompiler.dll", "dxil.dll"]:
        src = DAWN_BUILD / dll
        if not src.exists():
            write_error(f"{dll} not found in {DAWN_BUILD}. Build Dawn first.")
            sys.exit(1)
        dst = DAWN_DLL_DIR / dll
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        write_ok(f"Deployed {dll}")

def invoke_verify():
    invoke_runtime()

def invoke_runtime():
    invoke_msvc()
    assert_command("cmake", "Install via: pip install cmake")
    assert_command("ninja", "Install via: pip install ninja")

    dawn_import_lib = DAWN_BUILD / "src" / "dawn" / "native" / "webgpu_dawn.lib"
    missing = [
        path for path in [
            DAWN_DLL_DIR / "webgpu_dawn.dll",
            DAWN_DLL_DIR / "dxcompiler.dll",
            DAWN_DLL_DIR / "dxil.dll",
            dawn_import_lib,
        ]
        if not path.exists()
    ]
    if missing:
        write_error("Dawn artifacts are missing. Run: python build.py dawn")
        for path in missing:
            print(f"  missing: {path}")
        sys.exit(1)

    write_step("Configuring runtime")
    RUNTIME_BUILD.mkdir(parents=True, exist_ok=True)
    configure_cmd = (
        f"cmake -S {RUNTIME_DIR} -B {RUNTIME_BUILD} -G Ninja "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"-DDAWN_SRC={DAWN_DIR} "
        f"-DDAWN_BUILD={DAWN_BUILD} "
        f"-DDAWN_LIB={DAWN_DLL_DIR}"
    )
    if not run_cmd(configure_cmd, check=False):
        write_error("Runtime CMake configure failed")
        sys.exit(1)
    write_ok(f"Runtime configured at {RUNTIME_BUILD}")

    write_step("Building runtime")
    build_cmd = f"cmake --build {RUNTIME_BUILD} --target backpack_runtime --target backpack_llm"
    if not run_cmd(build_cmd, check=False):
        write_error("Runtime build failed")
        sys.exit(1)
    write_ok("Runtime build complete")

def invoke_setup():
    write_step("Backpack full setup")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Root:   {ROOT_DIR}\n")

    invoke_deps()
    invoke_clone()
    invoke_dawn()
    invoke_runtime()

    print("\n\033[32m============================================\033[0m")
    print("\033[32m  Backpack setup complete!\033[0m")
    print("\033[32m============================================\033[0m\n")
    print("  Verify:  python build.py verify")
    print("  Run:     gitignore\\runtime\\build\\backpack_llm.exe <model.gguf>\n")

def invoke_dawn_clean():
    write_step("Cleaning Dawn build")
    if DAWN_BUILD.exists():
        shutil.rmtree(DAWN_BUILD, ignore_errors=True)
    write_ok("Dawn build cleaned")

def invoke_clean():
    invoke_dawn_clean()
    write_step("Cleaning runtime build")
    if RUNTIME_BUILD.exists():
        shutil.rmtree(RUNTIME_BUILD, ignore_errors=True)
    write_ok("Runtime build cleaned")

def invoke_info():
    print("\n\033[36mBackpack Build Info\033[0m")
    print("===================")
    print(f"  ROOT_DIR:     {ROOT_DIR}")
    print(f"  DAWN_DIR:     {DAWN_DIR}")
    print(f"  DAWN_DLL_DIR: {DAWN_DLL_DIR}\n")
    print(f"  RUNTIME_BUILD:{RUNTIME_BUILD}")

    dawn_dll = DAWN_DLL_DIR / "webgpu_dawn.dll"
    runtime_exe = RUNTIME_BUILD / "backpack_runtime.exe"

    print(f"  Dawn DLL:     {'BUILT' if dawn_dll.exists() else 'NOT BUILT'}")
    print(f"  Runtime:      {'BUILT' if runtime_exe.exists() else 'NOT BUILT'}")
    print(f"  MSVC (cl):    {'Available' if shutil.which('cl') else 'Not loaded'}\n")

    run_cmd(f"{sys.executable} --version", check=False)
    run_cmd("cmake --version", check=False)

def main():
    parser = argparse.ArgumentParser(description="Backpack build script")
    parser.add_argument("target", nargs="?", default="setup",
                        choices=["setup", "deps", "clone", "msvc", "dawn", "dawn-deploy",
                                 "runtime", "verify", "clean", "dawn-clean", "info", "help"],
                        help="Build target to run")

    args = parser.parse_args()

    if args.target == "help":
        parser.print_help()
        print("\nTargets:")
        print("  setup         Full provision: deps + clone + dawn (default)")
        print("  deps          Install Python dependencies")
        print("  clone         Clone Dawn")
        print("  msvc          Load MSVC compiler environment")
        print("  dawn          Build Dawn WebGPU runtime from source")
        print("  dawn-deploy   Deploy Dawn DLLs (after manual Dawn build)")
        print("  runtime       Configure and build C++ runtime")
        print("  verify        Build C++ runtime")
        print("  clean         Clean Dawn and runtime builds")
        print("  dawn-clean    Clean Dawn build only")
        print("  info          Show build environment and status")
        return

    # Enable color output on Windows terminal
    if os.name == 'nt':
        os.system('color')

    funcs = {
        "setup": invoke_setup,
        "deps": invoke_deps,
        "clone": invoke_clone,
        "msvc": invoke_msvc,
        "dawn": invoke_dawn,
        "dawn-deploy": invoke_dawn_deploy,
        "runtime": invoke_runtime,
        "verify": invoke_verify,
        "clean": invoke_clean,
        "dawn-clean": invoke_dawn_clean,
        "info": invoke_info
    }

    funcs[args.target]()

if __name__ == "__main__":
    main()
