import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
THIRD_PARTY = ROOT_DIR / "third_party"
DAWN_DIR = THIRD_PARTY / "dawn"
TRITON_DIR = THIRD_PARTY / "triton-windows"
DAWN_BUILD = DAWN_DIR / "build"
MODELS_DIR = ROOT_DIR / "models"
DAWN_DLL_DIR = TRITON_DIR / "third_party" / "webgpu" / "dawn" / "build"

DAWN_REPO = "https://dawn.googlesource.com/dawn"
TRITON_REPO = "https://github.com/gyagp/triton-windows.git"
TRITON_BRANCH = "webgpu"

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
        subprocess.run(cmd, cwd=cwd, env=env, check=check, shell=True)
    except subprocess.CalledProcessError as e:
        if check:
            sys.exit(e.returncode)
        else:
            return False
    return True

def assert_command(cmd, hint):
    if shutil.which(cmd) is None:
        write_error(f"'{cmd}' not found. {hint}")
        sys.exit(1)

def invoke_deps():
    write_step("Installing Python dependencies")
    run_cmd(f"{sys.executable} -m pip install setuptools wheel cmake ninja pybind11 lit")
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
        
    if (TRITON_DIR / ".git").exists():
        write_skip(f"triton-windows already cloned at {TRITON_DIR}")
    else:
        print(f"  Cloning triton-windows (branch: {TRITON_BRANCH})...")
        run_cmd(f"git clone -b {TRITON_BRANCH} {TRITON_REPO} {TRITON_DIR}")
        write_ok("triton-windows cloned")

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
        
    print(f"  Loading from: {vcvars}")
    try:
        output = subprocess.check_output(f'"{vcvars}" >nul 2>&1 && set', shell=True, text=True)
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
        write_error(f"Failed to load MSVC environment: {e}")
        sys.exit(1)

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
        "-DTINT_ENABLE_INSTALL=OFF"
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
        shutil.copy2(src, DAWN_DLL_DIR / dll)
        write_ok(f"Deployed {dll}")

def invoke_install():
    write_step("Installing triton on Python path (lightweight)")
    try:
        import site
        site_dir = Path(site.getusersitepackages())
    except Exception as e:
        write_error(f"Failed to get site packages directory: {e}")
        sys.exit(1)
        
    site_dir.mkdir(parents=True, exist_ok=True)
    
    pth_content = f"import os; os.environ.setdefault('TRITON_BACKENDS_IN_TREE', '1')\n{str(TRITON_DIR.as_posix())}/python\n"
    pth_path = site_dir / "triton-backpack.pth"
    pth_path.write_text(pth_content, encoding="utf-8")
    write_ok(f"Wrote {pth_path}")
    
    os.environ["TRITON_BACKENDS_IN_TREE"] = "1"
    if not run_cmd(f'{sys.executable} -c "import triton; print(\'triton loaded from:\', triton.__file__)"', check=False):
        print("  \033[33m[WARN]\033[0m triton import check failed")
    else:
        run_cmd(f'{sys.executable} -c "from triton.backends import backends; print(\'Backends:\', list(backends.keys()))"')
        write_ok("triton registered with webgpu backend")

def invoke_triton():
    invoke_msvc()
    assert_command("cmake", "Install via: pip install cmake")
    assert_command("ninja", "Install via: pip install ninja")
    
    if not (TRITON_DIR / ".git").exists():
        write_error("triton-windows not cloned. Run: python build.py clone")
        sys.exit(1)
        
    write_step("Building triton-windows (this may take 15-45 minutes on first build)")
    
    env = os.environ.copy()
    env["TRITON_BUILD_PROTON"] = "0"
    env["TRITON_BUILD_UT"] = "0"
    env["TRITON_BUILD_BINARY"] = "0"
    
    pip_install_ok = run_cmd(f"{sys.executable} -m pip install --no-build-isolation --verbose -e .", cwd=TRITON_DIR, env=env, check=False)
    
    if not pip_install_ok:
        print("  \033[33m[WARN]\033[0m pip editable install failed, falling back to .pth install")
        invoke_install()
        return
        
    env["TRITON_BACKENDS_IN_TREE"] = "1"
    if not run_cmd(f'{sys.executable} -c "import triton; print(\'triton loaded from:\', triton.__file__)"', env=env, check=False):
        print("  \033[33m[WARN]\033[0m triton import failed after install")
    else:
        write_ok("triton-windows installed")

def invoke_verify():
    write_step("Running GPT-2 verification")
    env = os.environ.copy()
    env["TRITON_BACKENDS_IN_TREE"] = "1"
    
    gpt2_verify = MODELS_DIR / "gpt-2" / "model.py"
    if run_cmd(f"{sys.executable} {gpt2_verify} --verify", env=env, check=False):
        write_ok("Verification passed")
    else:
        write_error("Verification failed")
        sys.exit(1)

def invoke_setup():
    write_step("Backpack full setup")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Root:   {ROOT_DIR}\n")
    
    invoke_deps()
    invoke_clone()
    invoke_dawn()
    invoke_triton()
    
    print("\n\033[32m============================================\033[0m")
    print("\033[32m  Backpack setup complete!\033[0m")
    print("\033[32m============================================\033[0m\n")
    print("  Verify:  python build.py verify")
    print("  Run:     python models/gpt-2/model.py --prompt \"The future of AI is\"\n")

def invoke_dawn_clean():
    write_step("Cleaning Dawn build")
    if DAWN_BUILD.exists():
        shutil.rmtree(DAWN_BUILD, ignore_errors=True)
    if DAWN_DLL_DIR.exists():
        shutil.rmtree(DAWN_DLL_DIR, ignore_errors=True)
    write_ok("Dawn build cleaned")

def invoke_triton_clean():
    write_step("Cleaning triton build")
    triton_build = TRITON_DIR / "build"
    if triton_build.exists():
        shutil.rmtree(triton_build, ignore_errors=True)
    libtriton = TRITON_DIR / "python" / "triton" / "_C" / "libtriton.pyd"
    if libtriton.exists():
        libtriton.unlink()
    write_ok("Triton build cleaned")

def invoke_clean():
    invoke_dawn_clean()
    invoke_triton_clean()

def invoke_info():
    print("\n\033[36mBackpack Build Info\033[0m")
    print("===================")
    print(f"  ROOT_DIR:     {ROOT_DIR}")
    print(f"  DAWN_DIR:     {DAWN_DIR}")
    print(f"  TRITON_DIR:   {TRITON_DIR}")
    print(f"  DAWN_DLL_DIR: {DAWN_DLL_DIR}\n")
    
    dawn_dll = DAWN_DLL_DIR / "webgpu_dawn.dll"
    libtriton = TRITON_DIR / "python" / "triton" / "_C" / "libtriton.pyd"
    
    print(f"  Dawn DLL:     {'BUILT' if dawn_dll.exists() else 'NOT BUILT'}")
    print(f"  libtriton:    {'BUILT' if libtriton.exists() else 'NOT BUILT'}")
    print(f"  MSVC (cl):    {'Available' if shutil.which('cl') else 'Not loaded'}\n")
    
    run_cmd(f"{sys.executable} --version", check=False)
    run_cmd("cmake --version", check=False)

def main():
    parser = argparse.ArgumentParser(description="Backpack build script")
    parser.add_argument("target", nargs="?", default="setup", 
                        choices=["setup", "deps", "clone", "msvc", "dawn", "dawn-deploy", 
                                 "triton", "install", "verify", "clean", "dawn-clean", 
                                 "triton-clean", "info", "help"],
                        help="Build target to run")
    
    args = parser.parse_args()
    
    if args.target == "help":
        parser.print_help()
        print("\nTargets:")
        print("  setup         Full provision: deps + clone + dawn + triton (default)")
        print("  deps          Install Python dependencies")
        print("  clone         Clone Dawn and triton-windows repos")
        print("  msvc          Load MSVC compiler environment")
        print("  dawn          Build Dawn WebGPU runtime from source")
        print("  dawn-deploy   Deploy Dawn DLLs (after manual Dawn build)")
        print("  triton        Build triton-windows from source (pip editable)")
        print("  install       Install triton using .pth link (skips full rebuild)")
        print("  verify        Run GPT-2 pipeline verification")
        print("  clean         Clean Dawn and triton builds")
        print("  dawn-clean    Clean Dawn build only")
        print("  triton-clean  Clean triton build only")
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
        "triton": invoke_triton,
        "install": invoke_install,
        "verify": invoke_verify,
        "clean": invoke_clean,
        "dawn-clean": invoke_dawn_clean,
        "triton-clean": invoke_triton_clean,
        "info": invoke_info
    }
    
    funcs[args.target]()

if __name__ == "__main__":
    main()
