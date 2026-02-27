<#
.SYNOPSIS
    Backpack build script — provisions Dawn and triton-windows from source.

.DESCRIPTION
    Automates all provisioning steps: cloning third-party repos, loading MSVC,
    building Dawn, building triton-windows, and verifying the setup.

.PARAMETER Target
    Build target to run. Defaults to 'setup'.
    Targets: setup, deps, clone, msvc, dawn, dawn-deploy, triton, verify, clean, help

.EXAMPLE
    .\build.ps1              # Full setup
    .\build.ps1 dawn         # Build Dawn only
    .\build.ps1 triton       # Build triton only
    .\build.ps1 verify       # Run GPT-2 verification
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("setup","deps","clone","msvc","dawn","dawn-deploy","triton","install","verify","clean","dawn-clean","triton-clean","info","help")]
    [string]$Target = "setup"
)

$ErrorActionPreference = "Stop"

# ===========================================================================
# Configuration
# ===========================================================================

$ROOT_DIR     = $PSScriptRoot
$THIRD_PARTY  = Join-Path $ROOT_DIR "third_party"
$DAWN_DIR     = Join-Path $THIRD_PARTY "dawn"
$TRITON_DIR   = Join-Path $THIRD_PARTY "triton-windows"
$DAWN_BUILD   = Join-Path $DAWN_DIR "build"
$MODELS_DIR   = Join-Path $ROOT_DIR "models"

# dawn_runner.py searches: <triton_root>/third_party/webgpu/dawn/build/
$DAWN_DLL_DIR = Join-Path $TRITON_DIR "third_party\webgpu\dawn\build"

$DAWN_REPO    = "https://dawn.googlesource.com/dawn"
$TRITON_REPO  = "https://github.com/gyagp/triton-windows.git"
$TRITON_BRANCH = "webgpu"

# ===========================================================================
# Helpers
# ===========================================================================

function Write-Step($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
    Write-Host ""
}

function Write-OK($msg) {
    Write-Host "  [OK] $msg" -ForegroundColor Green
}

function Write-Skip($msg) {
    Write-Host "  [SKIP] $msg" -ForegroundColor Yellow
}

function Assert-Command($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "  [ERROR] '$cmd' not found. $hint" -ForegroundColor Red
        exit 1
    }
}

# ===========================================================================
# Target: deps — install Python packages
# ===========================================================================

function Invoke-Deps {
    Write-Step "Installing Python dependencies"
    pip install setuptools wheel cmake ninja pybind11 lit
    pip install numpy tiktoken requests
    Write-OK "Python dependencies installed"
}

# ===========================================================================
# Target: clone — clone third-party repos (skip if present)
# ===========================================================================

function Invoke-Clone {
    Write-Step "Cloning third-party repos"

    if (Test-Path (Join-Path $DAWN_DIR ".git")) {
        Write-Skip "Dawn already cloned at $DAWN_DIR"
    } else {
        Write-Host "  Cloning Dawn..."
        git clone $DAWN_REPO $DAWN_DIR
        Write-OK "Dawn cloned"
    }

    if (Test-Path (Join-Path $TRITON_DIR ".git")) {
        Write-Skip "triton-windows already cloned at $TRITON_DIR"
    } else {
        Write-Host "  Cloning triton-windows (branch: $TRITON_BRANCH)..."
        git clone -b $TRITON_BRANCH $TRITON_REPO $TRITON_DIR
        Write-OK "triton-windows cloned"
    }
}

# ===========================================================================
# Target: msvc — load Visual Studio compiler environment
# ===========================================================================

function Invoke-MSVC {
    Write-Step "Loading MSVC environment"

    # Check if cl.exe is already available
    if (Get-Command cl -ErrorAction SilentlyContinue) {
        Write-Skip "MSVC already loaded (cl.exe found)"
        return
    }

    # Search for vcvars64.bat
    $vsLocations = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools"
    )

    $vcvars = $null
    foreach ($vs in $vsLocations) {
        $candidate = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $candidate) {
            $vcvars = $candidate
            break
        }
    }

    if (-not $vcvars) {
        # Try vswhere as fallback
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $vsPath = & $vswhere -latest -property installationPath
            if ($vsPath) {
                $candidate = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
                if (Test-Path $candidate) { $vcvars = $candidate }
            }
        }
    }

    if (-not $vcvars) {
        Write-Host "  [ERROR] Could not find vcvars64.bat. Install Visual Studio 2022 Build Tools." -ForegroundColor Red
        exit 1
    }

    Write-Host "  Loading from: $vcvars"
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }

    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        Write-Host "  [ERROR] Failed to load MSVC environment" -ForegroundColor Red
        exit 1
    }

    $clVersion = (cl 2>&1 | Select-Object -First 1)
    Write-OK "MSVC loaded: $clVersion"
}

# ===========================================================================
# Target: dawn — configure, build, and deploy Dawn
# ===========================================================================

function Invoke-Dawn {
    Invoke-MSVC

    # Check if already built and deployed
    if ((Test-Path (Join-Path $DAWN_DLL_DIR "webgpu_dawn.dll")) -and
        (Test-Path (Join-Path $DAWN_DLL_DIR "dxcompiler.dll")) -and
        (Test-Path (Join-Path $DAWN_DLL_DIR "dxil.dll"))) {
        Write-Skip "Dawn DLLs already deployed at $DAWN_DLL_DIR"
        Write-Host "  To rebuild, run: .\build.ps1 dawn-clean; .\build.ps1 dawn"
        return
    }

    Assert-Command cmake "Install via: pip install cmake"
    Assert-Command ninja "Install via: pip install ninja"
    Assert-Command git   "Install Git from https://git-scm.com"

    if (-not (Test-Path (Join-Path $DAWN_DIR ".git"))) {
        Write-Host "  [ERROR] Dawn not cloned. Run: .\build.ps1 clone" -ForegroundColor Red
        exit 1
    }

    # Configure
    Write-Step "Configuring Dawn"
    $cmakeArgs = @(
        "-S", $DAWN_DIR,
        "-B", $DAWN_BUILD,
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
    )
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "  [ERROR] Dawn CMake configure failed" -ForegroundColor Red; exit 1 }
    Write-OK "Dawn configured"

    # Build
    Write-Step "Building Dawn (this may take 10-30 minutes)"
    & cmake --build $DAWN_BUILD --target webgpu_dawn --target dxcompiler
    if ($LASTEXITCODE -ne 0) { Write-Host "  [ERROR] Dawn build failed" -ForegroundColor Red; exit 1 }

    & cmake --build $DAWN_BUILD --target third_party/copy_dxil_dll
    if ($LASTEXITCODE -ne 0) { Write-Host "  [ERROR] dxil.dll copy failed" -ForegroundColor Red; exit 1 }
    Write-OK "Dawn built"

    # Deploy
    Invoke-DawnDeploy
}

function Invoke-DawnDeploy {
    Write-Step "Deploying Dawn DLLs"

    if (-not (Test-Path $DAWN_DLL_DIR)) {
        New-Item -ItemType Directory -Path $DAWN_DLL_DIR -Force | Out-Null
    }

    foreach ($dll in @("webgpu_dawn.dll", "dxcompiler.dll", "dxil.dll")) {
        $src = Join-Path $DAWN_BUILD $dll
        if (-not (Test-Path $src)) {
            Write-Host "  [ERROR] $dll not found in $DAWN_BUILD. Build Dawn first." -ForegroundColor Red
            exit 1
        }
        Copy-Item $src (Join-Path $DAWN_DLL_DIR $dll) -Force
        Write-OK "Deployed $dll"
    }
}

# ===========================================================================
# Target: install — lightweight .pth-based triton registration
# ===========================================================================

function Invoke-Install {
    Write-Step "Installing triton on Python path (lightweight)"

    # Create .pth file to add triton to sys.path
    $siteDir = python -c "import site; print(site.getusersitepackages())"
    if (-not (Test-Path $siteDir)) { New-Item -ItemType Directory -Path $siteDir -Force | Out-Null }

    $pthContent = @"
import os; os.environ.setdefault('TRITON_BACKENDS_IN_TREE', '1')
$($TRITON_DIR -replace '\\', '/')/python
"@
    $pthPath = Join-Path $siteDir "triton-backpack.pth"
    Set-Content -Path $pthPath -Value $pthContent -Encoding UTF8
    Write-OK "Wrote $pthPath"

    # Verify
    $env:TRITON_BACKENDS_IN_TREE = "1"
    python -c "import triton; print('triton loaded from:', triton.__file__)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] triton import check failed" -ForegroundColor Yellow
    } else {
        python -c "from triton.backends import backends; print('Backends:', list(backends.keys()))"
        Write-OK "triton registered with webgpu backend"
    }
}

# ===========================================================================
# Target: triton — build and install triton-windows
# ===========================================================================

function Invoke-Triton {
    Invoke-MSVC

    Assert-Command cmake "Install via: pip install cmake"
    Assert-Command ninja "Install via: pip install ninja"

    if (-not (Test-Path (Join-Path $TRITON_DIR ".git"))) {
        Write-Host "  [ERROR] triton-windows not cloned. Run: .\build.ps1 clone" -ForegroundColor Red
        exit 1
    }

    Write-Step "Building triton-windows (this may take 15-45 minutes on first build)"

    $env:TRITON_BUILD_PROTON = "0"
    $env:TRITON_BUILD_UT = "0"
    $env:TRITON_BUILD_BINARY = "0"

    $pipInstallOK = $false
    Push-Location $TRITON_DIR
    try {
        pip install --no-build-isolation --verbose -e .
        if ($LASTEXITCODE -eq 0) { $pipInstallOK = $true }
    } catch {}
    finally {
        Pop-Location
    }

    if (-not $pipInstallOK) {
        Write-Host "  [WARN] pip editable install failed, falling back to .pth install" -ForegroundColor Yellow
        Invoke-Install
        return
    }

    # Verify import
    $env:TRITON_BACKENDS_IN_TREE = "1"
    python -c "import triton; print('triton loaded from:', triton.__file__)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] triton import failed after install" -ForegroundColor Yellow
    } else {
        Write-OK "triton-windows installed"
    }
}

# ===========================================================================
# Target: verify — run GPT-2 pipeline verification
# ===========================================================================

function Invoke-Verify {
    Write-Step "Running GPT-2 verification"
    $env:TRITON_BACKENDS_IN_TREE = "1"
    python (Join-Path $MODELS_DIR "gpt-2\model.py") --verify
    if ($LASTEXITCODE -eq 0) {
        Write-OK "Verification passed"
    } else {
        Write-Host "  [FAIL] Verification failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

# ===========================================================================
# Target: setup — full provisioning from scratch
# ===========================================================================

function Invoke-Setup {
    Write-Step "Backpack full setup"
    Write-Host "  Python: $(python --version 2>&1)"
    Write-Host "  Root:   $ROOT_DIR"
    Write-Host ""

    Invoke-Deps
    Invoke-Clone
    Invoke-Dawn
    Invoke-Triton

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  Backpack setup complete!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Verify:  .\build.ps1 verify"
    Write-Host "  Run:     python models\gpt-2\model.py --prompt `"The future of AI is`""
    Write-Host ""
}

# ===========================================================================
# Target: clean
# ===========================================================================

function Invoke-DawnClean {
    Write-Step "Cleaning Dawn build"
    if (Test-Path $DAWN_BUILD) { Remove-Item $DAWN_BUILD -Recurse -Force }
    if (Test-Path $DAWN_DLL_DIR) { Remove-Item $DAWN_DLL_DIR -Recurse -Force }
    Write-OK "Dawn build cleaned"
}

function Invoke-TritonClean {
    Write-Step "Cleaning triton build"
    $tritonBuild = Join-Path $TRITON_DIR "build"
    if (Test-Path $tritonBuild) { Remove-Item $tritonBuild -Recurse -Force }
    $libtriton = Join-Path $TRITON_DIR "python\triton\_C\libtriton.pyd"
    if (Test-Path $libtriton) { Remove-Item $libtriton -Force }
    Write-OK "Triton build cleaned"
}

function Invoke-Clean {
    Invoke-DawnClean
    Invoke-TritonClean
}

# ===========================================================================
# Target: info
# ===========================================================================

function Invoke-Info {
    Write-Host ""
    Write-Host "Backpack Build Info" -ForegroundColor Cyan
    Write-Host "==================="
    Write-Host "  ROOT_DIR:     $ROOT_DIR"
    Write-Host "  DAWN_DIR:     $DAWN_DIR"
    Write-Host "  TRITON_DIR:   $TRITON_DIR"
    Write-Host "  DAWN_DLL_DIR: $DAWN_DLL_DIR"
    Write-Host ""

    $dawnDll = Join-Path $DAWN_DLL_DIR "webgpu_dawn.dll"
    $libtriton = Join-Path $TRITON_DIR "python\triton\_C\libtriton.pyd"

    Write-Host "  Dawn DLL:     $(if (Test-Path $dawnDll) { 'BUILT' } else { 'NOT BUILT' })"
    Write-Host "  libtriton:    $(if (Test-Path $libtriton) { 'BUILT' } else { 'NOT BUILT' })"
    Write-Host "  MSVC (cl):    $(if (Get-Command cl -ErrorAction SilentlyContinue) { 'Available' } else { 'Not loaded' })"
    Write-Host ""

    python --version 2>$null
    cmake --version 2>$null | Select-Object -First 1
}

# ===========================================================================
# Target: help
# ===========================================================================

function Invoke-Help {
    Write-Host ""
    Write-Host "Backpack build script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\build.ps1 [target]"
    Write-Host ""
    Write-Host "Targets:"
    Write-Host "  setup         Full provision: deps + clone + dawn + triton (default)"
    Write-Host "  deps          Install Python dependencies"
    Write-Host "  clone         Clone Dawn and triton-windows repos"
    Write-Host "  msvc          Load MSVC compiler environment"
    Write-Host "  dawn          Build Dawn WebGPU runtime from source"
    Write-Host "  dawn-deploy   Deploy Dawn DLLs (after manual Dawn build)"
    Write-Host "  triton        Build triton-windows from source (pip editable)"
    Write-Host "  install       Register triton on Python path (lightweight, no build)"
    Write-Host "  verify        Run GPT-2 pipeline verification"
    Write-Host "  clean         Remove all build artifacts"
    Write-Host "  dawn-clean    Remove Dawn build artifacts only"
    Write-Host "  triton-clean  Remove triton build artifacts only"
    Write-Host "  info          Show build configuration and status"
    Write-Host "  help          Show this message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\build.ps1                # Full setup from scratch"
    Write-Host "  .\build.ps1 dawn           # Rebuild Dawn only"
    Write-Host "  .\build.ps1 triton         # Rebuild triton only"
    Write-Host "  .\build.ps1 verify         # Verify the pipeline works"
    Write-Host ""
}

# ===========================================================================
# Dispatch
# ===========================================================================

switch ($Target) {
    "setup"        { Invoke-Setup }
    "deps"         { Invoke-Deps }
    "clone"        { Invoke-Clone }
    "msvc"         { Invoke-MSVC }
    "dawn"         { Invoke-Dawn }
    "dawn-deploy"  { Invoke-DawnDeploy }
    "triton"       { Invoke-Triton }
    "install"      { Invoke-Install }
    "verify"       { Invoke-Verify }
    "clean"        { Invoke-Clean }
    "dawn-clean"   { Invoke-DawnClean }
    "triton-clean" { Invoke-TritonClean }
    "info"         { Invoke-Info }
    "help"         { Invoke-Help }
    default        { Invoke-Help }
}
