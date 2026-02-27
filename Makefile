# Backpack — Build & Provision Makefile
#
# Builds Dawn (WebGPU runtime) and triton-windows (Triton compiler with WebGPU
# backend) from source, then installs triton as an editable Python package.
#
# Prerequisites:
#   - Python 3.13+
#   - MSVC v143+ (Visual Studio 2022 Build Tools or Community)
#   - CMake 3.20+, Ninja
#   - Git
#
# Quick start (from VS Developer PowerShell or after running vcvars64.bat):
#   make setup          # Full provision: deps + dawn + triton + install
#   make dawn           # Build Dawn only
#   make triton         # Build triton-windows only
#   make install        # pip editable install of triton
#   make verify         # Run GPT-2 verification
#
# On Windows without GNU Make, use the PowerShell equivalents shown in each
# target's recipe, or run: .\build.ps1 <target>

# ===========================================================================
# Configuration
# ===========================================================================

PYTHON         ?= python
PIP            ?= pip
CMAKE          ?= cmake
GIT            ?= git

# Directories (relative to this Makefile)
ROOT_DIR       := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
THIRD_PARTY    := $(ROOT_DIR)/third_party
TRITON_DIR     := $(THIRD_PARTY)/triton-windows
DAWN_DIR       := $(THIRD_PARTY)/dawn
DAWN_BUILD_DIR := $(DAWN_DIR)/build
MODELS_DIR     := $(ROOT_DIR)/models

# Dawn build output — dawn_runner.py searches for the DLL at:
#   <triton_root>/third_party/webgpu/dawn/build/webgpu_dawn.dll
DAWN_DLL_DIR   := $(TRITON_DIR)/third_party/webgpu/dawn/build
DAWN_DLL       := $(DAWN_DLL_DIR)/webgpu_dawn.dll

# Triton build output
TRITON_C_DIR   := $(TRITON_DIR)/python/triton/_C
LIBTRITON      := $(TRITON_C_DIR)/libtriton.pyd

# Git repos
DAWN_REPO      := https://dawn.googlesource.com/dawn
TRITON_REPO    := https://github.com/gyagp/triton-windows.git
TRITON_BRANCH  := webgpu

# ===========================================================================
# Dawn CMake flags  (matches the proven archive build configuration)
# ===========================================================================

DAWN_CMAKE_FLAGS := \
	-G Ninja \
	-DCMAKE_BUILD_TYPE=Release \
	-DDAWN_FETCH_DEPENDENCIES=ON \
	-DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED \
	-DBUILD_SHARED_LIBS=OFF \
	-DDAWN_USE_BUILT_DXC=ON \
	-DDAWN_ENABLE_D3D12=ON \
	-DDAWN_ENABLE_D3D11=ON \
	-DDAWN_ENABLE_VULKAN=ON \
	-DDAWN_ENABLE_NULL=ON \
	-DDAWN_ENABLE_INSTALL=ON \
	-DDAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON \
	-DDAWN_ENABLE_DESKTOP_GL=OFF \
	-DDAWN_ENABLE_OPENGLES=OFF \
	-DDAWN_ENABLE_METAL=OFF \
	-DDAWN_ENABLE_SWIFTSHADER=OFF \
	-DDAWN_BUILD_SAMPLES=OFF \
	-DDAWN_BUILD_TESTS=OFF \
	-DDAWN_BUILD_BENCHMARKS=OFF \
	-DDAWN_BUILD_NODE_BINDINGS=OFF \
	-DDAWN_BUILD_PROTOBUF=OFF \
	-DDAWN_USE_GLFW=OFF \
	-DDAWN_WERROR=OFF \
	-DTINT_BUILD_CMD_TOOLS=OFF \
	-DTINT_BUILD_TESTS=OFF \
	-DTINT_BUILD_BENCHMARKS=OFF \
	-DTINT_BUILD_IR_BINARY=OFF \
	-DTINT_BUILD_TINTD=OFF \
	-DTINT_ENABLE_INSTALL=OFF

# ===========================================================================
# Triton environment variables
# ===========================================================================

export TRITON_BUILD_PROTON  := 0
export TRITON_BUILD_UT      := 0
export TRITON_BUILD_BINARY  := 0

# ===========================================================================
# Shell detection — use cmd.exe on Windows for reliable mkdir/copy
# ===========================================================================

ifeq ($(OS),Windows_NT)
  SHELL  := cmd.exe
  MKDIR  = if not exist "$(subst /,\,$1)" mkdir "$(subst /,\,$1)"
  CP     = copy /Y "$(subst /,\,$1)" "$(subst /,\,$2)"
  RMDIR  = if exist "$(subst /,\,$1)" rd /s /q "$(subst /,\,$1)"
  RMDEL  = if exist "$(subst /,\,$1)" del /q "$(subst /,\,$1)"
  ECHO   = @echo $1
else
  MKDIR  = mkdir -p $1
  CP     = cp -f $1 $2
  RMDIR  = rm -rf $1
  RMDEL  = rm -f $1
  ECHO   = @echo '$1'
endif

# ===========================================================================
# Phony targets
# ===========================================================================

.PHONY: all setup clone dawn dawn-configure dawn-build dawn-deploy \
        triton install deps verify clean \
        dawn-clean triton-clean info help

# Default target
all: dawn triton install

# Full provisioning (from scratch)
setup: deps clone dawn triton
	$(call ECHO,)
	$(call ECHO,============================)
	$(call ECHO, Backpack setup complete!)
	$(call ECHO,============================)
	$(call ECHO,Run: $(PYTHON) models/gpt-2/model.py --verify)

# ===========================================================================
# Clone third-party repos (skip if already present)
# ===========================================================================

clone:
ifeq ($(OS),Windows_NT)
	@if not exist "$(subst /,\,$(DAWN_DIR))\.git" $(GIT) clone $(DAWN_REPO) $(DAWN_DIR)
	@if not exist "$(subst /,\,$(TRITON_DIR))\.git" $(GIT) clone -b $(TRITON_BRANCH) $(TRITON_REPO) $(TRITON_DIR)
else
	@test -d $(DAWN_DIR)/.git   || $(GIT) clone $(DAWN_REPO) $(DAWN_DIR)
	@test -d $(TRITON_DIR)/.git || $(GIT) clone -b $(TRITON_BRANCH) $(TRITON_REPO) $(TRITON_DIR)
endif

# ===========================================================================
# Dependencies
# ===========================================================================

deps:
	$(PIP) install setuptools wheel cmake ninja pybind11 lit
	$(PIP) install numpy tiktoken requests

# ===========================================================================
# Dawn (WebGPU runtime)
# ===========================================================================

# Configure Dawn build (DAWN_FETCH_DEPENDENCIES=ON auto-downloads third-party
# deps like spirv-tools, vulkan-headers, abseil, etc.)
dawn-configure:
	$(call ECHO,--- Configuring Dawn ---)
	$(CMAKE) -S $(DAWN_DIR) -B $(DAWN_BUILD_DIR) $(DAWN_CMAKE_FLAGS)

# Build Dawn monolithic shared library + DXC compiler
dawn-build: dawn-configure
	$(call ECHO,--- Building Dawn (this may take 10-30 minutes) ---)
	$(CMAKE) --build $(DAWN_BUILD_DIR) --target webgpu_dawn --target dxcompiler
	$(CMAKE) --build $(DAWN_BUILD_DIR) --target third_party/copy_dxil_dll

# Copy webgpu_dawn.dll (+ DXC libs) to the path expected by dawn_runner.py:
#   triton-windows/third_party/webgpu/dawn/build/
dawn-deploy: dawn-build
	$(call ECHO,--- Deploying Dawn DLLs ---)
	$(call MKDIR,$(DAWN_DLL_DIR))
	$(call CP,$(DAWN_BUILD_DIR)/webgpu_dawn.dll,$(DAWN_DLL_DIR)/)
	$(call CP,$(DAWN_BUILD_DIR)/dxcompiler.dll,$(DAWN_DLL_DIR)/)
	$(call CP,$(DAWN_BUILD_DIR)/dxil.dll,$(DAWN_DLL_DIR)/)
	$(call ECHO,Dawn deployed to $(DAWN_DLL_DIR))

# Combined target
dawn: dawn-deploy

# ===========================================================================
# Triton-Windows (Triton compiler with WebGPU backend)
# ===========================================================================

# Build and install triton-windows as editable pip package.
# setup.py handles CMake configuration, LLVM download, and compilation.
# The webgpu backend (third_party/webgpu/) is compiled into libtriton.pyd.
triton:
	$(call ECHO,--- Building triton-windows (this may take 15-45 minutes) ---)
	cd $(TRITON_DIR) && $(PIP) install --no-build-isolation --verbose -e .
	$(call ECHO,--- triton-windows build complete ---)

# Lightweight install: register triton on sys.path without rebuilding.
# Use this only if libtriton.pyd already exists (e.g., copied from a prior build).
install:
	$(call ECHO,--- Installing triton on Python path ---)
	$(PYTHON) -c "import site,os; p=site.getusersitepackages(); os.makedirs(p,exist_ok=True); f=open(os.path.join(p,'triton-backpack.pth'),'w'); f.write(r'$(TRITON_DIR)/python'+'\n'); f.close(); print('Wrote',os.path.join(p,'triton-backpack.pth'))"
	$(PYTHON) -c "import triton; print('triton loaded from:', triton.__file__)"

# ===========================================================================
# Verification
# ===========================================================================

verify:
	$(PYTHON) $(MODELS_DIR)/gpt-2/model.py --verify

# ===========================================================================
# Clean
# ===========================================================================

dawn-clean:
	$(call RMDIR,$(DAWN_BUILD_DIR))
	$(call RMDIR,$(DAWN_DLL_DIR))
	$(call ECHO,Dawn build cleaned)

triton-clean:
	$(call RMDIR,$(TRITON_DIR)/build)
	$(call RMDEL,$(LIBTRITON))
	$(call ECHO,Triton build cleaned)

clean: dawn-clean triton-clean

# ===========================================================================
# Info / Help
# ===========================================================================

info:
	$(call ECHO,)
	$(call ECHO,Backpack Build Info)
	$(call ECHO,====================)
	$(call ECHO,ROOT_DIR:       $(ROOT_DIR))
	$(call ECHO,DAWN_DIR:       $(DAWN_DIR))
	$(call ECHO,TRITON_DIR:     $(TRITON_DIR))
	$(call ECHO,DAWN_DLL:       $(DAWN_DLL))
	$(call ECHO,LIBTRITON:      $(LIBTRITON))
	$(call ECHO,PYTHON:         $(PYTHON))
	$(PYTHON) --version
	$(CMAKE) --version

help:
	$(call ECHO,)
	$(call ECHO,Backpack Makefile targets:)
	$(call ECHO,)
	$(call ECHO,  make setup          Full provision: deps + dawn + triton + install)
	$(call ECHO,  make dawn           Build Dawn WebGPU runtime from source)
	$(call ECHO,  make triton         Build triton-windows from source via pip editable)
	$(call ECHO,  make install        Register triton on Python path - lightweight)
	$(call ECHO,  make deps           Install Python dependencies)
	$(call ECHO,  make verify         Run GPT-2 pipeline verification)
	$(call ECHO,  make clean          Remove all build artifacts)
	$(call ECHO,  make dawn-clean     Remove Dawn build artifacts only)
	$(call ECHO,  make triton-clean   Remove triton build artifacts only)
	$(call ECHO,  make info           Show build configuration)
	$(call ECHO,  make help           Show this message)
	$(call ECHO,)
	$(call ECHO,Prerequisites:)
	$(call ECHO,  MSVC must be on PATH. From PowerShell run:)
	$(call ECHO,    cmd /c vcvars64.bat ... to load MSVC environment)
	$(call ECHO,)
