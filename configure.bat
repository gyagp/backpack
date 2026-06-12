@echo off
set "CONDA_SHLVL="
set "CONDA_DEFAULT_ENV="
set "CONDA_PROMPT_MODIFIER="
set "CONDA_EXE="
set "CONDA_PREFIX="
set "_CE_CONDA="
set "_CE_M="
cmd /K ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 && cd /d E:\workspace\project\backpack && cmake -B build -G Ninja && exit"
