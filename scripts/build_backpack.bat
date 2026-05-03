@echo off
REM Build backpack runtime (backpack.dll, backpack_llm.exe, etc.)
REM Requires Dawn to be built first (see build_dawn.bat)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d E:\workspace\project\agents\backpack\runtime

echo Building backpack...
cmake --build build --config Release 2>&1 | findstr /C:"error" /C:"Linking" /C:"failed"

echo.
if exist build\Release\backpack.dll (
    copy /Y build\Release\backpack.dll build\backpack.dll >nul
)

REM Copy Dawn DLL from _build
if exist ..\third_party\dawn\_build\webgpu_dawn.dll (
    copy /Y ..\third_party\dawn\_build\webgpu_dawn.dll build\webgpu_dawn.dll >nul
)

REM Copy d3dcompiler if missing
if not exist build\d3dcompiler_47.dll (
    copy /Y C:\Windows\System32\d3dcompiler_47.dll build\d3dcompiler_47.dll >nul 2>&1
)

echo SUCCESS: backpack built
