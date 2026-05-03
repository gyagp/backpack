@echo off
REM Clean rebuild backpack from scratch (reconfigure CMake + build)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d E:\workspace\project\agents\backpack\runtime

echo Cleaning build directory...
if exist build rmdir /s /q build

echo Configuring CMake...
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release . 2>&1

echo.
echo Building...
cmake --build build --config Release 2>&1

echo.
if exist build\Release\backpack.dll (
    copy /Y build\Release\backpack.dll build\backpack.dll >nul
    echo SUCCESS: Clean build complete
) else (
    echo FAILED
)
