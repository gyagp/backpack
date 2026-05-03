@echo off
REM Build Dawn as a shared library (webgpu_dawn.dll)
REM Outputs to third_party/dawn/_build/
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d E:\workspace\project\agents\backpack\third_party\dawn

echo Configuring Dawn...
cmake -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DDAWN_BUILD_SAMPLES=OFF ^
    -DDAWN_ENABLE_INSTALL=OFF ^
    -DTINT_BUILD_TESTS=OFF ^
    -DTINT_BUILD_CMD_TOOLS=OFF ^
    -DDAWN_BUILD_TESTS=OFF ^
    -DDAWN_USE_GLFW=OFF ^
    -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED 2>&1

echo.
echo Building webgpu_dawn.dll...
cmake --build _build --target webgpu_dawn -j 16 2>&1

echo.
if exist _build\webgpu_dawn.dll (
    echo SUCCESS: _build\webgpu_dawn.dll built
    echo Copying to expected locations...

    REM Copy to dawn/build/ (where backpack CMakeLists expects the .lib)
    if not exist build\src\dawn\native mkdir build\src\dawn\native
    if not exist build\gen\include\dawn mkdir build\gen\include\dawn
    copy /Y _build\webgpu_dawn.dll build\webgpu_dawn.dll
    copy /Y _build\src\dawn\native\webgpu_dawn.lib build\src\dawn\native\webgpu_dawn.lib
    xcopy /Y /S _build\gen\include\dawn\* build\gen\include\dawn\ >nul 2>&1

    REM Copy to triton-windows path (where backpack CMakeLists copies DLL from)
    copy /Y _build\webgpu_dawn.dll ..\triton-windows\third_party\webgpu\dawn\build\webgpu_dawn.dll

    echo Done.
) else (
    echo FAILED: webgpu_dawn.dll not found in _build\
)
