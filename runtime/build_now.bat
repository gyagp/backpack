@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d E:\workspace\project\agents\backpack\runtime
if exist build rmdir /s /q build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release . 2>&1
cmake --build build --config Release 2>&1
