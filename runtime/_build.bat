@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d E:\workspace\project\test\backpack\runtime
cmake --build build --config Release 2>&1 > E:\workspace\project\test\backpack\runtime\_build_output.txt
