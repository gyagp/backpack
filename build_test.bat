@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >/dev/null 2>&1
cd /d E:\workspace\project\backpack
cl.exe /std:c++20 /EHsc /I src /Fe:test_onnx_loader.exe tests\test_onnx_loader.cpp
if %ERRORLEVEL% EQU 0 test_onnx_loader.exe
