@echo off
REM Quick test: build backpack and verify it works with a small model
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cd /d E:\workspace\project\agents\backpack\runtime

echo Building...
cmake --build build --config Release 2>&1 | findstr /C:"error" /C:"Linking" /C:"failed"
if exist build\Release\backpack.dll copy /Y build\Release\backpack.dll build\backpack.dll >nul

echo.
echo Testing with TinyLlama...
build\backpack_llm.exe --model E:\workspace\project\agents\ai-models\TinyLlama-1.1B-Chat-v1.0 --benchmark --bench-prompt-len 128

echo.
echo Testing correctness...
build\backpack_llm.exe --model E:\workspace\project\agents\ai-models\TinyLlama-1.1B-Chat-v1.0 --chat "What is 2+2?" --max-tokens 20
