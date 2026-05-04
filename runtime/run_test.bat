@echo off
cd /d E:\workspace\project\test\backpack\runtime\build
.\backpack_runtime.exe --model E:\workspace\project\test\ai-models\gemma-4-E4B\webgpu --prompt Hello --max-tokens 5
echo EXIT=%ERRORLEVEL%
