@echo off
REM Update Dawn to latest and initialize submodules
cd /d E:\workspace\project\agents\backpack\third_party\dawn
echo Pulling latest Dawn...
git pull origin main
echo.
echo Initializing submodules (skipping 'build' which conflicts)...
git submodule update --init third_party/abseil-cpp
git submodule update --init third_party/spirv-headers/src
git submodule update --init third_party/spirv-tools/src
git submodule update --init third_party/vulkan-headers/src
git submodule update --init third_party/vulkan-loader/src
git submodule update --init third_party/vulkan-utility-libraries/src
git submodule update --init third_party/glslang/src
git submodule update --init third_party/dxc
git submodule update --init third_party/dxheaders/src
echo.
echo Dawn updated.
git log --oneline -1
