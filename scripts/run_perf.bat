@echo off
REM Run performance benchmarks on all models using all backends
REM Results saved to apps/perf/*.json
cd /d E:\workspace\project\agents\backpack
python apps\perf\run_perf.py %*
