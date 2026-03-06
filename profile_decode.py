"""Profile one decode token to find Python overhead."""
import sys, os, time
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, 'models'))
os.chdir(os.path.join(_root, 'models'))
sys.path.insert(0, os.path.join(_root, 'third_party', 'triton-windows'))

import numpy as np

_SCRIPT_DIR = os.path.abspath('qwen-3.5')
sys.path.insert(0, _SCRIPT_DIR)

code = open(os.path.join(_SCRIPT_DIR, 'model.py')).read()
main_idx = code.find('\ndef main')
code = f'__file__ = r"{os.path.join(_SCRIPT_DIR, "model.py")}"\n' + code[:main_idx]
exec(compile(code, os.path.join(_SCRIPT_DIR, 'model.py'), 'exec'))

gguf_path = os.path.join('..', 'gitignore', 'models', 'qwen-3.5', 'weights',
                         'Qwen3.5-27B-Q4_K_M.gguf')
weights = load_gguf_qwen35_runtime_raw(gguf_path)
cfg = {k:v for k,v in QWEN35_CONFIG.items() if k != 'hf_repo'}
model = Qwen35WebGPU(weights, **cfg, quantized=True, norm_add_one=False,
                     max_seq_len=64)

# Warmup
model.forward(np.array([1], dtype=np.int32), use_cache=True, pos_offset=0)
# Reset
model.kv_cache = None
for layer in model._gpu_kv_cache:
    k, v, _ = model._gpu_kv_cache[layer]
    model._gpu_kv_cache[layer] = (k, v, 0)
runner = model.cache.runner
for layer in list(model._ssm_gpu_states.keys()):
    runner.write_buffer(model._ssm_gpu_states[layer].handle,
                       bytes(48*128*128*4))
    runner.write_buffer(model._ssm_gpu_conv_states[layer].handle,
                       bytes(3*10240*4))

print("=== Profiling 1 decode token (sorted by tottime) ===")

# Add batch counter
_orig_rwp = runner._run_with_pipeline.__func__
_batch_ct = [0, 0]
def _counted_rwp(self, *args, **kwargs):
    batching = self.is_batching and (args[8] if len(args) > 8 else kwargs.get('gpu_outputs'))
    if batching:
        _batch_ct[0] += 1
    else:
        _batch_ct[1] += 1
    return _orig_rwp(self, *args, **kwargs)
import types
runner._run_with_pipeline = types.MethodType(_counted_rwp, runner)

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
model.forward(np.array([42], dtype=np.int32), use_cache=True, pos_offset=1)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(40)
print(s.getvalue())

# Also print cumulative
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('cumulative')
ps2.print_stats(20)
print("=== Top 20 by cumulative time ===")
print(s2.getvalue())
print(f"Batch count: batched={_batch_ct[0]}, non-batched={_batch_ct[1]}")
