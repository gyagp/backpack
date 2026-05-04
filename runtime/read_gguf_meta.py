import struct, sys

def read_u8(f): return struct.unpack('<B', f.read(1))[0]
def read_i8(f): return struct.unpack('<b', f.read(1))[0]
def read_u16(f): return struct.unpack('<H', f.read(2))[0]
def read_i16(f): return struct.unpack('<h', f.read(2))[0]
def read_u32(f): return struct.unpack('<I', f.read(4))[0]
def read_i32(f): return struct.unpack('<i', f.read(4))[0]
def read_f32(f): return struct.unpack('<f', f.read(4))[0]
def read_u64(f): return struct.unpack('<Q', f.read(8))[0]
def read_i64(f): return struct.unpack('<q', f.read(8))[0]
def read_f64(f): return struct.unpack('<d', f.read(8))[0]
def read_bool(f): return bool(read_u8(f))
def read_string(f):
    n = read_u64(f)
    return f.read(n).decode('utf-8', errors='replace')

def read_value(f, vtype):
    readers = {
        0: read_u8, 1: read_i8, 2: read_u16, 3: read_i16,
        4: read_u32, 5: read_i32, 6: read_f32, 7: read_bool,
        8: read_string, 10: read_u64, 11: read_i64, 12: read_f64,
    }
    if vtype == 9:  # array
        elem_type = read_u32(f)
        count = read_u64(f)
        if count > 200:
            # Skip the data
            elem_sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
            if elem_type in elem_sizes:
                f.seek(count * elem_sizes[elem_type], 1)
            elif elem_type == 8:  # string array - must read each
                for _ in range(count):
                    n = read_u64(f)
                    f.seek(n, 1)
            else:
                for _ in range(count):
                    read_value(f, elem_type)
            return f"[array of {count} elements, type={elem_type}]"
        return [read_value(f, elem_type) for _ in range(count)]
    if vtype in readers:
        return readers[vtype](f)
    return f"<unknown type {vtype}>"

files = [
    r"E:\workspace\project\agents\ai-models\granite-3.1-2b-instruct\gguf\granite-3.1-2b-instruct-Q4_K_M.gguf",
    r"E:\workspace\project\agents\ai-models\Nemotron-Mini-4B-Instruct\gguf\Nemotron-Mini-4B-Instruct-Q4_K_M.gguf",
    r"E:\workspace\project\agents\ai-models\internlm2-chat-1_8b\gguf\internlm2-chat-1_8b.Q4_K_M.gguf",
]

for path in files:
    print("=" * 100)
    print(f"FILE: {path}")
    print("=" * 100)
    with open(path, 'rb') as f:
        magic = read_u32(f)
        version = read_u32(f)
        n_tensors = read_u64(f)
        n_kv = read_u64(f)
        print(f"Magic: 0x{magic:08X}  Version: {version}  Tensors: {n_tensors}  KV pairs: {n_kv}")
        print("-" * 100)
        for i in range(n_kv):
            key = read_string(f)
            vtype = read_u32(f)
            val = read_value(f, vtype)
            # Truncate very long values
            s = repr(val)
            if len(s) > 500:
                s = s[:500] + "... [truncated]"
            print(f"  {key} = {s}")
    print()
