#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../src/mmap_file.h"
#include "../src/gguf_parser.h"

#ifdef _WIN32
#include <windows.h>
#include <cstdlib>
#endif

static std::string temp_path(const std::string& name) {
#ifdef _WIN32
    char buf[MAX_PATH];
    GetTempPathA(MAX_PATH, buf);
    return std::string(buf) + name;
#else
    return "/tmp/" + name;
#endif
}

static void write_file(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), size);
}

void test_mmap_basic() {
    std::string path = temp_path("test_mmap_basic.bin");
    std::vector<uint8_t> content = {0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD};
    write_file(path, content.data(), content.size());

    MmapFile mmap(path);
    assert(mmap.is_open());
    assert(mmap.size() == 8);
    assert(mmap.data()[0] == 0x01);
    assert(mmap.data()[4] == 0xAA);
    assert(mmap.data()[7] == 0xDD);

    mmap.close();
    assert(!mmap.is_open());
    assert(mmap.size() == 0);
    printf("  PASS: mmap basic open/read/close\n");
}

void test_mmap_move() {
    std::string path = temp_path("test_mmap_move.bin");
    std::vector<uint8_t> content = {0x10, 0x20, 0x30};
    write_file(path, content.data(), content.size());

    MmapFile a(path);
    assert(a.is_open());

    MmapFile b(std::move(a));
    assert(!a.is_open());
    assert(b.is_open());
    assert(b.size() == 3);
    assert(b.data()[0] == 0x10);

    MmapFile c;
    c = std::move(b);
    assert(!b.is_open());
    assert(c.is_open());
    assert(c.data()[2] == 0x30);
    printf("  PASS: mmap move semantics\n");
}

void test_mmap_nonexistent() {
    bool threw = false;
    try { MmapFile m("nonexistent_file_12345.bin"); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    printf("  PASS: mmap nonexistent file throws\n");
}

void test_gguf_parse_via_mmap() {
    std::string path = temp_path("test_mmap_gguf.bin");

    std::vector<uint8_t> buf;
    auto write_val = [&](auto val) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
        buf.insert(buf.end(), p, p + sizeof(val));
    };
    auto write_str = [&](const std::string& s) {
        uint64_t len = s.size();
        write_val(len);
        buf.insert(buf.end(), s.begin(), s.end());
    };

    write_val(uint32_t(0x46475547)); // magic
    write_val(uint32_t(3));           // version
    write_val(uint64_t(1));           // tensor_count
    write_val(uint64_t(1));           // metadata_kv_count

    // one kv: "arch" = "llama"
    write_str("arch");
    write_val(uint32_t(8)); // STRING type
    write_str("llama");

    // one tensor
    write_str("weight");
    write_val(uint32_t(1));  // ndims
    write_val(uint64_t(256)); // dim[0]
    write_val(uint32_t(0));  // F32
    write_val(uint64_t(0));  // offset

    write_file(path, buf.data(), buf.size());

    auto file = GGUFParser::parse(path);
    assert(file.version == 3);
    assert(file.tensor_count == 1);
    assert(std::get<std::string>(file.metadata["arch"]) == "llama");
    assert(file.tensors[0].name == "weight");
    assert(file.tensors[0].dimensions[0] == 256);
    assert(file.data_offset % 32 == 0);
    printf("  PASS: GGUF parse via mmap file path\n");
}

int main() {
    printf("test_mmap_file:\n");
    test_mmap_basic();
    test_mmap_move();
    test_mmap_nonexistent();
    test_gguf_parse_via_mmap();
    printf("All mmap tests passed.\n");
    return 0;
}
