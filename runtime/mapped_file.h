#pragma once
/**
 * mapped_file.h -- Cross-platform memory-mapped file (read-only).
 *
 * Uses CreateFileMapping/MapViewOfFile on Windows, mmap on POSIX.
 * RAII: unmaps automatically on destruction. Move-only.
 */

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

struct MappedFile {
    const uint8_t* data = nullptr;
    size_t size = 0;

    MappedFile() = default;
    ~MappedFile() { close(); }

    // Move-only
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept
        : data(o.data), size(o.size)
#ifdef _WIN32
        , hFile_(o.hFile_), hMapping_(o.hMapping_)
#else
        , fd_(o.fd_)
#endif
    {
        o.data = nullptr;
        o.size = 0;
#ifdef _WIN32
        o.hFile_ = INVALID_HANDLE_VALUE;
        o.hMapping_ = nullptr;
#else
        o.fd_ = -1;
#endif
    }
    MappedFile& operator=(MappedFile&& o) noexcept {
        if (this != &o) {
            close();
            data = o.data; size = o.size;
#ifdef _WIN32
            hFile_ = o.hFile_; hMapping_ = o.hMapping_;
            o.hFile_ = INVALID_HANDLE_VALUE; o.hMapping_ = nullptr;
#else
            fd_ = o.fd_; o.fd_ = -1;
#endif
            o.data = nullptr; o.size = 0;
        }
        return *this;
    }

    bool open(const std::string& path) {
        close();
#ifdef _WIN32
        // Convert UTF-8 path to wide string
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wpath(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);

        hFile_ = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile_ == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile_, &fileSize)) { close(); return false; }
        size = (size_t)fileSize.QuadPart;
        if (size == 0) { close(); return false; }

        hMapping_ = CreateFileMappingW(hFile_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMapping_) { close(); return false; }

        data = (const uint8_t*)MapViewOfFile(hMapping_, FILE_MAP_READ, 0, 0, 0);
        if (!data) { close(); return false; }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;

        struct stat st;
        if (fstat(fd_, &st) != 0) { close(); return false; }
        size = (size_t)st.st_size;
        if (size == 0) { close(); return false; }

        data = (const uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data == MAP_FAILED) { data = nullptr; close(); return false; }
#endif
        return true;
    }

    void close() {
#ifdef _WIN32
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (hMapping_) { CloseHandle(hMapping_); hMapping_ = nullptr; }
        if (hFile_ != INVALID_HANDLE_VALUE) { CloseHandle(hFile_); hFile_ = INVALID_HANDLE_VALUE; }
#else
        if (data) { munmap((void*)data, size); data = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        size = 0;
    }

private:
#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
    HANDLE hMapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};
