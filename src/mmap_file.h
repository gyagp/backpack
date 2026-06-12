#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

class MmapFile {
public:
    MmapFile() = default;

    explicit MmapFile(const std::string& path) {
        open(path);
    }

    ~MmapFile() {
        close();
    }

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    MmapFile(MmapFile&& other) noexcept
        : data_(other.data_), size_(other.size_)
#ifdef _WIN32
        , file_handle_(other.file_handle_), mapping_handle_(other.mapping_handle_)
#else
        , fd_(other.fd_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#ifdef _WIN32
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.mapping_handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
    }

    MmapFile& operator=(MmapFile&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
#ifdef _WIN32
            file_handle_ = other.file_handle_;
            mapping_handle_ = other.mapping_handle_;
            other.file_handle_ = INVALID_HANDLE_VALUE;
            other.mapping_handle_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void open(const std::string& path) {
        close();
#ifdef _WIN32
        file_handle_ = CreateFileA(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("MmapFile: cannot open " + path);

        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(file_handle_, &file_size)) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("MmapFile: cannot get size of " + path);
        }
        size_ = static_cast<size_t>(file_size.QuadPart);

        if (size_ == 0) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return;
        }

        mapping_handle_ = CreateFileMappingA(
            file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_handle_) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("MmapFile: cannot create mapping for " + path);
        }

        data_ = static_cast<const uint8_t*>(
            MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
        if (!data_) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("MmapFile: cannot map view of " + path);
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("MmapFile: cannot open " + path);

        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("MmapFile: cannot stat " + path);
        }
        size_ = static_cast<size_t>(st.st_size);

        if (size_ == 0) {
            ::close(fd_);
            fd_ = -1;
            return;
        }

        void* ptr = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (ptr == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("MmapFile: cannot mmap " + path);
        }
        data_ = static_cast<const uint8_t*>(ptr);
#endif
    }

    void close() {
#ifdef _WIN32
        if (data_) { UnmapViewOfFile(data_); data_ = nullptr; }
        if (mapping_handle_) { CloseHandle(mapping_handle_); mapping_handle_ = nullptr; }
        if (file_handle_ != INVALID_HANDLE_VALUE) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE; }
#else
        if (data_ && size_ > 0) { munmap(const_cast<uint8_t*>(data_), size_); data_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        size_ = 0;
    }

    bool is_open() const { return data_ != nullptr; }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};
