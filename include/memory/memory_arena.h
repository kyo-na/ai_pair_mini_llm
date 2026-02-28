#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>

class MemoryArena {
public:
    explicit MemoryArena(size_t bytes)
        : buf_(bytes), head_(0) {}

    void reset() { head_ = 0; }

    void* alloc(size_t bytes, size_t align = 64) {
        size_t p = (head_ + (align - 1)) & ~(align - 1);
        if (p + bytes > buf_.size()) throw std::bad_alloc();
        void* out = buf_.data() + p;
        head_ = p + bytes;
        return out;
    }

    size_t capacity() const { return buf_.size(); }
    size_t used() const { return head_; }

private:
    std::vector<uint8_t> buf_;
    size_t head_;
};