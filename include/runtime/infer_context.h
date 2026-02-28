#pragma once
#include "memory/memory_arena.h"

struct InferContext {
    MemoryArena arena;

    explicit InferContext(size_t bytes)
        : arena(bytes) {}

    void reset_step() {
        arena.reset();
    }
};