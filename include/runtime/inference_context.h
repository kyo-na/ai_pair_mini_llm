#pragma once
#include "memory/memory_arena.h"

class InferenceContext {
public:
    InferenceContext(size_t bytes)
        : arena_(bytes) {}

    void reset()
    {
        arena_.reset();
    }

    MemoryArena& arena()
    {
        return arena_;
    }

private:
    MemoryArena arena_;
};