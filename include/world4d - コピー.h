#pragma once
#include <vector>
#include <cmath>

struct World4D {
    size_t hidden;
    std::vector<float> state;

    World4D(size_t h)
        : hidden(h), state(h, 0.0f) {}

    void reset_energy() {
        for (auto& x : state) x = 0.0f;
    }

    void step(int token) {
        for (size_t i = 0; i < hidden; ++i) {
            state[i] = std::tanh(
                state[i] * 0.7f + 0.1f * token
            );
        }
    }

    const std::vector<float>& get_state() const {
        return state;
    }

    size_t hidden_size() const {
        return hidden;
    }
};