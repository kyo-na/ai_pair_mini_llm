#include "decode/repetition.h"
#include <cmath>

void apply_repetition_penalty(
    std::vector<float>& logits,
    const std::vector<unsigned>& history,
    float penalty
) {
    if (penalty <= 1.0f) return;

    for (unsigned tok : history) {
        if (tok >= logits.size()) continue;

        float& l = logits[tok];
        if (l > 0.0f)
            l /= penalty;
        else
            l *= penalty;
    }
}