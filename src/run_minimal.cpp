#include <iostream>
#include <cmath>

#include "model/mini_AI.h"
#include "world/world_model.h"
#include "tensor4d.h"

static float l2norm(const Tensor4D& t){
    double s = 0.0;
    for(float v : t.data) s += (double)v * (double)v;
    return (float)std::sqrt(s);
}

int main(){
    std::cout << "[mini_llm] minimal run\n";

    // ---- input ----
    Tensor4D x(1, 4, 1, 64);
    for(auto& v : x.data) v = 0.1f;

    // ---- transformer ----
    MiniAI ai(/*layers=*/2, /*dim=*/64);
    Tensor4D h = ai.forward(x);

    std::cout << "ai_out_norm=" << l2norm(h) << "\n";

    // ---- world model (ZIP準拠) ----
    WorldModel world(x.B, x.T, x.H, x.D);
    world.init();

    // 観測を注入して1ステップ遷移
    world.inject_observation(h);
    world.step_forward();

    // 現在の潜在状態（world_state.h: current.latent）を確認
    std::cout << "world_latent_norm=" << l2norm(world.current.latent) << "\n";

    // 学習を回すなら lr を渡す（scoreではない）
    // world.update(0.001f);

    std::cout << "done.\n";
    return 0;
}