#include <cstdio>
#include "tensor4d.h"
#include "world/world_model.h"
#include "world/world_loss.h"

int main(){

    // -------------------------
    // ハイパーパラメータ
    // -------------------------
    int B = 1;   // batch
    int T = 6;   // time steps
    int H = 1;   // heads
    int D = 16;  // latent dim
    float lr = 1e-3f;

    // -------------------------
    // World 構築
    // -------------------------
    WorldModel world(B, T, H, D);
    world.init();

    WorldConsistencyLoss world_loss;

    // -------------------------
    // 観測データ（仮）
    // -------------------------
    Tensor4D obs_t(B, T, H, D);
    Tensor4D obs_t1(B, T, H, D);

    // 簡単な「世界」：全部 0 → 全部 1
    for(auto& v : obs_t.data)  v = 0.0f;
    for(auto& v : obs_t1.data) v = 1.0f;

    // -------------------------
    // 学習ループ
    // -------------------------
    for(int epoch = 0; epoch < 50; epoch++){

        // 現実を注入
        world.inject_observation(obs_t);

        // 世界を 1 step 予測
        world.step_forward();

        // World consistency loss
        float loss = world_loss.forward(
            world.current.latent,
            obs_t1
        );

        // backward
        world.backward(world_loss.backward());
        world.update(lr);

        printf("epoch %02d | world_loss = %.6f\n", epoch, loss);
    }

    return 0;
}