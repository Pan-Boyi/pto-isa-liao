/**
 * x86 demo for mla_indexer_prolog_quant (simplified).
 */

#include <stdio.h>
#include <stdlib.h>

#define T 4
#define H 4
#define Q_RANK 4
#define OUT 4
#define HEAD_NUM 2

void mla_indexer_prolog_quant_x86(const float* x, const float* w_dq,
                                  const float* w_qb, const float* w_proj,
                                  float* q_out, float* weights) {
    float q_norm[T][Q_RANK] = {0};

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < Q_RANK; j++) {
            float sum = 0.0f;
            for (int k = 0; k < H; k++) {
                sum += x[i * H + k] * w_dq[k * Q_RANK + j];
            }
            q_norm[i][j] = sum;
        }
    }

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < OUT; j++) {
            float sum = 0.0f;
            for (int k = 0; k < Q_RANK; k++) {
                sum += q_norm[i][k] * w_qb[k * OUT + j];
            }
            q_out[i * OUT + j] = sum;
        }
    }

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < HEAD_NUM; j++) {
            float sum = 0.0f;
            for (int k = 0; k < H; k++) {
                sum += x[i * H + k] * w_proj[k * HEAD_NUM + j];
            }
            weights[i * HEAD_NUM + j] = sum;
        }
    }
}

int main(void) {
    float x[T * H];
    float w_dq[H * Q_RANK];
    float w_qb[Q_RANK * OUT];
    float w_proj[H * HEAD_NUM];
    float q_out[T * OUT];
    float weights[T * HEAD_NUM];

    for (int i = 0; i < T * H; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < H * Q_RANK; i++) w_dq[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < Q_RANK * OUT; i++) w_qb[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < H * HEAD_NUM; i++) w_proj[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    mla_indexer_prolog_quant_x86(x, w_dq, w_qb, w_proj, q_out, weights);

    printf("q_out[0..3]: ");
    for (int i = 0; i < OUT; i++) printf("%f ", q_out[i]);
    printf("\nweights[0..1]: %f %f\n", weights[0], weights[1]);
    return 0;
}
