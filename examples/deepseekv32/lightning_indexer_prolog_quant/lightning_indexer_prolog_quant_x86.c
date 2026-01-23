/**
 * x86 demo for lightning_indexer_prolog_quant (simplified).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define T 4
#define Q_LORA_RANK 4
#define H 4
#define HEAD_NUM 2
#define HEAD_DIM 2
#define OUT_DIM (HEAD_NUM * HEAD_DIM)

void lightning_indexer_prolog_quant_x86(
    const float* q_norm,
    const float* q_norm_scale,
    const float* w_qb,
    const float* w_qb_scale,
    const float* x,
    const float* w_proj,
    float* q_out,
    float* weights_out
) {
    float q_matmul[T][OUT_DIM] = {0};
    float q_scaled_row[T][OUT_DIM] = {0};
    float weights_raw[T][HEAD_NUM] = {0};
    float weight_scale = 1.0f / sqrtf((float)(HEAD_NUM * HEAD_DIM));

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < OUT_DIM; j++) {
            float sum = 0.0f;
            for (int k = 0; k < Q_LORA_RANK; k++) {
                sum += q_norm[i * Q_LORA_RANK + k] * w_qb[k * OUT_DIM + j];
            }
            q_matmul[i][j] = sum;
        }
    }

    for (int i = 0; i < T; i++) {
        float scale_row = q_norm_scale[i];
        for (int j = 0; j < OUT_DIM; j++) {
            q_scaled_row[i][j] = q_matmul[i][j] * scale_row;
        }
    }

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < OUT_DIM; j++) {
            q_out[i * OUT_DIM + j] = q_scaled_row[i][j] * w_qb_scale[j];
        }
    }

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < HEAD_NUM; j++) {
            float sum = 0.0f;
            for (int k = 0; k < H; k++) {
                sum += x[i * H + k] * w_proj[k * HEAD_NUM + j];
            }
            weights_raw[i][j] = sum;
        }
    }

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < HEAD_NUM; j++) {
            weights_out[i * HEAD_NUM + j] = weights_raw[i][j] * weight_scale;
        }
    }
}

int main(void) {
    float q_norm[T * Q_LORA_RANK];
    float q_norm_scale[T];
    float w_qb[Q_LORA_RANK * OUT_DIM];
    float w_qb_scale[OUT_DIM];
    float x[T * H];
    float w_proj[H * HEAD_NUM];
    float q_out[T * OUT_DIM];
    float weights_out[T * HEAD_NUM];

    for (int i = 0; i < T * Q_LORA_RANK; i++) q_norm[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < T; i++) q_norm_scale[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < Q_LORA_RANK * OUT_DIM; i++) w_qb[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < OUT_DIM; i++) w_qb_scale[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < T * H; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < H * HEAD_NUM; i++) w_proj[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    lightning_indexer_prolog_quant_x86(
        q_norm, q_norm_scale, w_qb, w_qb_scale, x, w_proj, q_out, weights_out
    );

    printf("q_out[0..3]: ");
    for (int i = 0; i < OUT_DIM; i++) printf("%f ", q_out[i]);
    printf("\nweights_out[0..1]: %f %f\n", weights_out[0], weights_out[1]);
    return 0;
}
