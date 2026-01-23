/**
 * x86 demo for mla_prolog_quant (simplified).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define T 4
#define H 4
#define OUT 4

void mla_prolog_quant_x86(const float* x, const float* w_dq, const float* w_uq_qr, float* q_out) {
    float x_proj[T][H] = {0};
    float x_norm[T][H] = {0};
    float eps = 1e-6f;

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < H; j++) {
            float sum = 0.0f;
            for (int k = 0; k < H; k++) {
                sum += x[i * H + k] * w_dq[k * H + j];
            }
            x_proj[i][j] = sum;
        }
    }

    for (int i = 0; i < T; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < H; j++) {
            sum_sq += x_proj[i][j] * x_proj[i][j];
        }
        float mean_sq = sum_sq / (float)H;
        float rms = sqrtf(mean_sq + eps);
        for (int j = 0; j < H; j++) {
            x_norm[i][j] = x_proj[i][j] / rms;
        }
    }

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < OUT; j++) {
            float sum = 0.0f;
            for (int k = 0; k < H; k++) {
                sum += x_norm[i][k] * w_uq_qr[k * OUT + j];
            }
            q_out[i * OUT + j] = sum;
        }
    }
}

int main(void) {
    float x[T * H];
    float w_dq[H * H];
    float w_uq_qr[H * OUT];
    float q_out[T * OUT];

    for (int i = 0; i < T * H; i++) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < H * H; i++) w_dq[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < H * OUT; i++) w_uq_qr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    mla_prolog_quant_x86(x, w_dq, w_uq_qr, q_out);

    printf("q_out[0..3]: ");
    for (int i = 0; i < OUT; i++) printf("%f ", q_out[i]);
    printf("\n");
    return 0;
}
