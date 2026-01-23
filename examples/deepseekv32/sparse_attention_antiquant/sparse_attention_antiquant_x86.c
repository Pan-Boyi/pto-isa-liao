/**
 * x86 demo for sparse_attention_antiquant (simplified attention core).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TQ 4
#define TK 4
#define D 4

void sparse_attention_antiquant_x86(const float* q, const float* k, const float* v, float* out) {
    float k_t[D][TK] = {0};
    float scores[TQ][TK] = {0};
    float shifted[TQ][TK] = {0};
    float exp_scores[TQ][TK] = {0};
    float probs[TQ][TK] = {0};

    for (int i = 0; i < TK; i++) {
        for (int j = 0; j < D; j++) {
            k_t[j][i] = k[i * D + j];
        }
    }

    for (int i = 0; i < TQ; i++) {
        for (int j = 0; j < TK; j++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += q[i * D + d] * k_t[d][j];
            }
            scores[i][j] = sum;
        }
    }

    for (int i = 0; i < TQ; i++) {
        float maxv = scores[i][0];
        for (int j = 1; j < TK; j++) {
            if (scores[i][j] > maxv) maxv = scores[i][j];
        }
        float sum = 0.0f;
        for (int j = 0; j < TK; j++) {
            shifted[i][j] = scores[i][j] - maxv;
            exp_scores[i][j] = expf(shifted[i][j]);
            sum += exp_scores[i][j];
        }
        for (int j = 0; j < TK; j++) {
            probs[i][j] = exp_scores[i][j] / sum;
        }
    }

    for (int i = 0; i < TQ; i++) {
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int j = 0; j < TK; j++) {
                sum += probs[i][j] * v[j * D + d];
            }
            out[i * D + d] = sum;
        }
    }
}

int main(void) {
    float q[TQ * D];
    float k[TK * D];
    float v[TK * D];
    float out[TQ * D];

    for (int i = 0; i < TQ * D; i++) q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < TK * D; i++) k[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < TK * D; i++) v[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    sparse_attention_antiquant_x86(q, k, v, out);

    printf("out[0..3]: ");
    for (int i = 0; i < D; i++) printf("%f ", out[i]);
    printf("\n");
    return 0;
}
