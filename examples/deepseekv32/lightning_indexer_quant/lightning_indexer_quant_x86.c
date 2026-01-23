/**
 * x86 demo for lightning_indexer_quant (simplified).
 */

#include <stdio.h>
#include <stdlib.h>

#define T 4
#define D 4

void lightning_indexer_quant_x86(const float* q, const float* k, float* scores) {
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += q[i * D + d] * k[j * D + d];
            }
            scores[i * T + j] = sum;
        }
    }
}

int main(void) {
    float q[T * D];
    float k[T * D];
    float scores[T * T];

    for (int i = 0; i < T * D; i++) q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < T * D; i++) k[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    lightning_indexer_quant_x86(q, k, scores);

    printf("scores[0..3]: ");
    for (int i = 0; i < T; i++) printf("%f ", scores[i]);
    printf("\n");
    return 0;
}
