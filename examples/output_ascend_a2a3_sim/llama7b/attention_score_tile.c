// PTO Program: attention_score_tile
// Backend: Ascend A2/A3 Cycle Simulator (runs on ARM64)
// This code simulates NPU execution for cycle estimation

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Cycle cost constants (simplified model)
#ifndef CYCLE_TLOAD_PER_ELEMENT
#define CYCLE_TLOAD_PER_ELEMENT  0.0625f  // 1 cycle per 16 elements (512-bit bus)
#define CYCLE_TSTORE_PER_ELEMENT 0.0625f
#define CYCLE_VECTOR_OP          1        // Most vector ops = 1 cycle
#define CYCLE_REDUCTION_PER_ROW  1        // Row reductions
#define CYCLE_MATMUL_PER_MAC     0.001f   // Cube unit: ~1000 MACs per cycle
#endif

// Tile buffers (for scalar simulation)
static float _tile_attention_score_tile_q[4096];
static float _tile_attention_score_tile_k_t[16384];
static float _tile_attention_score_tile_scores[4096];
static float _tile_attention_score_tile_scaled_scores[4096];

int64_t attention_score_tile_sim(float* input_q, float* input_kt, float* output, float scale) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(4096 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD q
    _cycle_count += (int64_t)(16384 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD k_t
    _cycle_count += (int64_t)(524288 * CYCLE_MATMUL_PER_MAC);  // TMATMUL
    scale = 0.08838834764831843;
    _cycle_count += CYCLE_VECTOR_OP;  // TMULS scaled_scores
    _cycle_count += (int64_t)(4096 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE scaled_scores

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void attention_score_tile(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t attention_score_tile_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* input_q = (float*)args[0];
    float* input_kt = (float*)args[1];
    float* output = (float*)args[2];
    float scale = *(float*)args[3];
    return attention_score_tile_sim(input_q, input_kt, output, scale);
}