// PTO Program: flash_attn_score_block
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
static float _tile_flash_attn_score_block_q_block[8192];
static float _tile_flash_attn_score_block_k_block[8192];
static float _tile_flash_attn_score_block_s_block[4096];
static float _tile_flash_attn_score_block_s_scaled[4096];

int64_t flash_attn_score_block_sim(float* input_q, float* input_k, float* output_s, float scale) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(8192 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD q_block
    _cycle_count += (int64_t)(8192 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD k_block
    _cycle_count += (int64_t)(262144 * CYCLE_MATMUL_PER_MAC);  // TMATMUL
    scale = 0.08838834764831843;
    _cycle_count += CYCLE_VECTOR_OP;  // TMULS s_scaled
    _cycle_count += (int64_t)(4096 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE s_scaled

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void flash_attn_score_block(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t flash_attn_score_block_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* input_q = (float*)args[0];
    float* input_k = (float*)args[1];
    float* output_s = (float*)args[2];
    float scale = *(float*)args[3];
    return flash_attn_score_block_sim(input_q, input_k, output_s, scale);
}