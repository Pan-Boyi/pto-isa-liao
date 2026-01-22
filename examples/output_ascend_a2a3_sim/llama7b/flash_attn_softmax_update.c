// PTO Program: flash_attn_softmax_update
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
static float _tile_flash_attn_softmax_update_s_block[4096];
static float _tile_flash_attn_softmax_update_m_prev[64];
static float _tile_flash_attn_softmax_update_l_prev[64];
static float _tile_flash_attn_softmax_update_m_new[64];
static float _tile_flash_attn_softmax_update_m_cur[64];
static float _tile_flash_attn_softmax_update_l_new[64];
static float _tile_flash_attn_softmax_update_p_block[4096];
static float _tile_flash_attn_softmax_update_s_shifted[4096];
static float _tile_flash_attn_softmax_update_scale_old[64];
static float _tile_flash_attn_softmax_update_m_diff[64];
static float _tile_flash_attn_softmax_update_l_scaled[64];
static float _tile_flash_attn_softmax_update_p_rowsum[64];

int64_t flash_attn_softmax_update_sim(float* input_s, float* input_m_prev, float* input_l_prev, float* output_m_new, float* output_l_new, float* output_p, float* output_scale_old) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(4096 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD s_block
    _cycle_count += (int64_t)(64 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD m_prev
    _cycle_count += (int64_t)(64 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD l_prev
    _cycle_count += 64 * CYCLE_REDUCTION_PER_ROW;  // TROWMAX s_block
    _cycle_count += CYCLE_VECTOR_OP;  // TMAX m_new
    _cycle_count += CYCLE_VECTOR_OP;  // TROWEXPANDSUB s_shifted
    _cycle_count += CYCLE_VECTOR_OP;  // TEXP p_block
    _cycle_count += CYCLE_VECTOR_OP;  // TSUB m_diff
    _cycle_count += CYCLE_VECTOR_OP;  // TEXP scale_old
    _cycle_count += CYCLE_VECTOR_OP;  // TMUL l_scaled
    _cycle_count += 64 * CYCLE_REDUCTION_PER_ROW;  // TROWSUM p_block
    _cycle_count += CYCLE_VECTOR_OP;  // TADD l_new
    _cycle_count += (int64_t)(64 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE m_new
    _cycle_count += (int64_t)(64 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE l_new
    _cycle_count += (int64_t)(4096 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE p_block
    _cycle_count += (int64_t)(64 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE scale_old

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void flash_attn_softmax_update(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t flash_attn_softmax_update_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* input_s = (float*)args[0];
    float* input_m_prev = (float*)args[1];
    float* input_l_prev = (float*)args[2];
    float* output_m_new = (float*)args[3];
    float* output_l_new = (float*)args[4];
    float* output_p = (float*)args[5];
    float* output_scale_old = (float*)args[6];
    return flash_attn_softmax_update_sim(input_s, input_m_prev, input_l_prev, output_m_new, output_l_new, output_p, output_scale_old);
}