// PTO Program: flash_attn_output_update
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
static float _tile_flash_attn_output_update_o_prev[8192];
static float _tile_flash_attn_output_update_p_block[4096];
static float _tile_flash_attn_output_update_v_block[8192];
static float _tile_flash_attn_output_update_scale_old[64];
static float _tile_flash_attn_output_update_o_scaled[8192];
static float _tile_flash_attn_output_update_pv[8192];
static float _tile_flash_attn_output_update_o_new[8192];

int64_t flash_attn_output_update_sim(float* input_o_prev, float* input_p, float* input_v, float* input_scale, float* output_o) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(8192 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD o_prev
    _cycle_count += (int64_t)(4096 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD p_block
    _cycle_count += (int64_t)(8192 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD v_block
    _cycle_count += (int64_t)(64 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD scale_old
    _cycle_count += CYCLE_VECTOR_OP;  // TROWEXPANDMUL o_scaled
    _cycle_count += (int64_t)(1048576 * CYCLE_MATMUL_PER_MAC);  // TMATMUL
    _cycle_count += CYCLE_VECTOR_OP;  // TADD o_new
    _cycle_count += (int64_t)(8192 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE o_new

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void flash_attn_output_update(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t flash_attn_output_update_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* input_o_prev = (float*)args[0];
    float* input_p = (float*)args[1];
    float* input_v = (float*)args[2];
    float* input_scale = (float*)args[3];
    float* output_o = (float*)args[4];
    return flash_attn_output_update_sim(input_o_prev, input_p, input_v, input_scale, output_o);
}