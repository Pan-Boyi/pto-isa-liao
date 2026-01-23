// PTO Program: sparse_flash_attention_quant_demo
// Backend: Ascend A2/A3 Cycle Simulator (runs on ARM64)
// This code simulates NPU execution for cycle estimation
// is_cube: 0 (0=vector worker, 1=cube worker)

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

// Worker type flag: 0 = vector worker, 1 = cube worker
static const int sparse_flash_attention_quant_demo_is_cube = 0;

// Tile buffers (for scalar simulation)
static float _tile_sparse_flash_attention_quant_demo_q[16];
static float _tile_sparse_flash_attention_quant_demo_k[16];
static float _tile_sparse_flash_attention_quant_demo_v[16];
static float _tile_sparse_flash_attention_quant_demo_k_t[16];
static float _tile_sparse_flash_attention_quant_demo_scores[16];
static float _tile_sparse_flash_attention_quant_demo_scaled_scores[16];
static float _tile_sparse_flash_attention_quant_demo_row_max[4];
static float _tile_sparse_flash_attention_quant_demo_shifted[16];
static float _tile_sparse_flash_attention_quant_demo_exp_scores[16];
static float _tile_sparse_flash_attention_quant_demo_sum_exp[4];
static float _tile_sparse_flash_attention_quant_demo_probs[16];
static float _tile_sparse_flash_attention_quant_demo_out[16];

int64_t sparse_flash_attention_quant_demo_sim(float* q_mem, float* k_mem, float* v_mem, float* out_mem) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(16 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD q
    _cycle_count += (int64_t)(16 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD k
    _cycle_count += (int64_t)(16 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD v
    _cycle_count += CYCLE_VECTOR_OP;  // TTRANS k_t
    _cycle_count += (int64_t)(64 * CYCLE_MATMUL_PER_MAC);  // TMATMUL
    _cycle_count += CYCLE_VECTOR_OP;  // TMULS scaled_scores
    _cycle_count += 4 * CYCLE_REDUCTION_PER_ROW;  // TROWMAX scaled_scores
    _cycle_count += CYCLE_VECTOR_OP;  // TROWEXPANDSUB shifted
    _cycle_count += CYCLE_VECTOR_OP;  // TEXP exp_scores
    _cycle_count += 4 * CYCLE_REDUCTION_PER_ROW;  // TROWSUM exp_scores
    _cycle_count += CYCLE_VECTOR_OP;  // TROWEXPANDDIV probs
    _cycle_count += (int64_t)(64 * CYCLE_MATMUL_PER_MAC);  // TMATMUL
    _cycle_count += (int64_t)(16 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE out

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void sparse_flash_attention_quant_demo(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t sparse_flash_attention_quant_demo_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* q_mem = (float*)args[0];
    float* k_mem = (float*)args[1];
    float* v_mem = (float*)args[2];
    float* out_mem = (float*)args[3];
    return sparse_flash_attention_quant_demo_sim(q_mem, k_mem, v_mem, out_mem);
}

// Get is_cube flag for this function
int sparse_flash_attention_quant_demo_get_is_cube(void) {
    return sparse_flash_attention_quant_demo_is_cube;
}