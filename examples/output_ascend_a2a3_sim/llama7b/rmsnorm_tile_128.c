// PTO Program: rmsnorm_tile_128
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
static float _tile_rmsnorm_tile_128_x[16384];
static float _tile_rmsnorm_tile_128_x_sq[16384];
static float _tile_rmsnorm_tile_128_row_sum[128];
static float _tile_rmsnorm_tile_128_row_mean[128];
static float _tile_rmsnorm_tile_128_row_rsqrt[128];
static float _tile_rmsnorm_tile_128_x_norm[16384];
static float _tile_rmsnorm_tile_128_gamma[16384];
static float _tile_rmsnorm_tile_128_result[16384];

int64_t rmsnorm_tile_128_sim(float* input, float* weights, float* output, float eps, float inv_cols) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(16384 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD x
    _cycle_count += (int64_t)(16384 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD gamma
    _cycle_count += CYCLE_VECTOR_OP;  // TMUL x_sq
    _cycle_count += 128 * CYCLE_REDUCTION_PER_ROW;  // TROWSUM x_sq
    inv_cols = 0.0078125;
    _cycle_count += CYCLE_VECTOR_OP;  // TMULS row_mean
    eps = 1e-05;
    _cycle_count += CYCLE_VECTOR_OP;  // TADDS row_mean
    _cycle_count += CYCLE_VECTOR_OP;  // TRSQRT row_rsqrt
    _cycle_count += CYCLE_VECTOR_OP;  // TROWEXPANDMUL x_norm
    _cycle_count += CYCLE_VECTOR_OP;  // TMUL result
    _cycle_count += (int64_t)(16384 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE result

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void rmsnorm_tile_128(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t rmsnorm_tile_128_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* weights = (float*)args[1];
    float* output = (float*)args[2];
    float eps = *(float*)args[3];
    float inv_cols = *(float*)args[4];
    return rmsnorm_tile_128_sim(input, weights, output, eps, inv_cols);
}