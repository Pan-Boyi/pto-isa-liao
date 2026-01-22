// PTO Program: tile_add
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
static float _tile_tile_add_a[4096];
static float _tile_tile_add_b[4096];
static float _tile_tile_add_result[4096];

int64_t tile_add_sim(float* input_a, float* input_b, float* output) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(4096 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD a
    _cycle_count += (int64_t)(4096 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD b
    _cycle_count += CYCLE_VECTOR_OP;  // TADD result
    _cycle_count += (int64_t)(4096 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE result

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void tile_add(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t tile_add_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* input_a = (float*)args[0];
    float* input_b = (float*)args[1];
    float* output = (float*)args[2];
    return tile_add_sim(input_a, input_b, output);
}