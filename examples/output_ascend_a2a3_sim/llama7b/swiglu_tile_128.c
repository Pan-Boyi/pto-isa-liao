// PTO Program: swiglu_tile_128
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
static float _tile_swiglu_tile_128_gate[16384];
static float _tile_swiglu_tile_128_up[16384];
static float _tile_swiglu_tile_128_neg_gate[16384];
static float _tile_swiglu_tile_128_exp_neg_gate[16384];
static float _tile_swiglu_tile_128_one_plus_exp[16384];
static float _tile_swiglu_tile_128_sigmoid_gate[16384];
static float _tile_swiglu_tile_128_gate_silu[16384];
static float _tile_swiglu_tile_128_result[16384];

int64_t swiglu_tile_128_sim(float* input_gate, float* input_up, float* output) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(16384 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD gate
    _cycle_count += (int64_t)(16384 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD up
    _cycle_count += CYCLE_VECTOR_OP;  // TNEG neg_gate
    _cycle_count += CYCLE_VECTOR_OP;  // TEXP exp_neg_gate
    _cycle_count += CYCLE_VECTOR_OP;  // TADDS one_plus_exp
    _cycle_count += CYCLE_VECTOR_OP;  // TRECIP sigmoid_gate
    _cycle_count += CYCLE_VECTOR_OP;  // TMUL gate_silu
    _cycle_count += CYCLE_VECTOR_OP;  // TMUL result
    _cycle_count += (int64_t)(16384 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE result

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void swiglu_tile_128(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t swiglu_tile_128_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* input_gate = (float*)args[0];
    float* input_up = (float*)args[1];
    float* output = (float*)args[2];
    return swiglu_tile_128_sim(input_gate, input_up, output);
}