// PTO Program: mla_prolog_quant_demo
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
static const int mla_prolog_quant_demo_is_cube = 0;

// Tile buffers (for scalar simulation)
static float _tile_mla_prolog_quant_demo_x[16];
static float _tile_mla_prolog_quant_demo_w_dq[16];
static float _tile_mla_prolog_quant_demo_w_uq_qr[16];
static float _tile_mla_prolog_quant_demo_x_proj[16];
static float _tile_mla_prolog_quant_demo_x_sq[16];
static float _tile_mla_prolog_quant_demo_sum_sq[4];
static float _tile_mla_prolog_quant_demo_mean_sq[4];
static float _tile_mla_prolog_quant_demo_mean_eps[4];
static float _tile_mla_prolog_quant_demo_rms[4];
static float _tile_mla_prolog_quant_demo_x_norm[16];
static float _tile_mla_prolog_quant_demo_q_out[16];

int64_t mla_prolog_quant_demo_sim(float* x_mem, float* w_dq_mem, float* w_uq_qr_mem, float* q_out_mem) {
    int64_t _cycle_count = 0;


    _cycle_count += (int64_t)(16 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD x
    _cycle_count += (int64_t)(16 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD w_dq
    _cycle_count += (int64_t)(16 * CYCLE_TLOAD_PER_ELEMENT);  // TLOAD w_uq_qr
    _cycle_count += (int64_t)(64 * CYCLE_MATMUL_PER_MAC);  // TMATMUL
    _cycle_count += CYCLE_VECTOR_OP;  // TMUL x_sq
    _cycle_count += 4 * CYCLE_REDUCTION_PER_ROW;  // TROWSUM x_sq
    _cycle_count += CYCLE_VECTOR_OP;  // TDIVS mean_sq
    _cycle_count += CYCLE_VECTOR_OP;  // TADDS mean_eps
    _cycle_count += CYCLE_VECTOR_OP;  // TSQRT rms
    _cycle_count += CYCLE_VECTOR_OP;  // TROWEXPANDDIV x_norm
    _cycle_count += (int64_t)(64 * CYCLE_MATMUL_PER_MAC);  // TMATMUL
    _cycle_count += (int64_t)(16 * CYCLE_TSTORE_PER_ELEMENT);  // TSTORE q_out

    return _cycle_count;
}

// PTO Runtime compatible wrapper
void mla_prolog_quant_demo(void** args, int32_t num_args) {
    (void)args; (void)num_args;
    // This wrapper is for compatibility; use _sim version for cycle count
}

// Get cycle count from runtime arguments
int64_t mla_prolog_quant_demo_cycle_count(void** args, int32_t num_args) {
    (void)num_args;
    float* x_mem = (float*)args[0];
    float* w_dq_mem = (float*)args[1];
    float* w_uq_qr_mem = (float*)args[2];
    float* q_out_mem = (float*)args[3];
    return mla_prolog_quant_demo_sim(x_mem, w_dq_mem, w_uq_qr_mem, q_out_mem);
}

// Get is_cube flag for this function
int mla_prolog_quant_demo_get_is_cube(void) {
    return mla_prolog_quant_demo_is_cube;
}