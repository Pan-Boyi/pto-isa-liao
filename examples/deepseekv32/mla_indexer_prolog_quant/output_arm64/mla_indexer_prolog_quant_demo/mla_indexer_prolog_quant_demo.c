// PTO Program: mla_indexer_prolog_quant_demo
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: mla_indexer_prolog_quant_demo
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 384 bytes (0.4 KB)
//   Total capacity (w/ reuse): 384 bytes (0.4 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   q_norm               4x4        f32        64   [  4,  -1]           -
//   q_out                4x4        f32        64   [  5,   7]           -
//   w_dq                 4x4        f32        64   [  1,  -1]           -
//   w_proj               4x2        f32        32   [  3,  -1]           -
//   w_qb                 4x4        f32        64   [  2,  -1]           -
//   weights              4x2        f32        32   [  6,   8]           -
//   x                    4x4        f32        64   [  0,  -1]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void mla_indexer_prolog_quant_demo(float* x_mem, float* w_dq_mem, float* w_qb_mem, float* w_proj_mem, float* q_out_mem, float* weights_mem) {
    float x[4][4];
    float w_dq[4][4];
    float w_qb[4][4];
    float w_proj[4][2];
    float q_norm[4][4];
    float q_out[4][4];
    float weights[4][2];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): x=TLOAD(x_mem,0,0); w_dq=TLOAD(w_dq_mem,0,0); w_qb=TLOAD(w_qb_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&x_mem[_row * 4 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&w_dq_mem[_row * 4 + _col]);
            vst1q_f32(&w_dq[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&w_qb_mem[_row * 4 + _col]);
            vst1q_f32(&w_qb[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            x[_row][_col] = x_mem[_row * 4 + _col];
            w_dq[_row][_col] = w_dq_mem[_row * 4 + _col];
            w_qb[_row][_col] = w_qb_mem[_row * 4 + _col];
        }
    }

    // FUSED LOOP (1 ops): w_proj=TLOAD(w_proj_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 2; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&w_proj_mem[_row * 2 + _col]);
            vst1q_f32(&w_proj[_row][_col], _vl3);
        }
        // Scalar cleanup
        for (; _col < 2; _col++) {
            w_proj[_row][_col] = w_proj_mem[_row * 2 + _col];
        }
    }

    // TMATMUL: q_norm = x @ w_dq
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 4; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += x[_i][_k] * w_dq[_k][_j];}
            q_norm[_i][_j] = _sum;}}

    // TMATMUL: q_out = q_norm @ w_qb
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 4; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += q_norm[_i][_k] * w_qb[_k][_j];}
            q_out[_i][_j] = _sum;}}

    // TMATMUL: weights = x @ w_proj
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 2; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += x[_i][_k] * w_proj[_k][_j];}
            weights[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): q_out_mem=TSTORE(q_out,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vs4 = vld1q_f32(&q_out[_row][_col]);
            vst1q_f32(&q_out_mem[_row * 4 + _col], _vs4);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            q_out_mem[_row * 4 + _col] = q_out[_row][_col];
        }
    }

    // FUSED LOOP (1 ops): weights_mem=TSTORE(weights,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 2; _col += 4) {
            float32x4_t _vs5 = vld1q_f32(&weights[_row][_col]);
            vst1q_f32(&weights_mem[_row * 2 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 2; _col++) {
            weights_mem[_row * 2 + _col] = weights[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "mla_indexer_prolog_quant_demo"; }
enum { kPtoNumMemrefs = 6 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "x_mem",
    "w_dq_mem",
    "w_qb_mem",
    "w_proj_mem",
    "q_out_mem",
    "weights_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(64),
    (size_t)(64),
    (size_t)(64),
    (size_t)(32),
    (size_t)(64),
    (size_t)(32),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
    0,
    0,
    1,
    1,
};
int pto_num_memrefs() { return kPtoNumMemrefs; }
const char* pto_memref_name(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefNames[idx];
}
size_t pto_memref_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefBytes[idx];
}
const char* pto_memref_dtype(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefDtypes[idx];
}
size_t pto_memref_elem_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefElemBytes[idx];
}
int pto_memref_is_output(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefIsOutput[idx];
}
void pto_launch(void **args, void *stream) {
    (void)stream;
    mla_indexer_prolog_quant_demo((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4], (float*)args[5]);
}
#endif  // PTO_CPU_SMOKE_RUNNER