// PTO Program: lightning_indexer_quant_demo
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: lightning_indexer_quant_demo
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 256 bytes (0.2 KB)
//   Total capacity (w/ reuse): 192 bytes (0.2 KB)
//   Reuse savings:            64 bytes (25.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   k                    4x4        f32        64   [  1,   2]           -
//   k_t                  4x4        f32        64   [  2,  -1]           -
//   q                    4x4        f32        64   [  0,  -1]           -
//   scores               4x4        f32        64   [  3,   4]           <- k
//
// BUFFER REUSE MAP:
//   scores reuses buffer of k
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void lightning_indexer_quant_demo(float* q_mem, float* k_mem, float* scores_mem) {
    float q[4][4];
    float k[4][4];
    float k_t[4][4];
    float scores[4][4];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (2 ops): q=TLOAD(q_mem,0,0); k=TLOAD(k_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&q_mem[_row * 4 + _col]);
            vst1q_f32(&q[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&k_mem[_row * 4 + _col]);
            vst1q_f32(&k[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            q[_row][_col] = q_mem[_row * 4 + _col];
            k[_row][_col] = k_mem[_row * 4 + _col];
        }
    }

    // TMATMUL: scores = q @ k_t
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 4; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += q[_i][_k] * k_t[_k][_j];}
            scores[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): scores_mem=TSTORE(scores,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vs2 = vld1q_f32(&scores[_row][_col]);
            vst1q_f32(&scores_mem[_row * 4 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            scores_mem[_row * 4 + _col] = scores[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "lightning_indexer_quant_demo"; }
enum { kPtoNumMemrefs = 3 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "q_mem",
    "k_mem",
    "scores_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(64),
    (size_t)(64),
    (size_t)(64),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
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
    lightning_indexer_quant_demo((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_CPU_SMOKE_RUNNER