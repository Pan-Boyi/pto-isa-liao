// PTO Program: sparse_attention_antiquant_demo
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: sparse_attention_antiquant_demo
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 608 bytes (0.6 KB)
//   Total capacity (w/ reuse): 336 bytes (0.3 KB)
//   Reuse savings:            272 bytes (44.7%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_scores           4x4        f32        64   [  7,   9]           <- scores
//   k                    4x4        f32        64   [  1,   3]           -
//   k_t                  4x4        f32        64   [  3,  -1]           -
//   out                  4x4        f32        64   [ 10,  11]           <- exp_scores
//   probs                4x4        f32        64   [  9,  -1]           <- shifted
//   q                    4x4        f32        64   [  0,  -1]           -
//   row_max              4x1        f32        16   [  5,   6]           -
//   scores               4x4        f32        64   [  4,   6]           <- k
//   shifted              4x4        f32        64   [  6,   7]           -
//   sum_exp              4x1        f32        16   [  8,   9]           <- row_max
//   v                    4x4        f32        64   [  2,  -1]           -
//
// BUFFER REUSE MAP:
//   scores reuses buffer of k
//   exp_scores reuses buffer of scores
//   sum_exp reuses buffer of row_max
//   probs reuses buffer of shifted
//   out reuses buffer of exp_scores
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void sparse_attention_antiquant_demo(float* q_mem, float* k_mem, float* v_mem, float* out_mem) {
    float q[4][4];
    float k[4][4];
    float v[4][4];
    float k_t[4][4];
    float scores[4][4];
    float row_max[4][1];
    float shifted[4][4];
    float exp_scores[4][4];
    float sum_exp[4][1];
    float probs[4][4];
    float out[4][4];

    // Loop fusion: 3 loop overheads saved

    // FUSED LOOP (3 ops): q=TLOAD(q_mem,0,0); k=TLOAD(k_mem,0,0); v=TLOAD(v_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&q_mem[_row * 4 + _col]);
            vst1q_f32(&q[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&k_mem[_row * 4 + _col]);
            vst1q_f32(&k[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&v_mem[_row * 4 + _col]);
            vst1q_f32(&v[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            q[_row][_col] = q_mem[_row * 4 + _col];
            k[_row][_col] = k_mem[_row * 4 + _col];
            v[_row][_col] = v_mem[_row * 4 + _col];
        }
    }

    // TMATMUL: scores = q @ k_t
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 4; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += q[_i][_k] * k_t[_k][_j];}
            scores[_i][_j] = _sum;}}

    // TROWMAX: row_max = rowmax(scores)
    for (int _row = 0; _row < 4; _row++) {
        float _max = scores[_row][0];
        for (int _col = 1; _col < 4; _col++) {
            if (scores[_row][_col] > _max) _max = scores[_row][_col];
        }
        row_max[_row][0] = _max;}

    // FUSED LOOP (2 ops): shifted=TROWEXPANDSUB(scores,row_max); exp_scores=TEXP(shifted)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _v03 = vld1q_f32(&scores[_row][_col]);
            float32x4_t _vb5 = vdupq_n_f32(row_max[_row][0]);
            float32x4_t _vr4 = vsubq_f32(_v03, _vb5);
            vst1q_f32(&shifted[_row][_col], _vr4);
            float32x4_t _v6 = vld1q_f32(&shifted[_row][_col]);
            float32x4_t _vr7 = _v6;
            vst1q_f32(&exp_scores[_row][_col], _vr7);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            shifted[_row][_col] = scores[_row][_col] - row_max[_row][0];
            exp_scores[_row][_col] = expf(shifted[_row][_col]);
        }
    }

    // TROWSUM: sum_exp = rowsum(exp_scores)
    for (int _row = 0; _row < 4; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 4; _col++) {
            _sum += exp_scores[_row][_col];
        }
        sum_exp[_row][0] = _sum;}

    // FUSED LOOP (1 ops): probs=TROWEXPANDDIV(exp_scores,sum_exp)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _v08 = vld1q_f32(&exp_scores[_row][_col]);
            float32x4_t _vb10 = vdupq_n_f32(sum_exp[_row][0]);
            float32x4_t _vr9 = vdivq_f32(_v08, _vb10);
            vst1q_f32(&probs[_row][_col], _vr9);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            probs[_row][_col] = exp_scores[_row][_col] / sum_exp[_row][0];
        }
    }

    // TMATMUL: out = probs @ v
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 4; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += probs[_i][_k] * v[_k][_j];}
            out[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): out_mem=TSTORE(out,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vs11 = vld1q_f32(&out[_row][_col]);
            vst1q_f32(&out_mem[_row * 4 + _col], _vs11);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            out_mem[_row * 4 + _col] = out[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "sparse_attention_antiquant_demo"; }
enum { kPtoNumMemrefs = 4 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "q_mem",
    "k_mem",
    "v_mem",
    "out_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(64),
    (size_t)(64),
    (size_t)(64),
    (size_t)(64),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
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
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
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
    sparse_attention_antiquant_demo((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3]);
}
#endif  // PTO_CPU_SMOKE_RUNNER