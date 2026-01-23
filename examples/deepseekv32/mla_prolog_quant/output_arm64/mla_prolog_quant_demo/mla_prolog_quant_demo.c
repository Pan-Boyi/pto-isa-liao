// PTO Program: mla_prolog_quant_demo
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: mla_prolog_quant_demo
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     11
//   Total capacity (no reuse): 512 bytes (0.5 KB)
//   Total capacity (w/ reuse): 352 bytes (0.3 KB)
//   Reuse savings:            160 bytes (31.2%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   mean_eps             4x1        f32        16   [  7,   8]           <- sum_sq
//   mean_sq              4x1        f32        16   [  6,   7]           -
//   q_out                4x4        f32        64   [ 10,  11]           <- x_proj
//   rms                  4x1        f32        16   [  8,   9]           <- mean_sq
//   sum_sq               4x1        f32        16   [  5,   6]           -
//   w_dq                 4x4        f32        64   [  1,  -1]           -
//   w_uq_qr              4x4        f32        64   [  2,  -1]           -
//   x                    4x4        f32        64   [  0,  -1]           -
//   x_norm               4x4        f32        64   [  9,  -1]           <- x_sq
//   x_proj               4x4        f32        64   [  3,   9]           -
//   x_sq                 4x4        f32        64   [  4,   5]           -
//
// BUFFER REUSE MAP:
//   mean_eps reuses buffer of sum_sq
//   rms reuses buffer of mean_sq
//   x_norm reuses buffer of x_sq
//   q_out reuses buffer of x_proj
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void mla_prolog_quant_demo(float* x_mem, float* w_dq_mem, float* w_uq_qr_mem, float* q_out_mem) {
    float x[4][4];
    float w_dq[4][4];
    float w_uq_qr[4][4];
    float x_proj[4][4];
    float x_sq[4][4];
    float sum_sq[4][1];
    float mean_sq[4][1];
    float mean_eps[4][1];
    float rms[4][1];
    float x_norm[4][4];
    float q_out[4][4];

    // Loop fusion: 4 loop overheads saved

    // FUSED LOOP (3 ops): x=TLOAD(x_mem,0,0); w_dq=TLOAD(w_dq_mem,0,0); w_uq_qr=TLOAD(w_uq_qr_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&x_mem[_row * 4 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&w_dq_mem[_row * 4 + _col]);
            vst1q_f32(&w_dq[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&w_uq_qr_mem[_row * 4 + _col]);
            vst1q_f32(&w_uq_qr[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            x[_row][_col] = x_mem[_row * 4 + _col];
            w_dq[_row][_col] = w_dq_mem[_row * 4 + _col];
            w_uq_qr[_row][_col] = w_uq_qr_mem[_row * 4 + _col];
        }
    }

    // TMATMUL: x_proj = x @ w_dq
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 4; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += x[_i][_k] * w_dq[_k][_j];}
            x_proj[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): x_sq=TMUL(x_proj,x_proj)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&x_proj[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&x_proj[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v3, _v4);
            vst1q_f32(&x_sq[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            x_sq[_row][_col] = x_proj[_row][_col] * x_proj[_row][_col];
        }
    }

    // TROWSUM: sum_sq = rowsum(x_sq)
    for (int _row = 0; _row < 4; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 4; _col++) {
            _sum += x_sq[_row][_col];
        }
        sum_sq[_row][0] = _sum;}

    // FUSED LOOP (3 ops): mean_sq=TDIVS(sum_sq,4.0f); mean_eps=TADDS(mean_sq,1e-06f); rms=TSQRT(mean_eps)
    float32x4_t _vs6 = vdupq_n_f32(4.0f);
    float32x4_t _vs7 = vdupq_n_f32(1e-06f);
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v8 = vld1q_f32(&sum_sq[_row][_col]);
            float32x4_t _vr9 = vdivq_f32(_v8, _vs6);
            vst1q_f32(&mean_sq[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&mean_sq[_row][_col]);
            float32x4_t _vr11 = vaddq_f32(_v10, _vs7);
            vst1q_f32(&mean_eps[_row][_col], _vr11);
            float32x4_t _v12 = vld1q_f32(&mean_eps[_row][_col]);
            float32x4_t _vr13 = vsqrtq_f32(_v12);
            vst1q_f32(&rms[_row][_col], _vr13);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            mean_sq[_row][_col] = sum_sq[_row][_col] / 4.0f;
            mean_eps[_row][_col] = mean_sq[_row][_col] + 1e-06f;
            rms[_row][_col] = sqrtf(mean_eps[_row][_col]);
        }
    }

    // FUSED LOOP (1 ops): x_norm=TROWEXPANDDIV(x_proj,rms)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _v014 = vld1q_f32(&x_proj[_row][_col]);
            float32x4_t _vb16 = vdupq_n_f32(rms[_row][0]);
            float32x4_t _vr15 = vdivq_f32(_v014, _vb16);
            vst1q_f32(&x_norm[_row][_col], _vr15);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            x_norm[_row][_col] = x_proj[_row][_col] / rms[_row][0];
        }
    }

    // TMATMUL: q_out = x_norm @ w_uq_qr
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 4; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += x_norm[_i][_k] * w_uq_qr[_k][_j];}
            q_out[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): q_out_mem=TSTORE(q_out,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vs17 = vld1q_f32(&q_out[_row][_col]);
            vst1q_f32(&q_out_mem[_row * 4 + _col], _vs17);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            q_out_mem[_row * 4 + _col] = q_out[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "mla_prolog_quant_demo"; }
enum { kPtoNumMemrefs = 4 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "x_mem",
    "w_dq_mem",
    "w_uq_qr_mem",
    "q_out_mem",
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
    mla_prolog_quant_demo((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3]);
}
#endif  // PTO_CPU_SMOKE_RUNNER