// PTO Program: lightning_indexer_prolog_quant_demo
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: lightning_indexer_prolog_quant_demo
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 576 bytes (0.6 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            64 bytes (11.1%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   q_matmul             4x4        f32        64   [  6,   7]           -
//   q_norm               4x4        f32        64   [  0,  -1]           -
//   q_norm_scale         4x1        f32        16   [  1,   7]           -
//   q_out                4x4        f32        64   [  9,  12]           -
//   q_scaled_row         4x4        f32        64   [  7,   9]           -
//   w_proj               4x2        f32        32   [  5,  -1]           -
//   w_qb                 4x4        f32        64   [  2,  -1]           -
//   w_qb_scale           1x4        f32        16   [  3,   8]           -
//   w_qb_scale_expand    4x4        f32        64   [  8,   9]           <- q_matmul
//   weights_out          4x2        f32        32   [ 11,  13]           -
//   weights_raw          4x2        f32        32   [ 10,  11]           -
//   x                    4x4        f32        64   [  4,  -1]           -
//
// BUFFER REUSE MAP:
//   w_qb_scale_expand reuses buffer of q_matmul
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void lightning_indexer_prolog_quant_demo(float* q_norm_mem, float* q_norm_scale_mem, float* w_qb_mem, float* w_qb_scale_mem, float* x_mem, float* w_proj_mem, float* q_out_mem, float* weights_out_mem) {
    float q_norm[4][4];
    float q_norm_scale[4][1];
    float w_qb[4][4];
    float w_qb_scale[1][4];
    float w_qb_scale_expand[4][4];
    float q_matmul[4][4];
    float q_scaled_row[4][4];
    float q_out[4][4];
    float x[4][4];
    float w_proj[4][2];
    float weights_raw[4][2];
    float weights_out[4][2];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (1 ops): q_norm=TLOAD(q_norm_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&q_norm_mem[_row * 4 + _col]);
            vst1q_f32(&q_norm[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            q_norm[_row][_col] = q_norm_mem[_row * 4 + _col];
        }
    }

    // FUSED LOOP (1 ops): q_norm_scale=TLOAD(q_norm_scale_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&q_norm_scale_mem[_row * 1 + _col]);
            vst1q_f32(&q_norm_scale[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            q_norm_scale[_row][_col] = q_norm_scale_mem[_row * 1 + _col];
        }
    }

    // FUSED LOOP (1 ops): w_qb=TLOAD(w_qb_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&w_qb_mem[_row * 4 + _col]);
            vst1q_f32(&w_qb[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            w_qb[_row][_col] = w_qb_mem[_row * 4 + _col];
        }
    }

    // FUSED LOOP (1 ops): w_qb_scale=TLOAD(w_qb_scale_mem,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl3 = vld1q_f32(&w_qb_scale_mem[_row * 4 + _col]);
            vst1q_f32(&w_qb_scale[_row][_col], _vl3);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            w_qb_scale[_row][_col] = w_qb_scale_mem[_row * 4 + _col];
        }
    }

    // FUSED LOOP (1 ops): x=TLOAD(x_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl4 = vld1q_f32(&x_mem[_row * 4 + _col]);
            vst1q_f32(&x[_row][_col], _vl4);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            x[_row][_col] = x_mem[_row * 4 + _col];
        }
    }

    // FUSED LOOP (1 ops): w_proj=TLOAD(w_proj_mem,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 2; _col += 4) {
            float32x4_t _vl5 = vld1q_f32(&w_proj_mem[_row * 2 + _col]);
            vst1q_f32(&w_proj[_row][_col], _vl5);
        }
        // Scalar cleanup
        for (; _col < 2; _col++) {
            w_proj[_row][_col] = w_proj_mem[_row * 2 + _col];
        }
    }

    // TMATMUL: q_matmul = q_norm @ w_qb
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 4; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += q_norm[_i][_k] * w_qb[_k][_j];}
            q_matmul[_i][_j] = _sum;}}

    // FUSED LOOP (2 ops): q_scaled_row=TROWEXPANDMUL(q_matmul,q_norm_scale); q_out=TMUL(q_scaled_row,w_qb_scale_expand)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _v06 = vld1q_f32(&q_matmul[_row][_col]);
            float32x4_t _vb8 = vdupq_n_f32(q_norm_scale[_row][0]);
            float32x4_t _vr7 = vmulq_f32(_v06, _vb8);
            vst1q_f32(&q_scaled_row[_row][_col], _vr7);
            float32x4_t _v9 = vld1q_f32(&q_scaled_row[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&w_qb_scale_expand[_row][_col]);
            float32x4_t _vr11 = vmulq_f32(_v9, _v10);
            vst1q_f32(&q_out[_row][_col], _vr11);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            q_scaled_row[_row][_col] = q_matmul[_row][_col] * q_norm_scale[_row][0];
            q_out[_row][_col] = q_scaled_row[_row][_col] * w_qb_scale_expand[_row][_col];
        }
    }

    // TMATMUL: weights_raw = x @ w_proj
    for (int _i = 0; _i < 4; _i++) {
        for (int _j = 0; _j < 2; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 4; _k++) {
                _sum += x[_i][_k] * w_proj[_k][_j];}
            weights_raw[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): weights_out=TMULS(weights_raw,0.5f)
    float32x4_t _vs12 = vdupq_n_f32(0.5f);
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 2; _col += 4) {
            float32x4_t _v13 = vld1q_f32(&weights_raw[_row][_col]);
            float32x4_t _vr14 = vmulq_f32(_v13, _vs12);
            vst1q_f32(&weights_out[_row][_col], _vr14);
        }
        // Scalar cleanup
        for (; _col < 2; _col++) {
            weights_out[_row][_col] = weights_raw[_row][_col] * 0.5f;
        }
    }

    // FUSED LOOP (1 ops): q_out_mem=TSTORE(q_out,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vs15 = vld1q_f32(&q_out[_row][_col]);
            vst1q_f32(&q_out_mem[_row * 4 + _col], _vs15);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            q_out_mem[_row * 4 + _col] = q_out[_row][_col];
        }
    }

    // FUSED LOOP (1 ops): weights_out_mem=TSTORE(weights_out,0,0)
    for (int _row = 0; _row < 4; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 2; _col += 4) {
            float32x4_t _vs16 = vld1q_f32(&weights_out[_row][_col]);
            vst1q_f32(&weights_out_mem[_row * 2 + _col], _vs16);
        }
        // Scalar cleanup
        for (; _col < 2; _col++) {
            weights_out_mem[_row * 2 + _col] = weights_out[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "lightning_indexer_prolog_quant_demo"; }
enum { kPtoNumMemrefs = 8 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "q_norm_mem",
    "q_norm_scale_mem",
    "w_qb_mem",
    "w_qb_scale_mem",
    "x_mem",
    "w_proj_mem",
    "q_out_mem",
    "weights_out_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(64),
    (size_t)(16),
    (size_t)(64),
    (size_t)(16),
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
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
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
    lightning_indexer_prolog_quant_demo((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4], (float*)args[5], (float*)args[6], (float*)args[7]);
}
#endif  // PTO_CPU_SMOKE_RUNNER