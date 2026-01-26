// PTO Program: indexer_prolog
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: indexer_prolog
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     54
//   Total capacity (no reuse): 8,256 bytes (8.1 KB)
//   Total capacity (w/ reuse): 5,056 bytes (4.9 KB)
//   Reuse savings:            3,200 bytes (38.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_x                8x8        f32       256   [ 27,  70]           -
//   beta                 8x8        f32       256   [ 43,  54]           -
//   beta_1n              1x8        f32        32   [ 41,  43]           -
//   cache_index          8x1        i64        32   [ 77,  -1]           -
//   cos                  8x4        f32       128   [ 15,  63]           -
//   diff                 8x8        f32       256   [ 46,  52]           -
//   gamma                8x8        f32       256   [ 42,  53]           -
//   gamma_1n             1x8        f32        32   [ 40,  42]           <- w_scale_1n
//   hadamard_q           8x8        f32       256   [ 25,  -1]           <- w_scale_exp
//   k_lin                8x8        f32       256   [ 39,  46]           -
//   k_ln                 8x8        f32       256   [ 52,  74]           <- sq
//   k_ln_scaled          8x8        f32       256   [ 53,  54]           <- diff
//   k_nope               8x4        f32       128   [ 56,  -1]           -
//   k_rope               8x4        f32       128   [ 55,  63]           -
//   left                 8x2        f32        64   [ 17,  -1]           -
//   max_safe             8x1        f32        32   [ 29,  76]           -
//   mean                 8x1        f32        32   [ 45,  46]           -
//   part1                8x4        f32       128   [ 21,  65]           -
//   part2                8x4        f32       128   [ 22,  65]           <- q_rope
//   q                    8x8        f32       256   [ 12,  14]           <- w_f32
//   q_nope               8x4        f32       128   [ 14,  -1]           -
//   q_norm               8x8        i8         64   [  3,   7]           -
//   q_norm_f32           8x8        f32       256   [  7,   8]           -
//   q_norm_fp32          8x8        f32       256   [  8,  -1]           -
//   q_norm_scale         8x1        f32        32   [  4,   8]           -
//   q_pre                8x8        f32       256   [ 26,  32]           <- q
//   q_rope               8x4        f32       128   [ 13,  21]           -
//   q_rope_rotated       8x4        f32       128   [ 23,  -1]           -
//   right                8x2        f32        64   [ 18,  61]           -
//   right_neg            8x2        f32        64   [ 19,  -1]           -
//   rotated              8x4        f32       128   [ 20,  64]           -
//   rowmax               8x1        f32        32   [ 28,  71]           <- q_norm_scale
//   scale_127_tile       8x8        f32       256   [ 30,  73]           -
//   scale_dequant        8x1        f32        32   [ 34,  79]           -
//   scale_quant          8x8        f32       256   [ 31,  74]           -
//   sin                  8x4        f32       128   [ 16,  64]           -
//   sq                   8x8        f32       256   [ 47,  48]           <- k_lin
//   std                  8x1        f32        32   [ 51,  52]           <- var
//   sum                  8x1        f32        32   [ 44,  45]           -
//   var                  8x1        f32        32   [ 49,  50]           <- mean
//   var_eps              8x1        f32        32   [ 50,  51]           <- var_sum
//   var_sum              8x1        f32        32   [ 48,  49]           <- sum
//   w                    8x8        i8         64   [  5,   9]           -
//   w_f32                8x8        f32       256   [  9,  11]           <- q_norm_f32
//   w_fp32               8x8        f32       256   [ 11,  -1]           -
//   w_k                  8x8        f32       256   [ 38,  -1]           -
//   w_proj               8x8        f32       256   [ 80,  -1]           <- abs_x
//   w_scale_1n           1x8        f32        32   [  6,  10]           -
//   w_scale_exp          8x8        f32       256   [ 10,  11]           -
//   weights              8x8        f32       256   [ 81,  82]           <- scale_127_tile
//   weights_scaled       8x8        f32       256   [ 82,  83]           <- scale_quant
//   x                    8x8        f32       256   [ 37,  -1]           <- q_pre
//   y                    8x8        f32       256   [ 32,  75]           -
//   y_int8               8x8        i8         64   [ 33,  78]           <- q_norm
//
// BUFFER REUSE MAP:
//   w_f32 reuses buffer of q_norm_f32
//   q reuses buffer of w_f32
//   part2 reuses buffer of q_rope
//   hadamard_q reuses buffer of w_scale_exp
//   q_pre reuses buffer of q
//   rowmax reuses buffer of q_norm_scale
//   y_int8 reuses buffer of q_norm
//   x reuses buffer of q_pre
//   sq reuses buffer of k_lin
//   var_sum reuses buffer of sum
//   var reuses buffer of mean
//   var_eps reuses buffer of var_sum
//   std reuses buffer of var
//   k_ln reuses buffer of sq
//   k_ln_scaled reuses buffer of diff
//   gamma_1n reuses buffer of w_scale_1n
//   w_proj reuses buffer of abs_x
//   weights reuses buffer of scale_127_tile
//   weights_scaled reuses buffer of scale_quant
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void indexer_prolog(float* input_x, int8_t* input_q_norm, float* input_q_norm_scale, int8_t* input_w, float* input_w_qb_scale, float* input_w_k, float* input_w_proj, float* input_cos, float* input_sin, float* input_hadamard_q, float* input_hadamard_k, float* input_layer_norm_gamma, float* input_layer_norm_beta, int64_t* cache_index_src, int8_t* k_cache, float* k_scale_cache, int8_t* output_query, float* output_query_scale, float* output_weights) {
    int8_t q_norm[8][8];
    float q_norm_scale[8][1];
    float q_norm_f32[8][8];
    float q_norm_fp32[8][8];
    int8_t w[8][8];
    float w_scale_1n[1][8];
    float w_f32[8][8];
    float w_scale_exp[8][8];
    float w_fp32[8][8];
    float q[8][8];
    float q_rope[8][4];
    float q_nope[8][4];
    float cos[8][4];
    float sin[8][4];
    float left[8][2];
    float right[8][2];
    float right_neg[8][2];
    float rotated[8][4];
    float part1[8][4];
    float part2[8][4];
    float q_rope_rotated[8][4];
    float hadamard_q[8][8];
    float q_pre[8][8];
    float abs_x[8][8];
    float rowmax[8][1];
    float max_safe[8][1];
    float scale_127_tile[8][8];
    float scale_quant[8][8];
    float y[8][8];
    int8_t y_int8[8][8];
    float scale_dequant[8][1];
    float x[8][8];
    float w_k[8][8];
    float k_lin[8][8];
    float sum[8][1];
    float mean[8][1];
    float diff[8][8];
    float sq[8][8];
    float var_sum[8][1];
    float var[8][1];
    float var_eps[8][1];
    float std[8][1];
    float k_ln[8][8];
    float k_ln_scaled[8][8];
    float gamma_1n[1][8];
    float beta_1n[1][8];
    float gamma[8][8];
    float beta[8][8];
    float k_rope[8][4];
    float k_nope[8][4];
    int64_t cache_index[8][1];
    float w_proj[8][8];
    float weights[8][8];
    float weights_scaled[8][8];

    // Loop fusion: 18 loop overheads saved

    int inv_cols = 8;

    int eps = 0;

    int scale = 8;

    // FUSED LOOP (1 ops): q_norm=TLOAD(input_q_norm,0,0)
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 8; _col++) {
            q_norm[_row][_col] = input_q_norm[_row * 8 + _col];
        }
    }

    // FUSED LOOP (1 ops): q_norm_scale=TLOAD(input_q_norm_scale,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_q_norm_scale[_row * 1 + _col]);
            vst1q_f32(&q_norm_scale[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            q_norm_scale[_row][_col] = input_q_norm_scale[_row * 1 + _col];
        }
    }

    // FUSED LOOP (1 ops): w=TLOAD(input_w,0,0)
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 8; _col++) {
            w[_row][_col] = input_w[_row * 8 + _col];
        }
    }

    // FUSED LOOP (1 ops): w_scale_1n=TLOAD(input_w_qb_scale,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_w_qb_scale[_row * 8 + _col]);
            vst1q_f32(&w_scale_1n[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            w_scale_1n[_row][_col] = input_w_qb_scale[_row * 8 + _col];
        }
    }

    // TCVT: q_norm_f32 = (float)(q_norm) rmode=CAST_RINT
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            q_norm_f32[_i][_j] = (float)(q_norm[_i][_j]);
        }}

    // FUSED LOOP (1 ops): q_norm_fp32=TROWEXPANDMUL(q_norm_f32,q_norm_scale)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v02 = vld1q_f32(&q_norm_f32[_row][_col]);
            float32x4_t _vb4 = vdupq_n_f32(q_norm_scale[_row][0]);
            float32x4_t _vr3 = vmulq_f32(_v02, _vb4);
            vst1q_f32(&q_norm_fp32[_row][_col], _vr3);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            q_norm_fp32[_row][_col] = q_norm_f32[_row][_col] * q_norm_scale[_row][0];
        }
    }

    // TCVT: w_f32 = (float)(w) rmode=CAST_RINT
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            w_f32[_i][_j] = (float)(w[_i][_j]);
        }}

    // TCOLEXPAND: w_scale_exp = broadcast first row of w_scale_1n -> dst[i,j]=src[0,j]
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            w_scale_exp[_i][_j] = w_scale_1n[0][_j];
        }}

    // FUSED LOOP (1 ops): w_fp32=TMUL(w_f32,w_scale_exp)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v5 = vld1q_f32(&w_f32[_row][_col]);
            float32x4_t _v6 = vld1q_f32(&w_scale_exp[_row][_col]);
            float32x4_t _vr7 = vmulq_f32(_v5, _v6);
            vst1q_f32(&w_fp32[_row][_col], _vr7);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            w_fp32[_row][_col] = w_f32[_row][_col] * w_scale_exp[_row][_col];
        }
    }

    // TMATMUL: q = q_norm_fp32 @ w_fp32
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += q_norm_fp32[_i][_k] * w_fp32[_k][_j];}
            q[_i][_j] = _sum;}}

    // TEXTRACT: q_rope = extract(q, [0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 4; _col++) {
            q_rope[_row][_col] = q[(0) + _row][(0) + _col];
        }}

    // TEXTRACT: q_nope = extract(q, [0, 4])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 4; _col++) {
            q_nope[_row][_col] = q[(0) + _row][(4) + _col];
        }}

    // FUSED LOOP (2 ops): cos=TLOAD(input_cos,0,0); sin=TLOAD(input_sin,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl8 = vld1q_f32(&input_cos[_row * 4 + _col]);
            vst1q_f32(&cos[_row][_col], _vl8);
            float32x4_t _vl9 = vld1q_f32(&input_sin[_row * 4 + _col]);
            vst1q_f32(&sin[_row][_col], _vl9);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            cos[_row][_col] = input_cos[_row * 4 + _col];
            sin[_row][_col] = input_sin[_row * 4 + _col];
        }
    }

    // TEXTRACT: left = extract(q_rope, [0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 2; _col++) {
            left[_row][_col] = q_rope[(0) + _row][(0) + _col];
        }}

    // TEXTRACT: right = extract(q_rope, [0, 2])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 2; _col++) {
            right[_row][_col] = q_rope[(0) + _row][(2) + _col];
        }}

    // FUSED LOOP (1 ops): right_neg=TNEG(right)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 2; _col += 4) {
            float32x4_t _v10 = vld1q_f32(&right[_row][_col]);
            float32x4_t _vr11 = vnegq_f32(_v10);
            vst1q_f32(&right_neg[_row][_col], _vr11);
        }
        // Scalar cleanup
        for (; _col < 2; _col++) {
            right_neg[_row][_col] = -right[_row][_col];
        }
    }

    // TCONCAT(axis=1): rotated = [ right_neg | left ]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 2; _col++) {
            rotated[_row][_col] = right_neg[_row][_col];
        }
        for (int _col = 0; _col < 2; _col++) {
            rotated[_row][2 + _col] = left[_row][_col];
        }}

    // FUSED LOOP (3 ops): part1=TMUL(q_rope,cos); part2=TMUL(rotated,sin); q_rope_rotated=TADD(part1,part2)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _v12 = vld1q_f32(&q_rope[_row][_col]);
            float32x4_t _v13 = vld1q_f32(&cos[_row][_col]);
            float32x4_t _vr14 = vmulq_f32(_v12, _v13);
            vst1q_f32(&part1[_row][_col], _vr14);
            float32x4_t _v15 = vld1q_f32(&rotated[_row][_col]);
            float32x4_t _v16 = vld1q_f32(&sin[_row][_col]);
            float32x4_t _vr17 = vmulq_f32(_v15, _v16);
            vst1q_f32(&part2[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&part1[_row][_col]);
            float32x4_t _v19 = vld1q_f32(&part2[_row][_col]);
            float32x4_t _vr20 = vaddq_f32(_v18, _v19);
            vst1q_f32(&q_rope_rotated[_row][_col], _vr20);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            part1[_row][_col] = q_rope[_row][_col] * cos[_row][_col];
            part2[_row][_col] = rotated[_row][_col] * sin[_row][_col];
            q_rope_rotated[_row][_col] = part1[_row][_col] + part2[_row][_col];
        }
    }

    // TCONCAT(axis=1): q = [ q_rope_rotated | q_nope ]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 4; _col++) {
            q[_row][_col] = q_rope_rotated[_row][_col];
        }
        for (int _col = 0; _col < 4; _col++) {
            q[_row][4 + _col] = q_nope[_row][_col];
        }}

    // FUSED LOOP (1 ops): hadamard_q=TLOAD(input_hadamard_q,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl21 = vld1q_f32(&input_hadamard_q[_row * 8 + _col]);
            vst1q_f32(&hadamard_q[_row][_col], _vl21);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            hadamard_q[_row][_col] = input_hadamard_q[_row * 8 + _col];
        }
    }

    // TMATMUL: q_pre = q @ hadamard_q
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += q[_i][_k] * hadamard_q[_k][_j];}
            q_pre[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): abs_x=TABS(q_pre)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v22 = vld1q_f32(&q_pre[_row][_col]);
            float32x4_t _vr23 = vabsq_f32(_v22);
            vst1q_f32(&abs_x[_row][_col], _vr23);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            abs_x[_row][_col] = fabsf(q_pre[_row][_col]);
        }
    }

    // TROWMAX: rowmax = rowmax(abs_x)
    for (int _row = 0; _row < 8; _row++) {
        float _max = abs_x[_row][0];
        for (int _col = 1; _col < 8; _col++) {
            if (abs_x[_row][_col] > _max) _max = abs_x[_row][_col];
        }
        rowmax[_row][0] = _max;}

    // FUSED LOOP (3 ops): scale_127_tile=TEXPANDS(127.0f); scale_quant=TROWEXPANDDIV(scale_127_tile,max_safe); y=TMUL(q_pre,scale_quant)
    float32x4_t _vs24 = vdupq_n_f32(127.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            vst1q_f32(&scale_127_tile[_row][_col], _vs24);
            float32x4_t _v025 = vld1q_f32(&scale_127_tile[_row][_col]);
            float32x4_t _vb27 = vdupq_n_f32(max_safe[_row][0]);
            float32x4_t _vr26 = vdivq_f32(_v025, _vb27);
            vst1q_f32(&scale_quant[_row][_col], _vr26);
            float32x4_t _v28 = vld1q_f32(&q_pre[_row][_col]);
            float32x4_t _v29 = vld1q_f32(&scale_quant[_row][_col]);
            float32x4_t _vr30 = vmulq_f32(_v28, _v29);
            vst1q_f32(&y[_row][_col], _vr30);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            scale_127_tile[_row][_col] = 127.0f;
            scale_quant[_row][_col] = scale_127_tile[_row][_col] / max_safe[_row][0];
            y[_row][_col] = q_pre[_row][_col] * scale_quant[_row][_col];
        }
    }

    // TCVT: y_int8 = (int8_t)lrintf(y) F32->I8 rmode=CAST_RINT
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            y_int8[_i][_j] = (int8_t)lrintf(y[_i][_j]);
        }}

    // FUSED LOOP (1 ops): scale_dequant=TDIVS(max_safe,127.0f)
    float32x4_t _vs31 = vdupq_n_f32(127.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v32 = vld1q_f32(&max_safe[_row][_col]);
            float32x4_t _vr33 = vdivq_f32(_v32, _vs31);
            vst1q_f32(&scale_dequant[_row][_col], _vr33);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            scale_dequant[_row][_col] = max_safe[_row][_col] / 127.0f;
        }
    }

    // FUSED LOOP (1 ops): output_query=TSTORE(y_int8,0,0)
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 8; _col++) {
            output_query[_row * 8 + _col] = y_int8[_row][_col];
        }
    }

    // FUSED LOOP (1 ops): output_query_scale=TSTORE(scale_dequant,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vs34 = vld1q_f32(&scale_dequant[_row][_col]);
            vst1q_f32(&output_query_scale[_row * 1 + _col], _vs34);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            output_query_scale[_row * 1 + _col] = scale_dequant[_row][_col];
        }
    }

    // FUSED LOOP (2 ops): x=TLOAD(input_x,0,0); w_k=TLOAD(input_w_k,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl35 = vld1q_f32(&input_x[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl35);
            float32x4_t _vl36 = vld1q_f32(&input_w_k[_row * 8 + _col]);
            vst1q_f32(&w_k[_row][_col], _vl36);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input_x[_row * 8 + _col];
            w_k[_row][_col] = input_w_k[_row * 8 + _col];
        }
    }

    // TMATMUL: k_lin = x @ w_k
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += x[_i][_k] * w_k[_k][_j];}
            k_lin[_i][_j] = _sum;}}

    // FUSED LOOP (2 ops): gamma_1n=TLOAD(input_layer_norm_gamma,0,0); beta_1n=TLOAD(input_layer_norm_beta,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl37 = vld1q_f32(&input_layer_norm_gamma[_row * 8 + _col]);
            vst1q_f32(&gamma_1n[_row][_col], _vl37);
            float32x4_t _vl38 = vld1q_f32(&input_layer_norm_beta[_row * 8 + _col]);
            vst1q_f32(&beta_1n[_row][_col], _vl38);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            gamma_1n[_row][_col] = input_layer_norm_gamma[_row * 8 + _col];
            beta_1n[_row][_col] = input_layer_norm_beta[_row * 8 + _col];
        }
    }

    // TCOLEXPAND: gamma = broadcast first row of gamma_1n -> dst[i,j]=src[0,j]
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            gamma[_i][_j] = gamma_1n[0][_j];
        }}

    // TCOLEXPAND: beta = broadcast first row of beta_1n -> dst[i,j]=src[0,j]
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            beta[_i][_j] = beta_1n[0][_j];
        }}

    // TROWSUM: sum = rowsum(k_lin)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += k_lin[_row][_col];
        }
        sum[_row][0] = _sum;}

    // FUSED LOOP (1 ops): mean=TDIVS(sum,inv_colsf)
    float32x4_t _vs39 = vdupq_n_f32(inv_colsf);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v40 = vld1q_f32(&sum[_row][_col]);
            float32x4_t _vr41 = vdivq_f32(_v40, _vs39);
            vst1q_f32(&mean[_row][_col], _vr41);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            mean[_row][_col] = sum[_row][_col] / inv_colsf;
        }
    }

    // FUSED LOOP (2 ops): diff=TROWEXPANDSUB(k_lin,mean); sq=TMUL(diff,diff)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v042 = vld1q_f32(&k_lin[_row][_col]);
            float32x4_t _vb44 = vdupq_n_f32(mean[_row][0]);
            float32x4_t _vr43 = vsubq_f32(_v042, _vb44);
            vst1q_f32(&diff[_row][_col], _vr43);
            float32x4_t _v45 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _v46 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vr47 = vmulq_f32(_v45, _v46);
            vst1q_f32(&sq[_row][_col], _vr47);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            diff[_row][_col] = k_lin[_row][_col] - mean[_row][0];
            sq[_row][_col] = diff[_row][_col] * diff[_row][_col];
        }
    }

    // TROWSUM: var_sum = rowsum(sq)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += sq[_row][_col];
        }
        var_sum[_row][0] = _sum;}

    // FUSED LOOP (3 ops): var=TDIVS(var_sum,inv_colsf); var_eps=TADDS(var,epsf); std=TSQRT(var_eps)
    float32x4_t _vs48 = vdupq_n_f32(inv_colsf);
    float32x4_t _vs49 = vdupq_n_f32(epsf);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v50 = vld1q_f32(&var_sum[_row][_col]);
            float32x4_t _vr51 = vdivq_f32(_v50, _vs48);
            vst1q_f32(&var[_row][_col], _vr51);
            float32x4_t _v52 = vld1q_f32(&var[_row][_col]);
            float32x4_t _vr53 = vaddq_f32(_v52, _vs49);
            vst1q_f32(&var_eps[_row][_col], _vr53);
            float32x4_t _v54 = vld1q_f32(&var_eps[_row][_col]);
            float32x4_t _vr55 = vsqrtq_f32(_v54);
            vst1q_f32(&std[_row][_col], _vr55);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            var[_row][_col] = var_sum[_row][_col] / inv_colsf;
            var_eps[_row][_col] = var[_row][_col] + epsf;
            std[_row][_col] = sqrtf(var_eps[_row][_col]);
        }
    }

    // FUSED LOOP (3 ops): k_ln=TROWEXPANDDIV(diff,std); k_ln_scaled=TMUL(k_ln,gamma); k_ln=TADD(k_ln_scaled,beta)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v056 = vld1q_f32(&diff[_row][_col]);
            float32x4_t _vb58 = vdupq_n_f32(std[_row][0]);
            float32x4_t _vr57 = vdivq_f32(_v056, _vb58);
            vst1q_f32(&k_ln[_row][_col], _vr57);
            float32x4_t _v59 = vld1q_f32(&k_ln[_row][_col]);
            float32x4_t _v60 = vld1q_f32(&gamma[_row][_col]);
            float32x4_t _vr61 = vmulq_f32(_v59, _v60);
            vst1q_f32(&k_ln_scaled[_row][_col], _vr61);
            float32x4_t _v62 = vld1q_f32(&k_ln_scaled[_row][_col]);
            float32x4_t _v63 = vld1q_f32(&beta[_row][_col]);
            float32x4_t _vr64 = vaddq_f32(_v62, _v63);
            vst1q_f32(&k_ln[_row][_col], _vr64);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            k_ln[_row][_col] = diff[_row][_col] / std[_row][0];
            k_ln_scaled[_row][_col] = k_ln[_row][_col] * gamma[_row][_col];
            k_ln[_row][_col] = k_ln_scaled[_row][_col] + beta[_row][_col];
        }
    }

    // TEXTRACT: k_rope = extract(k_ln, [0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 4; _col++) {
            k_rope[_row][_col] = k_ln[(0) + _row][(0) + _col];
        }}

    // TEXTRACT: k_nope = extract(k_ln, [0, 4])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 4; _col++) {
            k_nope[_row][_col] = k_ln[(0) + _row][(4) + _col];
        }}

    // FUSED LOOP (2 ops): cos=TLOAD(input_cos,0,0); sin=TLOAD(input_sin,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _vl65 = vld1q_f32(&input_cos[_row * 4 + _col]);
            vst1q_f32(&cos[_row][_col], _vl65);
            float32x4_t _vl66 = vld1q_f32(&input_sin[_row * 4 + _col]);
            vst1q_f32(&sin[_row][_col], _vl66);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            cos[_row][_col] = input_cos[_row * 4 + _col];
            sin[_row][_col] = input_sin[_row * 4 + _col];
        }
    }

    // TEXTRACT: left = extract(k_rope, [0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 2; _col++) {
            left[_row][_col] = k_rope[(0) + _row][(0) + _col];
        }}

    // TEXTRACT: right = extract(k_rope, [0, 2])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 2; _col++) {
            right[_row][_col] = k_rope[(0) + _row][(2) + _col];
        }}

    // FUSED LOOP (1 ops): right_neg=TNEG(right)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 2; _col += 4) {
            float32x4_t _v67 = vld1q_f32(&right[_row][_col]);
            float32x4_t _vr68 = vnegq_f32(_v67);
            vst1q_f32(&right_neg[_row][_col], _vr68);
        }
        // Scalar cleanup
        for (; _col < 2; _col++) {
            right_neg[_row][_col] = -right[_row][_col];
        }
    }

    // TCONCAT(axis=1): rotated = [ right_neg | left ]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 2; _col++) {
            rotated[_row][_col] = right_neg[_row][_col];
        }
        for (int _col = 0; _col < 2; _col++) {
            rotated[_row][2 + _col] = left[_row][_col];
        }}

    // FUSED LOOP (3 ops): part1=TMUL(k_rope,cos); part2=TMUL(rotated,sin); q_rope_rotated=TADD(part1,part2)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4; _col += 4) {
            float32x4_t _v69 = vld1q_f32(&k_rope[_row][_col]);
            float32x4_t _v70 = vld1q_f32(&cos[_row][_col]);
            float32x4_t _vr71 = vmulq_f32(_v69, _v70);
            vst1q_f32(&part1[_row][_col], _vr71);
            float32x4_t _v72 = vld1q_f32(&rotated[_row][_col]);
            float32x4_t _v73 = vld1q_f32(&sin[_row][_col]);
            float32x4_t _vr74 = vmulq_f32(_v72, _v73);
            vst1q_f32(&part2[_row][_col], _vr74);
            float32x4_t _v75 = vld1q_f32(&part1[_row][_col]);
            float32x4_t _v76 = vld1q_f32(&part2[_row][_col]);
            float32x4_t _vr77 = vaddq_f32(_v75, _v76);
            vst1q_f32(&q_rope_rotated[_row][_col], _vr77);
        }
        // Scalar cleanup
        for (; _col < 4; _col++) {
            part1[_row][_col] = k_rope[_row][_col] * cos[_row][_col];
            part2[_row][_col] = rotated[_row][_col] * sin[_row][_col];
            q_rope_rotated[_row][_col] = part1[_row][_col] + part2[_row][_col];
        }
    }

    // TCONCAT(axis=1): q = [ q_rope_rotated | k_nope ]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 4; _col++) {
            q[_row][_col] = q_rope_rotated[_row][_col];
        }
        for (int _col = 0; _col < 4; _col++) {
            q[_row][4 + _col] = k_nope[_row][_col];
        }}

    // FUSED LOOP (1 ops): hadamard_q=TLOAD(input_hadamard_k,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl78 = vld1q_f32(&input_hadamard_k[_row * 8 + _col]);
            vst1q_f32(&hadamard_q[_row][_col], _vl78);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            hadamard_q[_row][_col] = input_hadamard_k[_row * 8 + _col];
        }
    }

    // TMATMUL: k_ln = q @ hadamard_q
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += q[_i][_k] * hadamard_q[_k][_j];}
            k_ln[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): abs_x=TABS(k_ln)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v79 = vld1q_f32(&k_ln[_row][_col]);
            float32x4_t _vr80 = vabsq_f32(_v79);
            vst1q_f32(&abs_x[_row][_col], _vr80);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            abs_x[_row][_col] = fabsf(k_ln[_row][_col]);
        }
    }

    // TROWMAX: rowmax = rowmax(abs_x)
    for (int _row = 0; _row < 8; _row++) {
        float _max = abs_x[_row][0];
        for (int _col = 1; _col < 8; _col++) {
            if (abs_x[_row][_col] > _max) _max = abs_x[_row][_col];
        }
        rowmax[_row][0] = _max;}

    // FUSED LOOP (3 ops): scale_127_tile=TEXPANDS(127.0f); scale_quant=TROWEXPANDDIV(scale_127_tile,max_safe); y=TMUL(k_ln,scale_quant)
    float32x4_t _vs81 = vdupq_n_f32(127.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            vst1q_f32(&scale_127_tile[_row][_col], _vs81);
            float32x4_t _v082 = vld1q_f32(&scale_127_tile[_row][_col]);
            float32x4_t _vb84 = vdupq_n_f32(max_safe[_row][0]);
            float32x4_t _vr83 = vdivq_f32(_v082, _vb84);
            vst1q_f32(&scale_quant[_row][_col], _vr83);
            float32x4_t _v85 = vld1q_f32(&k_ln[_row][_col]);
            float32x4_t _v86 = vld1q_f32(&scale_quant[_row][_col]);
            float32x4_t _vr87 = vmulq_f32(_v85, _v86);
            vst1q_f32(&y[_row][_col], _vr87);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            scale_127_tile[_row][_col] = 127.0f;
            scale_quant[_row][_col] = scale_127_tile[_row][_col] / max_safe[_row][0];
            y[_row][_col] = k_ln[_row][_col] * scale_quant[_row][_col];
        }
    }

    // TCVT: y_int8 = (int8_t)lrintf(y) F32->I8 rmode=CAST_RINT
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            y_int8[_i][_j] = (int8_t)lrintf(y[_i][_j]);
        }}

    // FUSED LOOP (1 ops): scale_dequant=TDIVS(max_safe,127.0f)
    float32x4_t _vs88 = vdupq_n_f32(127.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _v89 = vld1q_f32(&max_safe[_row][_col]);
            float32x4_t _vr90 = vdivq_f32(_v89, _vs88);
            vst1q_f32(&scale_dequant[_row][_col], _vr90);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            scale_dequant[_row][_col] = max_safe[_row][_col] / 127.0f;
        }
    }

    // FUSED LOOP (1 ops): cache_index=TLOAD(cache_index_src,0,0)
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            cache_index[_row][_col] = cache_index_src[_row * 1 + _col];
        }
    }

    // SCATTER_UPDATE: k_cache[cache_index[i,0]*8+j] = y_int8[i,j] (axis=-2)
    for (int _i = 0; _i < 8; _i++) {
        int _row = (int)cache_index[_i][0];
        for (int _j = 0; _j < 8; _j++) {
            k_cache[_row * 8 + _j] = y_int8[_i][_j];
        }}

    // SCATTER_UPDATE: k_scale_cache[cache_index[i,0]*1+j] = scale_dequant[i,j] (axis=-2)
    for (int _i = 0; _i < 8; _i++) {
        int _row = (int)cache_index[_i][0];
        for (int _j = 0; _j < 1; _j++) {
            k_scale_cache[_row * 1 + _j] = scale_dequant[_i][_j];
        }}

    // FUSED LOOP (1 ops): w_proj=TLOAD(input_w_proj,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl91 = vld1q_f32(&input_w_proj[_row * 8 + _col]);
            vst1q_f32(&w_proj[_row][_col], _vl91);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            w_proj[_row][_col] = input_w_proj[_row * 8 + _col];
        }
    }

    // TMATMUL: weights = x @ w_proj
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += x[_i][_k] * w_proj[_k][_j];}
            weights[_i][_j] = _sum;}}

    // FUSED LOOP (2 ops): weights_scaled=TDIVS(weights,scalef); output_weights=TSTORE(weights_scaled,0,0)
    float32x4_t _vs92 = vdupq_n_f32(scalef);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v93 = vld1q_f32(&weights[_row][_col]);
            float32x4_t _vr94 = vdivq_f32(_v93, _vs92);
            vst1q_f32(&weights_scaled[_row][_col], _vr94);
            float32x4_t _vs95 = vld1q_f32(&weights_scaled[_row][_col]);
            vst1q_f32(&output_weights[_row * 8 + _col], _vs95);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            weights_scaled[_row][_col] = weights[_row][_col] / scalef;
            output_weights[_row * 8 + _col] = weights_scaled[_row][_col];
        }
    }

}

#ifdef PTO_CPU_SMOKE_RUNNER
#include <stddef.h>
const char* pto_program_name() { return "indexer_prolog"; }
enum { kPtoNumMemrefs = 19 };
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "input_x",
    "input_q_norm",
    "input_q_norm_scale",
    "input_w",
    "input_w_qb_scale",
    "input_w_k",
    "input_w_proj",
    "input_cos",
    "input_sin",
    "input_hadamard_q",
    "input_hadamard_k",
    "input_layer_norm_gamma",
    "input_layer_norm_beta",
    "cache_index_src",
    "k_cache",
    "k_scale_cache",
    "output_query",
    "output_query_scale",
    "output_weights",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(256),
    (size_t)(64),
    (size_t)(32),
    (size_t)(64),
    (size_t)(32),
    (size_t)(256),
    (size_t)(256),
    (size_t)(128),
    (size_t)(128),
    (size_t)(256),
    (size_t)(256),
    (size_t)(32),
    (size_t)(32),
    (size_t)(64),
    (size_t)(1),
    (size_t)(4),
    (size_t)(64),
    (size_t)(32),
    (size_t)(256),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "i8",
    "f32",
    "i8",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "i64",
    "i8",
    "f32",
    "i8",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(1),
    (size_t)(4),
    (size_t)(1),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(8),
    (size_t)(1),
    (size_t)(4),
    (size_t)(1),
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
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
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
    indexer_prolog((float*)args[0], (int8_t*)args[1], (float*)args[2], (int8_t*)args[3], (float*)args[4], (float*)args[5], (float*)args[6], (float*)args[7], (float*)args[8], (float*)args[9], (float*)args[10], (float*)args[11], (float*)args[12], (int64_t*)args[13], (int8_t*)args[14], (float*)args[15], (int8_t*)args[16], (float*)args[17], (float*)args[18]);
}
#endif  // PTO_CPU_SMOKE_RUNNER