// PTO Program: flash_attn_normalize
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_normalize
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 65,792 bytes (64.2 KB)
//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_vec                64x1       f32       256   [  1,   2]           -
//   o_block              64x128     f32     32768   [  0,   2]           -
//   o_final              64x128     f32     32768   [  2,   3]           -
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_normalize(float* input_o, float* input_l, float* output) {
    float o_block[64][128];
    float l_vec[64][1];
    float o_final[64][128];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (1 ops): o_block=TLOAD(input_o,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_o[_row * 128 + _col]);
            vst1q_f32(&o_block[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            o_block[_row][_col] = input_o[_row * 128 + _col];
        }
    }

    // FUSED LOOP (1 ops): l_vec=TLOAD(input_l,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_l[_row * 1 + _col]);
            vst1q_f32(&l_vec[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            l_vec[_row][_col] = input_l[_row * 1 + _col];
        }
    }

    // FUSED LOOP (2 ops): o_final=TROWEXPANDDIV(o_block,l_vec); output=TSTORE(o_final,0,0)
    for (int _row = 0; _row < 64; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 128; _col += 4) {
            float32x4_t _v02 = vld1q_f32(&o_block[_row][_col]);
            float32x4_t _vb4 = vdupq_n_f32(l_vec[_row][0]);
            float32x4_t _vr3 = vdivq_f32(_v02, _vb4);
            vst1q_f32(&o_final[_row][_col], _vr3);
            float32x4_t _vs5 = vld1q_f32(&o_final[_row][_col]);
            vst1q_f32(&output[_row * 128 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 128; _col++) {
            o_final[_row][_col] = o_block[_row][_col] / l_vec[_row][0];
            output[_row * 128 + _col] = o_final[_row][_col];
        }
    }

}