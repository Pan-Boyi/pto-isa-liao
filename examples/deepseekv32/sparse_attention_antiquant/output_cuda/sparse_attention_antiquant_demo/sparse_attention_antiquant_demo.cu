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

// Auto-generated CUDA code from PTO ISA Compiler
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__device__ float q[4][4];
__device__ float k[4][4];
__device__ float v[4][4];
__device__ float k_t[4][4];
__device__ float scores[4][4];
__device__ float row_max[4][1];
__device__ float shifted[4][4];
__device__ float exp_scores[4][4];
__device__ float sum_exp[4][1];
__device__ float probs[4][4];
__device__ float out[4][4];

__global__ void sparse_attention_antiquant_demo_kernel(float* q_mem, float* k_mem, float* v_mem, float* out_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 3 loop overheads saved

    // FUSED (3 ops): q=TLOAD(...); k=TLOAD(...); v=TLOAD(...)
    if (_row < 4 && _col < 4) {
        q[_row][_col] = q_mem[_row * 4 + _col];
        k[_row][_col] = k_mem[_row * 4 + _col];
        v[_row][_col] = v_mem[_row * 4 + _col];
    }

    // TMATMUL: scores = q @ k_t
    if (_row < 4 && _col < 4) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += q[_row][_k] * k_t[_k][_col];
        scores[_row][_col] = _sum;}

    // TROWMAX: row_max = rowmax(scores)
    if (_col == 0 && _row < 4) {
        float _max = scores[_row][0];
        for (int _c = 1; _c < 4; _c++) if (scores[_row][_c] > _max) _max = scores[_row][_c];
        row_max[_row][0] = _max;}

    // FUSED (2 ops): shifted=TROWEXPANDSUB(...); exp_scores=TEXP(...)
    if (_row < 4 && _col < 4) {
        shifted[_row][_col] = scores[_row][_col] - row_max[_row][0];
        exp_scores[_row][_col] = __expf(shifted[_row][_col]);
    }

    // TROWSUM: sum_exp = rowsum(exp_scores)
    if (_col == 0 && _row < 4) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 4; _c++) _sum += exp_scores[_row][_c];
        sum_exp[_row][0] = _sum;}

    // FUSED (1 ops): probs=TROWEXPANDDIV(...)
    if (_row < 4 && _col < 4) {
        probs[_row][_col] = exp_scores[_row][_col] / sum_exp[_row][0];
    }

    // TMATMUL: out = probs @ v
    if (_row < 4 && _col < 4) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += probs[_row][_k] * v[_k][_col];
        out[_row][_col] = _sum;}

    // FUSED (1 ops): out_mem=TSTORE(...)
    if (_row < 4 && _col < 4) {
        out_mem[_row * 4 + _col] = out[_row][_col];
    }

}

void sparse_attention_antiquant_demo(float* q_mem, float* k_mem, float* v_mem, float* out_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    sparse_attention_antiquant_demo_kernel<<<grid, block>>>(q_mem, k_mem, v_mem, out_mem);
    cudaDeviceSynchronize();
}