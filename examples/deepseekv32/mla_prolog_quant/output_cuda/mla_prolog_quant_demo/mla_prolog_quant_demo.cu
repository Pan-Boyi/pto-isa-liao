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

__device__ float x[4][4];
__device__ float w_dq[4][4];
__device__ float w_uq_qr[4][4];
__device__ float x_proj[4][4];
__device__ float x_sq[4][4];
__device__ float sum_sq[4][1];
__device__ float mean_sq[4][1];
__device__ float mean_eps[4][1];
__device__ float rms[4][1];
__device__ float x_norm[4][4];
__device__ float q_out[4][4];

__global__ void mla_prolog_quant_demo_kernel(float* x_mem, float* w_dq_mem, float* w_uq_qr_mem, float* q_out_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); w_dq=TLOAD(...); w_uq_qr=TLOAD(...)
    if (_row < 4 && _col < 4) {
        x[_row][_col] = x_mem[_row * 4 + _col];
        w_dq[_row][_col] = w_dq_mem[_row * 4 + _col];
        w_uq_qr[_row][_col] = w_uq_qr_mem[_row * 4 + _col];
    }

    // TMATMUL: x_proj = x @ w_dq
    if (_row < 4 && _col < 4) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += x[_row][_k] * w_dq[_k][_col];
        x_proj[_row][_col] = _sum;}

    // FUSED (1 ops): x_sq=TMUL(...)
    if (_row < 4 && _col < 4) {
        x_sq[_row][_col] = x_proj[_row][_col] * x_proj[_row][_col];
    }

    // TROWSUM: sum_sq = rowsum(x_sq)
    if (_col == 0 && _row < 4) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 4; _c++) _sum += x_sq[_row][_c];
        sum_sq[_row][0] = _sum;}

    // FUSED (3 ops): mean_sq=TDIVS(...); mean_eps=TADDS(...); rms=TSQRT(...)
    if (_row < 4 && _col < 1) {
        mean_sq[_row][_col] = sum_sq[_row][_col] / 4.0f;
        mean_eps[_row][_col] = mean_sq[_row][_col] + 1e-06f;
        rms[_row][_col] = __fsqrt_rn(mean_eps[_row][_col]);
    }

    // FUSED (1 ops): x_norm=TROWEXPANDDIV(...)
    if (_row < 4 && _col < 4) {
        x_norm[_row][_col] = x_proj[_row][_col] / rms[_row][0];
    }

    // TMATMUL: q_out = x_norm @ w_uq_qr
    if (_row < 4 && _col < 4) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += x_norm[_row][_k] * w_uq_qr[_k][_col];
        q_out[_row][_col] = _sum;}

    // FUSED (1 ops): q_out_mem=TSTORE(...)
    if (_row < 4 && _col < 4) {
        q_out_mem[_row * 4 + _col] = q_out[_row][_col];
    }

}

void mla_prolog_quant_demo(float* x_mem, float* w_dq_mem, float* w_uq_qr_mem, float* q_out_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    mla_prolog_quant_demo_kernel<<<grid, block>>>(x_mem, w_dq_mem, w_uq_qr_mem, q_out_mem);
    cudaDeviceSynchronize();
}