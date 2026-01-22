// PTO Program: rmsnorm_tile_128
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rmsnorm_tile_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 329,216 bytes (321.5 KB)
//   Total capacity (w/ reuse): 197,632 bytes (193.0 KB)
//   Reuse savings:            131,584 bytes (40.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   gamma                128x128    f32     65536   [  1,  10]           -
//   result               128x128    f32     65536   [ 10,  11]           <- x
//   row_mean             128x1      f32       512   [  5,   8]           -
//   row_rsqrt            128x1      f32       512   [  8,   9]           <- row_sum
//   row_sum              128x1      f32       512   [  3,   5]           -
//   x                    128x128    f32     65536   [  0,   9]           -
//   x_norm               128x128    f32     65536   [  9,  10]           <- x_sq
//   x_sq                 128x128    f32     65536   [  2,   3]           -
//
// BUFFER REUSE MAP:
//   row_rsqrt reuses buffer of row_sum
//   x_norm reuses buffer of x_sq
//   result reuses buffer of x
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

__device__ float x[128][128];
__device__ float x_sq[128][128];
__device__ float row_sum[128][1];
__device__ float row_mean[128][1];
__device__ float row_rsqrt[128][1];
__device__ float x_norm[128][128];
__device__ float gamma[128][128];
__device__ float result[128][128];

__global__ void rmsnorm_tile_128_kernel(float* input, float* weights, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); gamma=TLOAD(...); x_sq=TMUL(...)
    if (_row < 128 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
        gamma[_row][_col] = weights[_row * 128 + _col];
        x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
    }

    // TROWSUM: row_sum = rowsum(x_sq)
    if (_col == 0 && _row < 128) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 128; _c++) _sum += x_sq[_row][_c];
        row_sum[_row][0] = _sum;}

    int inv_cols = 0.0078125;

    // FUSED (1 ops): row_mean=TMULS(...)
    if (_row < 128 && _col < 1) {
        row_mean[_row][_col] = row_sum[_row][_col] * inv_colsf;
    }

    int eps = 1e-05;

    // FUSED (2 ops): row_mean=TADDS(...); row_rsqrt=TRSQRT(...)
    if (_row < 128 && _col < 1) {
        row_mean[_row][_col] = row_mean[_row][_col] + epsf;
        row_rsqrt[_row][_col] = __frsqrt_rn(row_mean[_row][_col]);
    }

    // FUSED (3 ops): x_norm=TROWEXPANDMUL(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 128 && _col < 128) {
        x_norm[_row][_col] = x[_row][_col] * row_rsqrt[_row][0];
        result[_row][_col] = x_norm[_row][_col] * gamma[_row][_col];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void rmsnorm_tile_128(float* input, float* weights, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    rmsnorm_tile_128_kernel<<<grid, block>>>(input, weights, output, eps, inv_cols);
    cudaDeviceSynchronize();
}