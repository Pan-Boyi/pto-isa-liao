// PTO Program: softmax_tile_256
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: softmax_tile_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 526,336 bytes (514.0 KB)
//   Total capacity (w/ reuse): 263,168 bytes (257.0 KB)
//   Reuse savings:            263,168 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_x                256x128    f32    131072   [  3,   5]           <- x
//   result               256x128    f32    131072   [  5,   6]           <- x_shifted
//   row_max              256x1      f32      1024   [  1,   2]           -
//   row_sum              256x1      f32      1024   [  4,   5]           <- row_max
//   x                    256x128    f32    131072   [  0,   2]           -
//   x_shifted            256x128    f32    131072   [  2,   3]           -
//
// BUFFER REUSE MAP:
//   exp_x reuses buffer of x
//   row_sum reuses buffer of row_max
//   result reuses buffer of x_shifted
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

__device__ float x[256][128];
__device__ float row_max[256][1];
__device__ float x_shifted[256][128];
__device__ float exp_x[256][128];
__device__ float row_sum[256][1];
__device__ float result[256][128];

__global__ void softmax_tile_256_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 256 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
    }

    // TROWMAX: row_max = rowmax(x)
    if (_col == 0 && _row < 256) {
        float _max = x[_row][0];
        for (int _c = 1; _c < 128; _c++) if (x[_row][_c] > _max) _max = x[_row][_c];
        row_max[_row][0] = _max;}

    // FUSED (2 ops): x_shifted=TROWEXPANDSUB(...); exp_x=TEXP(...)
    if (_row < 256 && _col < 128) {
        x_shifted[_row][_col] = x[_row][_col] - row_max[_row][0];
        exp_x[_row][_col] = __expf(x_shifted[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(exp_x)
    if (_col == 0 && _row < 256) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 128; _c++) _sum += exp_x[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (2 ops): result=TROWEXPANDDIV(...); output=TSTORE(...)
    if (_row < 256 && _col < 128) {
        result[_row][_col] = exp_x[_row][_col] / row_sum[_row][0];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void softmax_tile_256(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    softmax_tile_256_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}