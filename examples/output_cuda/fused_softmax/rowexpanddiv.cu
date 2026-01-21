// PTO Program: rowexpanddiv
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rowexpanddiv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 544 bytes (0.5 KB)
//   Total capacity (w/ reuse): 544 bytes (0.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x8        f32       256   [  2,   3]           -
//   row_vals             8x1        f32        32   [  1,   2]           -
//   x                    8x8        f32       256   [  0,   2]           -
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

__device__ float x[8][8];
__device__ float row_vals[8][1];
__device__ float result[8][8];

__global__ void rowexpanddiv_kernel(float* input_x, float* input_row, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input_x[_row * 8 + _col];
    }

    // FUSED (1 ops): row_vals=TLOAD(...)
    if (_row < 8 && _col < 1) {
        row_vals[_row][_col] = input_row[_row * 1 + _col];
    }

    // FUSED (2 ops): result=TROWEXPANDDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = x[_row][_col] / row_vals[_row][0];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void rowexpanddiv(float* input_x, float* input_row, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    rowexpanddiv_kernel<<<grid, block>>>(input_x, input_row, output);
    cudaDeviceSynchronize();
}