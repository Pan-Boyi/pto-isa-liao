// PTO Program: rowsum
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rowsum
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 288 bytes (0.3 KB)
//   Total capacity (w/ reuse): 288 bytes (0.3 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x1        f32        32   [  1,   2]           -
//   x                    8x8        f32       256   [  0,   1]           -
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
__device__ float result[8][1];

__global__ void rowsum_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
    }

    // TROWSUM: result = rowsum(x)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += x[_row][_c];
        result[_row][0] = _sum;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 1) {
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void rowsum(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    rowsum_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}