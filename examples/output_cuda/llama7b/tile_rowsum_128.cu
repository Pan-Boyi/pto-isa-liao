// PTO Program: tile_rowsum_128
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_rowsum_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 66,048 bytes (64.5 KB)
//   Total capacity (w/ reuse): 66,048 bytes (64.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               128x1      f32       512   [  1,   2]           -
//   x                    128x128    f32     65536   [  0,   1]           -
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
__device__ float result[128][1];

__global__ void tile_rowsum_128_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 0 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 128 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
    }

    // TROWSUM: result = rowsum(x)
    if (_col == 0 && _row < 128) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 128; _c++) _sum += x[_row][_c];
        result[_row][0] = _sum;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 128 && _col < 1) {
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void tile_rowsum_128(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tile_rowsum_128_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}