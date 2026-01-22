// PTO Program: tile_muls_128
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_muls_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 131,072 bytes (128.0 KB)
//   Total capacity (w/ reuse): 131,072 bytes (128.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   a                    128x128    f32     65536   [  0,   1]           -
//   result               128x128    f32     65536   [  1,   2]           -
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

__device__ float a[128][128];
__device__ float result[128][128];

__global__ void tile_muls_128_kernel(float* input, float* output, float scale) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): a=TLOAD(...); result=TMULS(...); output=TSTORE(...)
    if (_row < 128 && _col < 128) {
        a[_row][_col] = input[_row * 128 + _col];
        result[_row][_col] = a[_row][_col] * scalef;
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void tile_muls_128(float* input, float* output, float scale) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tile_muls_128_kernel<<<grid, block>>>(input, output, scale);
    cudaDeviceSynchronize();
}