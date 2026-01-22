// PTO Program: tile_exp_256
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_exp_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 262,144 bytes (256.0 KB)
//   Total capacity (w/ reuse): 262,144 bytes (256.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               256x128    f32    131072   [  1,   2]           -
//   x                    256x128    f32    131072   [  0,   1]           -
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
__device__ float result[256][128];

__global__ void tile_exp_256_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); result=TEXP(...); output=TSTORE(...)
    if (_row < 256 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
        result[_row][_col] = __expf(x[_row][_col]);
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void tile_exp_256(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tile_exp_256_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}