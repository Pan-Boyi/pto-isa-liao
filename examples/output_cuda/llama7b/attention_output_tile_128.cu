// PTO Program: attention_output_tile_128
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: attention_output_tile_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 196,608 bytes (192.0 KB)
//   Total capacity (w/ reuse): 196,608 bytes (192.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               128x128    f32     65536   [  2,   3]           -
//   v                    128x128    f32     65536   [  1,  -1]           -
//   weights              128x128    f32     65536   [  0,  -1]           -
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

__device__ float weights[128][128];
__device__ float v[128][128];
__device__ float result[128][128];

__global__ void attention_output_tile_128_kernel(float* input_weights, float* input_v, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (2 ops): weights=TLOAD(...); v=TLOAD(...)
    if (_row < 128 && _col < 128) {
        weights[_row][_col] = input_weights[_row * 128 + _col];
        v[_row][_col] = input_v[_row * 128 + _col];
    }

    // TMATMUL: result = weights @ v
    if (_row < 128 && _col < 128) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 128; _k++) _sum += weights[_row][_k] * v[_k][_col];
        result[_row][_col] = _sum;}

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 128 && _col < 128) {
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void attention_output_tile_128(float* input_weights, float* input_v, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    attention_output_tile_128_kernel<<<grid, block>>>(input_weights, input_v, output);
    cudaDeviceSynchronize();
}