// PTO Program: lightning_indexer_quant_demo
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: lightning_indexer_quant_demo
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 256 bytes (0.2 KB)
//   Total capacity (w/ reuse): 192 bytes (0.2 KB)
//   Reuse savings:            64 bytes (25.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   k                    4x4        f32        64   [  1,   2]           -
//   k_t                  4x4        f32        64   [  2,  -1]           -
//   q                    4x4        f32        64   [  0,  -1]           -
//   scores               4x4        f32        64   [  3,   4]           <- k
//
// BUFFER REUSE MAP:
//   scores reuses buffer of k
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
__device__ float k_t[4][4];
__device__ float scores[4][4];

__global__ void lightning_indexer_quant_demo_kernel(float* q_mem, float* k_mem, float* scores_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (2 ops): q=TLOAD(...); k=TLOAD(...)
    if (_row < 4 && _col < 4) {
        q[_row][_col] = q_mem[_row * 4 + _col];
        k[_row][_col] = k_mem[_row * 4 + _col];
    }

    // TMATMUL: scores = q @ k_t
    if (_row < 4 && _col < 4) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += q[_row][_k] * k_t[_k][_col];
        scores[_row][_col] = _sum;}

    // FUSED (1 ops): scores_mem=TSTORE(...)
    if (_row < 4 && _col < 4) {
        scores_mem[_row * 4 + _col] = scores[_row][_col];
    }

}

void lightning_indexer_quant_demo(float* q_mem, float* k_mem, float* scores_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    lightning_indexer_quant_demo_kernel<<<grid, block>>>(q_mem, k_mem, scores_mem);
    cudaDeviceSynchronize();
}