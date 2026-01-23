// PTO Program: mla_indexer_prolog_quant_demo
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: mla_indexer_prolog_quant_demo
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 384 bytes (0.4 KB)
//   Total capacity (w/ reuse): 384 bytes (0.4 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   q_norm               4x4        f32        64   [  4,  -1]           -
//   q_out                4x4        f32        64   [  5,   7]           -
//   w_dq                 4x4        f32        64   [  1,  -1]           -
//   w_proj               4x2        f32        32   [  3,  -1]           -
//   w_qb                 4x4        f32        64   [  2,  -1]           -
//   weights              4x2        f32        32   [  6,   8]           -
//   x                    4x4        f32        64   [  0,  -1]           -
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
__device__ float w_qb[4][4];
__device__ float w_proj[4][2];
__device__ float q_norm[4][4];
__device__ float q_out[4][4];
__device__ float weights[4][2];

__global__ void mla_indexer_prolog_quant_demo_kernel(float* x_mem, float* w_dq_mem, float* w_qb_mem, float* w_proj_mem, float* q_out_mem, float* weights_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (3 ops): x=TLOAD(...); w_dq=TLOAD(...); w_qb=TLOAD(...)
    if (_row < 4 && _col < 4) {
        x[_row][_col] = x_mem[_row * 4 + _col];
        w_dq[_row][_col] = w_dq_mem[_row * 4 + _col];
        w_qb[_row][_col] = w_qb_mem[_row * 4 + _col];
    }

    // FUSED (1 ops): w_proj=TLOAD(...)
    if (_row < 4 && _col < 2) {
        w_proj[_row][_col] = w_proj_mem[_row * 2 + _col];
    }

    // TMATMUL: q_norm = x @ w_dq
    if (_row < 4 && _col < 4) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += x[_row][_k] * w_dq[_k][_col];
        q_norm[_row][_col] = _sum;}

    // TMATMUL: q_out = q_norm @ w_qb
    if (_row < 4 && _col < 4) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += q_norm[_row][_k] * w_qb[_k][_col];
        q_out[_row][_col] = _sum;}

    // TMATMUL: weights = x @ w_proj
    if (_row < 4 && _col < 2) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += x[_row][_k] * w_proj[_k][_col];
        weights[_row][_col] = _sum;}

    // FUSED (1 ops): q_out_mem=TSTORE(...)
    if (_row < 4 && _col < 4) {
        q_out_mem[_row * 4 + _col] = q_out[_row][_col];
    }

    // FUSED (1 ops): weights_mem=TSTORE(...)
    if (_row < 4 && _col < 2) {
        weights_mem[_row * 2 + _col] = weights[_row][_col];
    }

}

void mla_indexer_prolog_quant_demo(float* x_mem, float* w_dq_mem, float* w_qb_mem, float* w_proj_mem, float* q_out_mem, float* weights_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    mla_indexer_prolog_quant_demo_kernel<<<grid, block>>>(x_mem, w_dq_mem, w_qb_mem, w_proj_mem, q_out_mem, weights_mem);
    cudaDeviceSynchronize();
}