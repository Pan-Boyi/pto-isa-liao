// PTO Program: lightning_indexer_prolog_quant_demo
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: lightning_indexer_prolog_quant_demo
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 576 bytes (0.6 KB)
//   Total capacity (w/ reuse): 512 bytes (0.5 KB)
//   Reuse savings:            64 bytes (11.1%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   q_matmul             4x4        f32        64   [  6,   7]           -
//   q_norm               4x4        f32        64   [  0,  -1]           -
//   q_norm_scale         4x1        f32        16   [  1,   7]           -
//   q_out                4x4        f32        64   [  9,  12]           -
//   q_scaled_row         4x4        f32        64   [  7,   9]           -
//   w_proj               4x2        f32        32   [  5,  -1]           -
//   w_qb                 4x4        f32        64   [  2,  -1]           -
//   w_qb_scale           1x4        f32        16   [  3,   8]           -
//   w_qb_scale_expand    4x4        f32        64   [  8,   9]           <- q_matmul
//   weights_out          4x2        f32        32   [ 11,  13]           -
//   weights_raw          4x2        f32        32   [ 10,  11]           -
//   x                    4x4        f32        64   [  4,  -1]           -
//
// BUFFER REUSE MAP:
//   w_qb_scale_expand reuses buffer of q_matmul
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

__device__ float q_norm[4][4];
__device__ float q_norm_scale[4][1];
__device__ float w_qb[4][4];
__device__ float w_qb_scale[1][4];
__device__ float w_qb_scale_expand[4][4];
__device__ float q_matmul[4][4];
__device__ float q_scaled_row[4][4];
__device__ float q_out[4][4];
__device__ float x[4][4];
__device__ float w_proj[4][2];
__device__ float weights_raw[4][2];
__device__ float weights_out[4][2];

__global__ void lightning_indexer_prolog_quant_demo_kernel(float* q_norm_mem, float* q_norm_scale_mem, float* w_qb_mem, float* w_qb_scale_mem, float* x_mem, float* w_proj_mem, float* q_out_mem, float* weights_out_mem) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): q_norm=TLOAD(...)
    if (_row < 4 && _col < 4) {
        q_norm[_row][_col] = q_norm_mem[_row * 4 + _col];
    }

    // FUSED (1 ops): q_norm_scale=TLOAD(...)
    if (_row < 4 && _col < 1) {
        q_norm_scale[_row][_col] = q_norm_scale_mem[_row * 1 + _col];
    }

    // FUSED (1 ops): w_qb=TLOAD(...)
    if (_row < 4 && _col < 4) {
        w_qb[_row][_col] = w_qb_mem[_row * 4 + _col];
    }

    // FUSED (1 ops): w_qb_scale=TLOAD(...)
    if (_row < 1 && _col < 4) {
        w_qb_scale[_row][_col] = w_qb_scale_mem[_row * 4 + _col];
    }

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 4 && _col < 4) {
        x[_row][_col] = x_mem[_row * 4 + _col];
    }

    // FUSED (1 ops): w_proj=TLOAD(...)
    if (_row < 4 && _col < 2) {
        w_proj[_row][_col] = w_proj_mem[_row * 2 + _col];
    }

    // TMATMUL: q_matmul = q_norm @ w_qb
    if (_row < 4 && _col < 4) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += q_norm[_row][_k] * w_qb[_k][_col];
        q_matmul[_row][_col] = _sum;}

    // FUSED (2 ops): q_scaled_row=TROWEXPANDMUL(...); q_out=TMUL(...)
    if (_row < 4 && _col < 4) {
        q_scaled_row[_row][_col] = q_matmul[_row][_col] * q_norm_scale[_row][0];
        q_out[_row][_col] = q_scaled_row[_row][_col] * w_qb_scale_expand[_row][_col];
    }

    // TMATMUL: weights_raw = x @ w_proj
    if (_row < 4 && _col < 2) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 4; _k++) _sum += x[_row][_k] * w_proj[_k][_col];
        weights_raw[_row][_col] = _sum;}

    // FUSED (1 ops): weights_out=TMULS(...)
    if (_row < 4 && _col < 2) {
        weights_out[_row][_col] = weights_raw[_row][_col] * 0.5f;
    }

    // FUSED (1 ops): q_out_mem=TSTORE(...)
    if (_row < 4 && _col < 4) {
        q_out_mem[_row * 4 + _col] = q_out[_row][_col];
    }

    // FUSED (1 ops): weights_out_mem=TSTORE(...)
    if (_row < 4 && _col < 2) {
        weights_out_mem[_row * 2 + _col] = weights_out[_row][_col];
    }

}

void lightning_indexer_prolog_quant_demo(float* q_norm_mem, float* q_norm_scale_mem, float* w_qb_mem, float* w_qb_scale_mem, float* x_mem, float* w_proj_mem, float* q_out_mem, float* weights_out_mem) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    lightning_indexer_prolog_quant_demo_kernel<<<grid, block>>>(q_norm_mem, q_norm_scale_mem, w_qb_mem, w_qb_scale_mem, x_mem, w_proj_mem, q_out_mem, weights_out_mem);
    cudaDeviceSynchronize();
}