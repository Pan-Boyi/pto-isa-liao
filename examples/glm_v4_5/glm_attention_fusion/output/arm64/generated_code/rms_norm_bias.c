// PTO Program: rms_norm_bias
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rms_norm_bias
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     9
//   Total capacity (no reuse): 983,136 bytes (960.1 KB)
//   Total capacity (w/ reuse): 655,392 bytes (640.0 KB)
//   Reuse savings:            327,744 bytes (33.3%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rms_norm_bias(float* input, float* gamma_weight, float* bias_weight, float* output, float eps, float inv_cols) {
    float x[8][5120];
    float x_sq[8][5120];
    float row_sum[8][1];
    float row_mean[8][1];
    float row_rsqrt[8][1];
    float x_norm[8][5120];
    float gamma[8][5120];
    float bias[8][5120];
    float result[8][5120];

    // Loop fusion: 2 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            x[_row][_col] = input[_row * 5120 + _col];
        }}

    // TLOAD: gamma = load(gamma_weight[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            gamma[_row][_col] = gamma_weight[_row * 5120 + _col];
        }}

    // TLOAD: bias = load(bias_weight[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            bias[_row][_col] = bias_weight[_row * 5120 + _col];
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
        }}

    // TROWSUM: row_sum = rowsum(x_sq)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 5120; _col++) {
            _sum += x_sq[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // LI: Not implemented

    // Fused loop: 1 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            row_mean[_row][_col] = row_sum[_row][_col] * inv_cols;
        }}

    // LI: Not implemented

    // Fused loop: 2 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            row_mean[_row][_col] = row_mean[_row][_col] + eps;
            row_rsqrt[_row][_col] = 1.0f / sqrtf(row_mean[_row][_col]);
        }}

    // TROWEXPANDMUL: x_norm = x * broadcast(row_rsqrt)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_rsqrt[_row][0];
        for (int _col = 0; _col < 5120; _col++) {
            x_norm[_row][_col] = x[_row][_col] * _broadcast_val;
        }}

    // Fused loop: 2 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            result[_row][_col] = x_norm[_row][_col] * gamma[_row][_col];
            result[_row][_col] = result[_row][_col] + bias[_row][_col];
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            output[_row * 5120 + _col] = result[_row][_col];
        }}

}