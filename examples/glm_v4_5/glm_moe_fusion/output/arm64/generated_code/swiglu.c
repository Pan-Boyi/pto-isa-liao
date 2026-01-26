// PTO Program: swiglu
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 55,296 bytes (54.0 KB)
//   Total capacity (w/ reuse): 36,864 bytes (36.0 KB)
//   Reuse savings:            18,432 bytes (33.3%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void swiglu(float* input, float* output) {
    float up_proj[8][384];
    float left[8][192];
    float right[8][192];
    float left_neg[8][192];
    float left_exp[8][192];
    float left_exp_plus_one[8][192];
    float sigmoid_left[8][192];
    float result[8][192];

    // Loop fusion: 6 loop overheads saved

    // TLOAD: up_proj = load(input[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 384; _col++) {
            up_proj[_row][_col] = input[_row * 384 + _col];
        }}

    // TLOAD: left = load(input[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 192; _col++) {
            left[_row][_col] = input[_row * 384 + _col];  // FIX: stride is 384 (intermediate_size * 2)
        }}

    // TLOAD: right = load(input[0, 192])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 192; _col++) {
            right[_row][_col] = input[_row * 384 + 192 + _col];  // FIX: stride is 384, offset is 192
        }}

    // Fused loop: 2 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 192; _col++) {
            left_neg[_row][_col] = left[_row][_col] * -1.0;
            left_exp[_row][_col] = expf(left_neg[_row][_col]);
        }}

    // LI: Not implemented

    // Fused loop: 6 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 192; _col++) {
            left_exp_plus_one[_row][_col] = left_exp[_row][_col] + 1.0;
            sigmoid_left[_row][_col] = left_exp_plus_one[_row][_col] * 0.0;
            sigmoid_left[_row][_col] = sigmoid_left[_row][_col] + 1.0;
            sigmoid_left[_row][_col] = sigmoid_left[_row][_col] / left_exp_plus_one[_row][_col];
            result[_row][_col] = left[_row][_col] * sigmoid_left[_row][_col];
            result[_row][_col] = result[_row][_col] * right[_row][_col];
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 192; _col++) {
            output[_row * 192 + _col] = result[_row][_col];
        }}

}