// PTO Program: sigmoid
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: sigmoid
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 30,720 bytes (30.0 KB)
//   Total capacity (w/ reuse): 10,240 bytes (10.0 KB)
//   Reuse savings:            20,480 bytes (66.7%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void sigmoid(float* input, float* output, float one) {
    float x[8][160];
    float x_neg[8][160];
    float x_exp[8][160];
    float x_exp_plus_one[8][160];
    float one_tile[8][160];
    float result[8][160];

    // Loop fusion: 4 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            x[_row][_col] = input[_row * 160 + _col];
        }}

    // Fused loop: 2 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            x_neg[_row][_col] = x[_row][_col] * -1.0;
            x_exp[_row][_col] = expf(x_neg[_row][_col]);
        }}

    // LI: Not implemented

    // Fused loop: 4 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            x_exp_plus_one[_row][_col] = x_exp[_row][_col] + one;
            one_tile[_row][_col] = x_exp_plus_one[_row][_col] * 0.0;
            one_tile[_row][_col] = one_tile[_row][_col] + 1.0;
            result[_row][_col] = one_tile[_row][_col] / x_exp_plus_one[_row][_col];
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            output[_row * 160 + _col] = result[_row][_col];
        }}

}