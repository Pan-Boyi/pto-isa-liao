// PTO Program: add_bias
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: add_bias
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 16,000 bytes (15.6 KB)
//   Total capacity (w/ reuse): 16,000 bytes (15.6 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void add_bias(float* input_x, float* input_bias, float* output) {
    float x[8][160];
    float bias_1d[1][160];
    // bias_expanded not needed
    float result[8][160];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input_x[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            x[_row][_col] = input_x[_row * 160 + _col];
        }}

    // TLOAD: bias_1d = load(input_bias[0, 0])
    for (int _row = 0; _row < 1; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            bias_1d[0][_col] = input_bias[_col];  // FIX: input_bias is 1D array
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            result[_row][_col] = x[_row][_col] + bias_1d[0][_col];  // FIX: broadcast bias[0, :] to all rows
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            output[_row * 160 + _col] = result[_row][_col];
        }}

}