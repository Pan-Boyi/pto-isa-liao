// PTO Program: rowexpanddiv
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rowexpanddiv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 327,712 bytes (320.0 KB)
//   Total capacity (w/ reuse): 327,712 bytes (320.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rowexpanddiv(float* input_x, float* input_row, float* output) {
    float x[8][5120];
    float row_vals[8][1];
    float result[8][5120];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input_x[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            x[_row][_col] = input_x[_row * 5120 + _col];
        }}

    // TLOAD: row_vals = load(input_row[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            row_vals[_row][_col] = input_row[_row * 1 + _col];
        }}

    // TROWEXPANDDIV: result = x / broadcast(row_vals)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = row_vals[_row][0];
        for (int _col = 0; _col < 5120; _col++) {
            result[_row][_col] = x[_row][_col] / _broadcast_val;
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            output[_row * 5120 + _col] = result[_row][_col];
        }}

}