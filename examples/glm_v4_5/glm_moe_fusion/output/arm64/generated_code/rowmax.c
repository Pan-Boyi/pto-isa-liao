// PTO Program: rowmax
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rowmax
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 163,872 bytes (160.0 KB)
//   Total capacity (w/ reuse): 163,872 bytes (160.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rowmax(float* input, float* output) {
    float x[8][5120];
    float result[8][1];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            x[_row][_col] = input[_row * 5120 + _col];
        }}

    // TROWMAX: result = rowmax(x)
    for (int _row = 0; _row < 8; _row++) {
        float _max = x[_row][0];
        for (int _col = 1; _col < 5120; _col++) {
            if (x[_row][_col] > _max) _max = x[_row][_col];
        }
        result[_row][0] = _max;}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            output[_row * 1 + _col] = result[_row][_col];
        }}

}