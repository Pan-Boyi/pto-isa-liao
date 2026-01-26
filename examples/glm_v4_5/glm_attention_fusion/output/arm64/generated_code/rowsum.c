// PTO Program: rowsum
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rowsum
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 4,128 bytes (4.0 KB)
//   Total capacity (w/ reuse): 4,128 bytes (4.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rowsum(float* input, float* output) {
    float x[8][128];
    float result[8][1];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
        }}

    // TROWSUM: result = rowsum(x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 128; _col++) {
            _sum += x[_row][_col];
        }
        result[_row][0] = _sum;}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            output[_row * 1 + _col] = result[_row][_col];
        }}

}