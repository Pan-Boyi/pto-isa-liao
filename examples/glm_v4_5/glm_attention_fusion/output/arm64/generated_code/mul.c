// PTO Program: mul
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: mul
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 96 bytes (0.1 KB)
//   Total capacity (w/ reuse): 96 bytes (0.1 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void mul(float* input_a, float* input_b, float* output) {
    float a[8][1];
    float b[8][1];
    float result[8][1];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(input_a[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            a[_row][_col] = input_a[_row * 1 + _col];
        }}

    // TLOAD: b = load(input_b[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            b[_row][_col] = input_b[_row * 1 + _col];
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            result[_row][_col] = a[_row][_col] * b[_row][_col];
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            output[_row * 1 + _col] = result[_row][_col];
        }}

}