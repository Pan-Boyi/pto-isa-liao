// PTO Program: add
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: add
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 491,520 bytes (480.0 KB)
//   Total capacity (w/ reuse): 491,520 bytes (480.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void add(float* input_a, float* input_b, float* output) {
    float a[8][5120];
    float b[8][5120];
    float result[8][5120];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(input_a[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            a[_row][_col] = input_a[_row * 5120 + _col];
        }}

    // TLOAD: b = load(input_b[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            b[_row][_col] = input_b[_row * 5120 + _col];
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            result[_row][_col] = a[_row][_col] + b[_row][_col];
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            output[_row * 5120 + _col] = result[_row][_col];
        }}

}