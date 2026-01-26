// PTO Program: scale_scores
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: scale_scores
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 8,192 bytes (8.0 KB)
//   Total capacity (w/ reuse): 8,192 bytes (8.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void scale_scores(float* input_s, float* output, float scale) {
    float s[8][128];
    float s_scaled[8][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: s = load(input_s[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            s[_row][_col] = input_s[_row * 128 + _col];
        }}

    // LI: Not implemented

    // Fused loop: 1 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            s_scaled[_row][_col] = s[_row][_col] * scale;
        }}

    // TSTORE: store(s_scaled) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = s_scaled[_row][_col];
        }}

}