// PTO Program: dequant_dynamic
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: dequant_dynamic
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     5
//   Total capacity (no reuse): 36,928 bytes (36.1 KB)
//   Total capacity (w/ reuse): 24,640 bytes (24.1 KB)
//   Reuse savings:            12,288 bytes (33.3%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void dequant_dynamic(float* input, float* input_scale_1, float* input_scale_2, float* output) {
    float input_fp32[8][384];
    float scale_1[8][1];
    float scale_2[8][1];
    float scaled_1[8][384];
    float result[8][384];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: input_fp32 = load(input[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 384; _col++) {
            input_fp32[_row][_col] = input[_row * 384 + _col];
        }}

    // TLOAD: scale_1 = load(input_scale_1[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            scale_1[_row][_col] = input_scale_1[_row * 1 + _col];
        }}

    // TLOAD: scale_2 = load(input_scale_2[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            scale_2[_row][_col] = input_scale_2[_row * 1 + _col];
        }}

    // TROWEXPANDMUL: scaled_1 = input_fp32 * broadcast(scale_2)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = scale_2[_row][0];
        for (int _col = 0; _col < 384; _col++) {
            scaled_1[_row][_col] = input_fp32[_row][_col] * _broadcast_val;
        }}

    // TROWEXPANDMUL: result = scaled_1 * broadcast(scale_1)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = scale_1[_row][0];
        for (int _col = 0; _col < 384; _col++) {
            result[_row][_col] = scaled_1[_row][_col] * _broadcast_val;
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 384; _col++) {
            output[_row * 384 + _col] = result[_row][_col];
        }}

}