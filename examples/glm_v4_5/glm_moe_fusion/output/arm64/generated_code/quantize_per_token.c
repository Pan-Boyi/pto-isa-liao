// PTO Program: quantize_per_token
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: quantize_per_token
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     9
//   Total capacity (no reuse): 655,520 bytes (640.2 KB)
//   Total capacity (w/ reuse): 491,520 bytes (480.0 KB)
//   Reuse savings:            164,000 bytes (25.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void quantize_per_token(float* input, float* output_quant, float* output_scale, float one_hundred_twenty_seven, float one) {
    float x[8][5120];
    float x_abs[8][5120];
    float x_neg[8][5120];
    float x_max[8][1];
    float x_scale[8][1];
    float x_scaled[8][5120];
    float scale_quant[8][1];
    float one_hundred_twenty_seven_tile[8][1];
    float one_tile[8][1];

    // Loop fusion: 3 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            x[_row][_col] = input[_row * 5120 + _col];
        }}

    // Fused loop: 2 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            x_neg[_row][_col] = x[_row][_col] * -1.0;
            x_abs[_row][_col] = (x[_row][_col] > x_neg[_row][_col]) ? x[_row][_col] : x_neg[_row][_col];
        }}

    // TROWMAX: x_max = rowmax(x_abs)
    for (int _row = 0; _row < 8; _row++) {
        float _max = x_abs[_row][0];
        for (int _col = 1; _col < 5120; _col++) {
            if (x_abs[_row][_col] > _max) _max = x_abs[_row][_col];
        }
        x_max[_row][0] = _max;}

    // LI: Not implemented

    // Fused loop: 2 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            one_hundred_twenty_seven_tile[_row][_col] = x_max[_row][_col] * 0.0;
            one_hundred_twenty_seven_tile[_row][_col] = one_hundred_twenty_seven_tile[_row][_col] + 127.0;
        }}

    // TROWEXPANDDIV: x_scale = one_hundred_twenty_seven_tile / broadcast(x_max)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = x_max[_row][0];
        for (int _col = 0; _col < 1; _col++) {
            x_scale[_row][_col] = one_hundred_twenty_seven_tile[_row][_col] / _broadcast_val;
        }}

    // TROWEXPANDMUL: x_scaled = x * broadcast(x_scale)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = x_scale[_row][0];
        for (int _col = 0; _col < 5120; _col++) {
            x_scaled[_row][_col] = x[_row][_col] * _broadcast_val;
        }}

    // LI: Not implemented

    // Fused loop: 2 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            one_tile[_row][_col] = x_scale[_row][_col] * 0.0;
            one_tile[_row][_col] = one_tile[_row][_col] + 1.0;
        }}

    // TROWEXPANDDIV: scale_quant = one_tile / broadcast(x_scale)
    for (int _row = 0; _row < 8; _row++) {
        float _broadcast_val = x_scale[_row][0];
        for (int _col = 0; _col < 1; _col++) {
            scale_quant[_row][_col] = one_tile[_row][_col] / _broadcast_val;
        }}

    // TSTORE: store(x_scaled) -> output_quant[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            output_quant[_row * 5120 + _col] = x_scaled[_row][_col];
        }}

    // TSTORE: store(scale_quant) -> output_scale[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            output_scale[_row * 1 + _col] = scale_quant[_row][_col];
        }}

}