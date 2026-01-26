// PTO Program: up_proj_matmul
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: up_proj_matmul
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 15,904,768 bytes (15532.0 KB)
//   Total capacity (w/ reuse): 15,892,480 bytes (15520.0 KB)
//   Reuse savings:            12,288 bytes (0.1%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void up_proj_matmul(float* input_hidden, float* input_w13, float* output) {
    // Use dynamic allocation to avoid stack overflow (~8MB local arrays)
    float (*hidden)[5120] = (float(*)[5120])malloc(8 * 5120 * sizeof(float));
    float (*w13)[384] = (float(*)[384])malloc(5120 * 384 * sizeof(float));
    float (*w13_t)[5120] = (float(*)[5120])malloc(384 * 5120 * sizeof(float));
    float (*result)[384] = (float(*)[384])malloc(8 * 384 * sizeof(float));
    if (!hidden || !w13 || !w13_t || !result) {
        fprintf(stderr, "ERROR: up_proj_matmul malloc failed\n");
        if (hidden) free(hidden);
        if (w13) free(w13);
        if (w13_t) free(w13_t);
        if (result) free(result);
        return;
    }

    // Loop fusion: 0 loop overheads saved

    // TLOAD: hidden = load(input_hidden[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            hidden[_row][_col] = input_hidden[_row * 5120 + _col];
        }}

    // TLOAD: w13 = load(input_w13[0, 0])
    for (int _row = 0; _row < 5120; _row++) {
        for (int _col = 0; _col < 384; _col++) {
            w13[_row][_col] = input_w13[_row * 384 + _col];
        }}

    // TTRANS: w13_t = transpose(w13)
    for (int _row = 0; _row < 384; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            w13_t[_row][_col] = w13[_col][_row];
        }}

    // TMATMUL: result = hidden @ w13_t
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 384; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 5120; _k++) {
                _sum += hidden[_i][_k] * w13_t[_k][_j];}
            result[_i][_j] = _sum;}}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 384; _col++) {
            output[_row * 384 + _col] = result[_row][_col];
        }}

}