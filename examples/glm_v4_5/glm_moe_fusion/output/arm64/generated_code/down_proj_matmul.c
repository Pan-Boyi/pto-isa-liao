// PTO Program: down_proj_matmul
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: down_proj_matmul
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 8,034,304 bytes (7846.0 KB)
//   Total capacity (w/ reuse): 7,870,464 bytes (7686.0 KB)
//   Reuse savings:            163,840 bytes (2.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void down_proj_matmul(float* input_swiglu, float* input_w2, float* output) {
    // Use dynamic allocation to avoid stack overflow (~4MB local arrays)
    float (*swiglu)[192] = (float(*)[192])malloc(8 * 192 * sizeof(float));
    float (*w2)[5120] = (float(*)[5120])malloc(192 * 5120 * sizeof(float));
    float (*w2_t)[192] = (float(*)[192])malloc(5120 * 192 * sizeof(float));
    float (*result)[5120] = (float(*)[5120])malloc(8 * 5120 * sizeof(float));
    if (!swiglu || !w2 || !w2_t || !result) {
        fprintf(stderr, "ERROR: down_proj_matmul malloc failed\n");
        if (swiglu) free(swiglu);
        if (w2) free(w2);
        if (w2_t) free(w2_t);
        if (result) free(result);
        return;
    }

    // Loop fusion: 0 loop overheads saved

    // TLOAD: swiglu = load(input_swiglu[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 192; _col++) {
            swiglu[_row][_col] = input_swiglu[_row * 192 + _col];
        }}

    // TLOAD: w2 = load(input_w2[0, 0])
    for (int _row = 0; _row < 192; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            w2[_row][_col] = input_w2[_row * 5120 + _col];
        }}

    // TTRANS: w2_t = transpose(w2)
    for (int _row = 0; _row < 5120; _row++) {
        for (int _col = 0; _col < 192; _col++) {
            w2_t[_row][_col] = w2[_col][_row];
        }}

    // TMATMUL: result = swiglu @ w2_t
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 5120; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 192; _k++) {
                _sum += swiglu[_i][_k] * w2_t[_k][_j];}
            result[_i][_j] = _sum;}}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            output[_row * 5120 + _col] = result[_row][_col];
        }}

}