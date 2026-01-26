// PTO Program: pv_matmul
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: pv_matmul
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 73,728 bytes (72.0 KB)
//   Total capacity (w/ reuse): 73,728 bytes (72.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void pv_matmul(float* input_p, float* input_v, float* output) {
    // Use dynamic allocation to avoid stack overflow (~68KB local arrays)
    float (*p)[128] = (float(*)[128])malloc(8 * 128 * sizeof(float));
    float (*v)[128] = (float(*)[128])malloc(128 * 128 * sizeof(float));
    float (*o)[128] = (float(*)[128])malloc(8 * 128 * sizeof(float));
    if (!p || !v || !o) {
        fprintf(stderr, "ERROR: pv_matmul malloc failed\n");
        if (p) free(p);
        if (v) free(v);
        if (o) free(o);
        return;
    }

    // Loop fusion: 0 loop overheads saved

    // TLOAD: p = load(input_p[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            p[_row][_col] = input_p[_row * 128 + _col];
        }}

    // TLOAD: v = load(input_v[0, 0])
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            v[_row][_col] = input_v[_row * 128 + _col];
        }}

    // TMATMUL: o = p @ v
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += p[_i][_k] * v[_k][_j];}
            o[_i][_j] = _sum;}}

    // TSTORE: store(o) -> output[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = o[_row][_col];
        }}

    // Free dynamically allocated memory
    free(p);
    free(v);
    free(o);
}