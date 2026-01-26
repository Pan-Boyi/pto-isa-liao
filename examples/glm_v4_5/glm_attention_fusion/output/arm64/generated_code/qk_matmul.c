// PTO Program: qk_matmul
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: qk_matmul
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 139,264 bytes (136.0 KB)
//   Total capacity (w/ reuse): 135,168 bytes (132.0 KB)
//   Reuse savings:            4,096 bytes (2.9%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void qk_matmul(float* input_q, float* input_k, float* output_s) {
    // Use dynamic allocation to avoid stack overflow (~136KB local arrays)
    float (*q)[128] = (float(*)[128])malloc(8 * 128 * sizeof(float));
    float (*k)[128] = (float(*)[128])malloc(128 * 128 * sizeof(float));
    float (*k_t)[128] = (float(*)[128])malloc(128 * 128 * sizeof(float));
    float (*s)[128] = (float(*)[128])malloc(8 * 128 * sizeof(float));
    if (!q || !k || !k_t || !s) {
        fprintf(stderr, "ERROR: qk_matmul malloc failed\n");
        if (q)        if (k)        if (k_t)        if (s)        return;
    }

    // Loop fusion: 0 loop overheads saved

    // TLOAD: q = load(input_q[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            q[_row][_col] = input_q[_row * 128 + _col];
        }}

    // TLOAD: k = load(input_k[0, 0])
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            k[_row][_col] = input_k[_row * 128 + _col];
        }}

    // TTRANS: k_t = transpose(k)
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            k_t[_row][_col] = k[_col][_row];
        }}

    // TMATMUL: s = q @ k_t
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += q[_i][_k] * k_t[_k][_j];}
            s[_i][_j] = _sum;}}

    // TSTORE: store(s) -> output_s[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_s[_row * 128 + _col] = s[_row][_col];
        }}


    // Free dynamically allocated memory
    free(q);
    free(k);
    free(k_t);
    free(s);
}