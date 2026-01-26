// PTO Program: gate_matmul
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: gate_matmul
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 6,722,560 bytes (6565.0 KB)
//   Total capacity (w/ reuse): 6,717,440 bytes (6560.0 KB)
//   Reuse savings:            5,120 bytes (0.1%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void gate_matmul(float* input_hidden, float* input_gate_weight, float* output_logits) {
    // Use dynamic allocation to avoid stack overflow (~3.3MB local arrays)
    float (*hidden)[5120] = (float(*)[5120])malloc(8 * 5120 * sizeof(float));
    float (*gate_weight)[5120] = (float(*)[5120])malloc(160 * 5120 * sizeof(float));
    float (*gate_weight_t)[160] = (float(*)[160])malloc(5120 * 160 * sizeof(float));
    float (*logits)[160] = (float(*)[160])malloc(8 * 160 * sizeof(float));
    if (!hidden || !gate_weight || !gate_weight_t || !logits) {
        fprintf(stderr, "ERROR: gate_matmul malloc failed\n");
        if (hidden) free(hidden);
        if (gate_weight) free(gate_weight);
        if (gate_weight_t) free(gate_weight_t);
        if (logits) free(logits);
        return;
    }

    // Loop fusion: 0 loop overheads saved

    // TLOAD: hidden = load(input_hidden[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            hidden[_row][_col] = input_hidden[_row * 5120 + _col];
        }}

    // TLOAD: gate_weight = load(input_gate_weight[0, 0])
    for (int _row = 0; _row < 160; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            gate_weight[_row][_col] = input_gate_weight[_row * 5120 + _col];
        }}

    // TTRANS: gate_weight_t = transpose(gate_weight)
    for (int _row = 0; _row < 5120; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            gate_weight_t[_row][_col] = gate_weight[_col][_row];
        }}

    // TMATMUL: logits = hidden @ gate_weight_t
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 160; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 5120; _k++) {
                _sum += hidden[_i][_k] * gate_weight_t[_k][_j];}
            logits[_i][_j] = _sum;}}

    // TSTORE: store(logits) -> output_logits[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            output_logits[_row * 160 + _col] = logits[_row][_col];
        }}

}