// PTO Program: attention_fusion_block
// Function Type: Orchestration (control flow only)
// Orchestration function - builds task graph using PTO runtime
#include "pto_runtime.h"
// Note: pto_runtime.c should be compiled separately to avoid duplicate symbols
#include <string.h>  // For strcmp in main
#include <time.h>    // For benchmark timing

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void attention_fusion_block(PTORuntime* rt, float* input_q, float* input_k, float* input_v, float* output_o, float* state_o, float* state_l, float* state_m, float* temp_s, float* temp_s_scaled, float* temp_m_new, float* temp_m_local, float* temp_s_shifted, float* temp_p, float* temp_l_local, float* temp_m_diff, float* temp_scale, float* temp_l_scaled, float* temp_o_scaled, float* temp_o_local) {
    float tile_o_scaled[8][128];
    float tile_o_local[8][128];
    float tile_o_sum[8][128];
    float tile_m_copy[8][1];

    // Loop fusion: 0 loop overheads saved

    // Task 0: qk_matmul
    int32_t t0 = pto_task_alloc(rt, "qk_matmul", NULL, 139264, 135168, 1);
    pto_task_add_input(rt, t0, input_q, 0, 0, 8, 128);
    pto_task_add_input(rt, t0, input_k, 0, 0, 128, 128);
    pto_task_add_output(rt, t0, temp_s, 0, 0, 8, 128);
    pto_task_submit(rt, t0);


    // Task 1: scale_scores
    int32_t t1 = pto_task_alloc(rt, "scale_scores", NULL, 8192, 8192, 0);
    pto_task_add_input(rt, t1, temp_s, 0, 0, 8, 128);
    pto_task_add_output(rt, t1, temp_s_scaled, 0, 0, 8, 128);
    pto_task_submit(rt, t1);


    // Task 2: rowmax
    int32_t t2 = pto_task_alloc(rt, "rowmax", NULL, 4128, 4128, 0);
    pto_task_add_input(rt, t2, temp_s_scaled, 0, 0, 8, 128);
    pto_task_add_output(rt, t2, temp_m_local, 0, 0, 8, 1);
    pto_task_submit(rt, t2);


    // Task 3: maximum
    int32_t t3 = pto_task_alloc(rt, "maximum", NULL, 96, 96, 0);
    pto_task_add_input(rt, t3, state_m, 0, 0, 8, 128);
    pto_task_add_input(rt, t3, temp_m_local, 0, 0, 8, 1);
    pto_task_add_output(rt, t3, temp_m_new, 0, 0, 8, 1);
    pto_task_submit(rt, t3);


    // Task 4: rowexpandsub
    int32_t t4 = pto_task_alloc(rt, "rowexpandsub", NULL, 8224, 8224, 0);
    pto_task_add_input(rt, t4, temp_s_scaled, 0, 0, 8, 128);
    pto_task_add_input(rt, t4, temp_m_new, 0, 0, 8, 1);
    pto_task_add_output(rt, t4, temp_s_shifted, 0, 0, 8, 128);
    pto_task_submit(rt, t4);


    // Task 5: elem_exp
    int32_t t5 = pto_task_alloc(rt, "elem_exp", NULL, 8192, 8192, 0);
    pto_task_add_input(rt, t5, temp_s_shifted, 0, 0, 8, 128);
    pto_task_add_output(rt, t5, temp_p, 0, 0, 8, 128);
    pto_task_submit(rt, t5);


    // Task 6: rowsum
    int32_t t6 = pto_task_alloc(rt, "rowsum", NULL, 4128, 4128, 0);
    pto_task_add_input(rt, t6, temp_p, 0, 0, 8, 128);
    pto_task_add_output(rt, t6, temp_l_local, 0, 0, 8, 1);
    pto_task_submit(rt, t6);


    // Task 7: sub
    int32_t t7 = pto_task_alloc(rt, "sub", NULL, 96, 96, 0);
    pto_task_add_input(rt, t7, state_m, 0, 0, 8, 128);
    pto_task_add_input(rt, t7, temp_m_new, 0, 0, 8, 1);
    pto_task_add_output(rt, t7, temp_m_diff, 0, 0, 8, 1);
    pto_task_submit(rt, t7);


    // Task 8: elem_exp
    int32_t t8 = pto_task_alloc(rt, "elem_exp", NULL, 8192, 8192, 0);
    pto_task_add_input(rt, t8, temp_m_diff, 0, 0, 8, 1);
    pto_task_add_output(rt, t8, temp_scale, 0, 0, 8, 1);
    pto_task_submit(rt, t8);


    // Task 9: mul
    int32_t t9 = pto_task_alloc(rt, "mul", NULL, 96, 96, 0);
    pto_task_add_input(rt, t9, temp_scale, 0, 0, 8, 1);
    pto_task_add_input(rt, t9, state_l, 0, 0, 8, 1);
    pto_task_add_output(rt, t9, temp_l_scaled, 0, 0, 8, 1);
    pto_task_submit(rt, t9);


    // Task 10: add
    int32_t t10 = pto_task_alloc(rt, "add", NULL, 96, 96, 0);
    pto_task_add_input(rt, t10, temp_l_scaled, 0, 0, 8, 1);
    pto_task_add_input(rt, t10, temp_l_local, 0, 0, 8, 1);
    pto_task_add_output(rt, t10, state_l, 0, 0, 8, 1);
    pto_task_submit(rt, t10);


    // Task 11: pv_matmul
    int32_t t11 = pto_task_alloc(rt, "pv_matmul", NULL, 73728, 73728, 1);
    pto_task_add_input(rt, t11, temp_p, 0, 0, 8, 128);
    pto_task_add_input(rt, t11, input_v, 0, 0, 128, 128);
    pto_task_add_output(rt, t11, temp_o_local, 0, 0, 8, 128);
    pto_task_submit(rt, t11);


    // Task 12: rowexpandmul
    int32_t t12 = pto_task_alloc(rt, "rowexpandmul", NULL, 8224, 8224, 0);
    pto_task_add_input(rt, t12, state_o, 0, 0, 8, 128);
    pto_task_add_input(rt, t12, temp_scale, 0, 0, 8, 1);
    pto_task_add_output(rt, t12, temp_o_scaled, 0, 0, 8, 128);
    pto_task_submit(rt, t12);


    // TLOAD: tile_o_scaled = load(temp_o_scaled[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            tile_o_scaled[_row][_col] = temp_o_scaled[_row * 128 + _col];
        }}

    // TLOAD: tile_o_local = load(temp_o_local[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            tile_o_local[_row][_col] = temp_o_local[_row * 128 + _col];
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            tile_o_sum[_row][_col] = tile_o_scaled[_row][_col] + tile_o_local[_row][_col];
        }}

    // TSTORE: store(tile_o_sum) -> state_o[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            state_o[_row * 128 + _col] = tile_o_sum[_row][_col];
        }}

    // TLOAD: tile_m_copy = load(temp_m_new[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            tile_m_copy[_row][_col] = temp_m_new[_row * 1 + _col];
        }}

    // TSTORE: store(tile_m_copy) -> state_m[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            state_m[_row * 128 + _col] = tile_m_copy[_row][_col];
        }}

    // Task 13: rowexpanddiv
    int32_t t13 = pto_task_alloc(rt, "rowexpanddiv", NULL, 8224, 8224, 0);
    pto_task_add_input(rt, t13, state_o, 0, 0, 8, 128);
    pto_task_add_input(rt, t13, state_l, 0, 0, 8, 1);
    pto_task_add_output(rt, t13, output_o, 0, 0, 8, 128);
    pto_task_submit(rt, t13);


}
// =============================================================================
// Main Function for ARM64 Standalone Execution
// =============================================================================
// Usage: attention_fusion_block [--benchmark-only] [seq_len] [tile_rows] [num_tiles] [zero]
// Flags:
//   --benchmark-only  - Only run orchestration (skip execution), output stats