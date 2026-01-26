// PTO Program: moe_fusion_block
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

void moe_fusion_block(PTORuntime* rt, float* input_hidden_states, float* input_gate_weight, float* input_e_score_bias, float* input_w13, float* input_w13_scale, float* input_w2, float* input_w2_scale, float* output_logits, float* output_ffn_res, float* temp_gate_logits, float* temp_sigmoid_out, float* temp_gate_weights, float* temp_up_proj, float* temp_swiglu_out, float* temp_down_proj) {
    float tile_gate[8][160];
    float tile_ffn_res[8][5120];

    // Loop fusion: 0 loop overheads saved

    // Task 0: gate_matmul
    int32_t t0 = pto_task_alloc(rt, "gate_matmul", NULL, 6722560, 6717440, 1);
    pto_task_add_input(rt, t0, input_hidden_states, 0, 0, 8, 5120);
    pto_task_add_input(rt, t0, input_gate_weight, 0, 0, 160, 5120);
    pto_task_add_output(rt, t0, temp_gate_logits, 0, 0, 8, 160);
    pto_task_submit(rt, t0);


    // Task 1: sigmoid
    int32_t t1 = pto_task_alloc(rt, "sigmoid", NULL, 30720, 10240, 0);
    pto_task_add_input(rt, t1, temp_gate_logits, 0, 0, 8, 160);
    pto_task_add_output(rt, t1, temp_sigmoid_out, 0, 0, 8, 160);
    pto_task_submit(rt, t1);


    // Task 2: add_bias
    int32_t t2 = pto_task_alloc(rt, "add_bias", NULL, 16000, 16000, 0);
    pto_task_add_input(rt, t2, temp_sigmoid_out, 0, 0, 8, 160);
    pto_task_add_input(rt, t2, input_e_score_bias, 0, 0, 1, 160);
    pto_task_add_output(rt, t2, temp_gate_weights, 0, 0, 8, 160);
    pto_task_submit(rt, t2);


    // TLOAD: tile_gate = load(temp_gate_weights[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            tile_gate[_row][_col] = temp_gate_weights[_row * 160 + _col];
        }}

    // TSTORE: store(tile_gate) -> output_logits[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 160; _col++) {
            output_logits[_row * 160 + _col] = tile_gate[_row][_col];
        }}

    // Task 3: up_proj_matmul
    int32_t t3 = pto_task_alloc(rt, "up_proj_matmul", NULL, 15904768, 15892480, 1);
    pto_task_add_input(rt, t3, input_hidden_states, 0, 0, 8, 5120);
    pto_task_add_input(rt, t3, input_w13, 0, 0, 5120, 384);
    pto_task_add_output(rt, t3, temp_up_proj, 0, 0, 8, 384);
    pto_task_submit(rt, t3);


    // Task 4: swiglu
    int32_t t4 = pto_task_alloc(rt, "swiglu", NULL, 55296, 36864, 0);
    pto_task_add_input(rt, t4, temp_up_proj, 0, 0, 8, 384);
    pto_task_add_output(rt, t4, temp_swiglu_out, 0, 0, 8, 192);
    pto_task_submit(rt, t4);


    // Task 5: down_proj_matmul
    int32_t t5 = pto_task_alloc(rt, "down_proj_matmul", NULL, 8034304, 7870464, 1);
    pto_task_add_input(rt, t5, temp_swiglu_out, 0, 0, 8, 192);
    pto_task_add_input(rt, t5, input_w2, 0, 0, 192, 5120);
    pto_task_add_output(rt, t5, temp_down_proj, 0, 0, 8, 5120);
    pto_task_submit(rt, t5);


    // TLOAD: tile_ffn_res = load(temp_down_proj[0, 0])
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            tile_ffn_res[_row][_col] = temp_down_proj[_row * 5120 + _col];
        }}

    // TSTORE: store(tile_ffn_res) -> output_ffn_res[0, 0]
    for (int _row = 0; _row < 8; _row++) {
        for (int _col = 0; _col < 5120; _col++) {
            output_ffn_res[_row * 5120 + _col] = tile_ffn_res[_row][_col];
        }}

}
// =============================================================================
// Main Function for ARM64 Standalone Execution
// =============================================================================
// Usage: moe_fusion_block [--benchmark-only] [seq_len] [tile_rows] [num_tiles] [zero]
// Flags:
//   --benchmark-only  - Only run orchestration (skip execution), output stats

int main(int argc, char** argv) {
    // Check for --benchmark-only flag
    int benchmark_only = 0;
    int arg_offset = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark-only") == 0) {
            benchmark_only = 1;
            arg_offset = 1;
            break;
        }
    }
    
    printf("============================================================\n");
    printf("  PTO ARM64 Runtime\n");
    printf("============================================================\n");
    
    // Initialize runtime (heap allocated - PTORuntime is too large for stack)
    PTORuntime* rt = (PTORuntime*)calloc(1, sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "Failed to allocate PTORuntime\n");
        return 1;
    }
    pto_runtime_init(rt);
    
    // Allocate test data
    float* input_hidden_states = (float*)calloc(1024 * 1024, sizeof(float));
    float* input_gate_weight = (float*)calloc(1024 * 1024, sizeof(float));
    float* input_e_score_bias = (float*)calloc(1024 * 1024, sizeof(float));
    float* input_w13 = (float*)calloc(1024 * 1024, sizeof(float));
    float* input_w13_scale = (float*)calloc(1024 * 1024, sizeof(float));
    float* input_w2 = (float*)calloc(1024 * 1024, sizeof(float));
    float* input_w2_scale = (float*)calloc(1024 * 1024, sizeof(float));
    float* output_logits = (float*)calloc(1024 * 1024, sizeof(float));
    float* output_ffn_res = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_gate_logits = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_sigmoid_out = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_gate_weights = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_up_proj = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_swiglu_out = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_down_proj = (float*)calloc(1024 * 1024, sizeof(float));

    
    if (benchmark_only) {
        // Benchmark mode: only measure orchestration time
        struct timespec start, end;
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        moe_fusion_block(rt, input_hidden_states, input_gate_weight, input_e_score_bias, input_w13, input_w13_scale, input_w2, input_w2_scale, output_logits, output_ffn_res, temp_gate_logits, temp_sigmoid_out, temp_gate_weights, temp_up_proj, temp_swiglu_out, temp_down_proj);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
        long long tasks_submitted = rt->total_tasks_scheduled;
        double tasks_per_ms = tasks_submitted / time_ms;
        
        // Output in machine-parseable format
        printf("BENCHMARK: tasks=%lld time_ms=%.3f tasks_per_ms=%.2f\n",
               tasks_submitted, time_ms, tasks_per_ms);
    } else {
        // Normal execution mode
        printf("Running orchestration function: moe_fusion_block\n");
        printf("------------------------------------------------------------\n");
        
        moe_fusion_block(rt, input_hidden_states, input_gate_weight, input_e_score_bias, input_w13, input_w13_scale, input_w2, input_w2_scale, output_logits, output_ffn_res, temp_gate_logits, temp_sigmoid_out, temp_gate_weights, temp_up_proj, temp_swiglu_out, temp_down_proj);
        
        printf("------------------------------------------------------------\n");
        printf("Submitted %lld tasks\n", (long long)rt->total_tasks_scheduled);
        
        // Execute all tasks
        pto_execute_all(rt);
        
        printf("Execution complete!\n");
    }
    
    // Cleanup - must call shutdown before free to destroy mutexes/condvars
    fflush(stdout);
    pto_runtime_shutdown(rt);
    free(input_hidden_states);
    free(input_gate_weight);
    free(input_e_score_bias);
    free(input_w13);
    free(input_w13_scale);
    free(input_w2);
    free(input_w2_scale);
    free(output_logits);
    free(output_ffn_res);
    free(temp_gate_logits);
    free(temp_sigmoid_out);
    free(temp_gate_weights);
    free(temp_up_proj);
    free(temp_swiglu_out);
    free(temp_down_proj);
    free(rt);
    
    return 0;
}
