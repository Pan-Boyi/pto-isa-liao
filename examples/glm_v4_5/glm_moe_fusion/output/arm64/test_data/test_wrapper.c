
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <signal.h>
#include <setjmp.h>
#include "pto_runtime.h"
#include "pto_runtime_common.h"

// Forward declarations of InCore functions
void gate_matmul(float* input_hidden, float* input_gate_weight, float* output_logits);
void sigmoid(float* input, float* output, float one);
void add_bias(float* input_x, float* input_bias, float* output);
void swiglu(float* input, float* output);
void up_proj_matmul(float* input_hidden, float* input_w13, float* output);
void down_proj_matmul(float* input_swiglu, float* input_w2, float* output);

// Wrapper functions

void gate_matmul_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        gate_matmul((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void sigmoid_wrapper(void** args, int32_t num_args) {
    if (num_args >= 2) {
        sigmoid((float*)args[0], (float*)args[1], 1.0f);
    }
}

void add_bias_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        add_bias((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void swiglu_wrapper(void** args, int32_t num_args) {
    if (num_args >= 2) {
        swiglu((float*)args[0], (float*)args[1]);
    }
}

void up_proj_matmul_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        up_proj_matmul((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void down_proj_matmul_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        down_proj_matmul((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

// Signal handler for debugging crashes
static jmp_buf crash_jmp;
static void crash_handler(int sig) {
    fprintf(stderr, "ERROR: Signal %d received (likely crash)\n", sig);
    longjmp(crash_jmp, 1);
}

int main(int argc, char** argv) {
    // Set up signal handlers for debugging
    signal(SIGABRT, crash_handler);
    signal(SIGSEGV, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp) != 0) {
        fprintf(stderr, "Program crashed. Attempting cleanup...\n");
        return 1;
    }
    
    if (argc < 7) {
        fprintf(stderr, "Usage: test_wrapper <hidden_states_file> <gate_weight_file> <e_score_bias_file> <w13_file> <w2_file> <gate_output_file> <ffn_output_file>\n");
        return 1;
    }
    
    const char* hidden_states_file = argv[1];
    const char* gate_weight_file = argv[2];
    const char* e_score_bias_file = argv[3];
    const char* w13_file = argv[4];
    const char* w2_file = argv[5];
    const char* gate_output_file = argv[6];
    const char* ffn_output_file = argv[7];
    
    // Dimensions
    int bs_tile = 8;
    int hidden_size = 5120;
    int intermediate_size = 192;
    int num_experts = 160;
    
    // Initialize runtime
    PTORuntime* rt = (PTORuntime*)calloc(1, sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "Failed to allocate PTORuntime\n");
        return 1;
    }
    pto_runtime_init(rt);
    
    // Allocate buffers
    float* input_hidden_states = (float*)malloc(bs_tile * hidden_size * sizeof(float));
    float* input_gate_weight = (float*)malloc(num_experts * hidden_size * sizeof(float));
    float* input_e_score_bias = (float*)malloc(num_experts * sizeof(float));
    float* input_w13 = (float*)malloc(hidden_size * intermediate_size * 2 * sizeof(float));
    float* input_w2 = (float*)malloc(intermediate_size * hidden_size * sizeof(float));
    float* output_logits = (float*)calloc(bs_tile * num_experts, sizeof(float));
    float* output_ffn_res = (float*)calloc(bs_tile * hidden_size, sizeof(float));
    
    // Temporary buffers
    float* temp_gate_logits = (float*)calloc(bs_tile * num_experts, sizeof(float));
    float* temp_sigmoid_out = (float*)calloc(bs_tile * num_experts, sizeof(float));
    float* temp_gate_weights = (float*)calloc(bs_tile * num_experts, sizeof(float));
    float* temp_up_proj = (float*)calloc(bs_tile * intermediate_size * 2, sizeof(float));
    float* temp_swiglu_out = (float*)calloc(bs_tile * intermediate_size, sizeof(float));
    float* temp_down_proj = (float*)calloc(bs_tile * hidden_size, sizeof(float));
    
    // Load input data
    FILE* fp = fopen(hidden_states_file, "rb");
    if (!fp || fread(input_hidden_states, sizeof(float), bs_tile * hidden_size, fp) != bs_tile * hidden_size) {
        fprintf(stderr, "Failed to read hidden_states file\n");
        return 1;
    }
    fclose(fp);
    
    fp = fopen(gate_weight_file, "rb");
    if (!fp || fread(input_gate_weight, sizeof(float), num_experts * hidden_size, fp) != num_experts * hidden_size) {
        fprintf(stderr, "Failed to read gate_weight file\n");
        return 1;
    }
    fclose(fp);
    
    fp = fopen(e_score_bias_file, "rb");
    if (!fp || fread(input_e_score_bias, sizeof(float), num_experts, fp) != num_experts) {
        fprintf(stderr, "Failed to read e_score_bias file\n");
        return 1;
    }
    fclose(fp);
    
    fp = fopen(w13_file, "rb");
    if (!fp || fread(input_w13, sizeof(float), hidden_size * intermediate_size * 2, fp) != hidden_size * intermediate_size * 2) {
        fprintf(stderr, "Failed to read w13 file\n");
        return 1;
    }
    fclose(fp);
    
    fp = fopen(w2_file, "rb");
    if (!fp || fread(input_w2, sizeof(float), intermediate_size * hidden_size, fp) != intermediate_size * hidden_size) {
        fprintf(stderr, "Failed to read w2 file\n");
        return 1;
    }
    fclose(fp);
    
    printf("Executing MoE fusion block...\n");
    fflush(stdout);
    
    // Call moe_fusion_block function
    // Note: This is a simplified version - we'll call InCore functions directly
    // Gate computation
    printf("  Step 1: Gate matmul...\n");
    fflush(stdout);
    gate_matmul(input_hidden_states, input_gate_weight, temp_gate_logits);
    
    printf("  Step 2: Sigmoid...\n");
    fflush(stdout);
    sigmoid(temp_gate_logits, temp_sigmoid_out, 1.0f);
    
    printf("  Step 3: Add bias...\n");
    fflush(stdout);
    add_bias(temp_sigmoid_out, input_e_score_bias, temp_gate_weights);
    
    // FFN computation
    printf("  Step 4: Up projection matmul...\n");
    fflush(stdout);
    up_proj_matmul(input_hidden_states, input_w13, temp_up_proj);
    
    printf("  Step 5: SwiGLU...\n");
    fflush(stdout);
    swiglu(temp_up_proj, temp_swiglu_out);
    
    printf("  Step 6: Down projection matmul...\n");
    fflush(stdout);
    down_proj_matmul(temp_swiglu_out, input_w2, temp_down_proj);
    
    // Copy results to output
    printf("  Step 7: Copying results...\n");
    fflush(stdout);
    memcpy(output_logits, temp_gate_weights, bs_tile * num_experts * sizeof(float));
    memcpy(output_ffn_res, temp_down_proj, bs_tile * hidden_size * sizeof(float));
    
    printf("Execution complete!\n");
    fflush(stdout);
    
    // Save outputs
    fp = fopen(gate_output_file, "wb");
    if (!fp || fwrite(output_logits, sizeof(float), bs_tile * num_experts, fp) != bs_tile * num_experts) {
        fprintf(stderr, "Failed to write gate output file\n");
        return 1;
    }
    fclose(fp);
    
    fp = fopen(ffn_output_file, "wb");
    if (!fp || fwrite(output_ffn_res, sizeof(float), bs_tile * hidden_size, fp) != bs_tile * hidden_size) {
        fprintf(stderr, "Failed to write ffn output file\n");
        return 1;
    }
    fclose(fp);
    
    // Cleanup
    pto_runtime_shutdown(rt);
    free(rt);
    free(input_hidden_states);
    free(input_gate_weight);
    free(input_e_score_bias);
    free(input_w13);
    free(input_w2);
    free(output_logits);
    free(output_ffn_res);
    free(temp_gate_logits);
    free(temp_sigmoid_out);
    free(temp_gate_weights);
    free(temp_up_proj);
    free(temp_swiglu_out);
    free(temp_down_proj);
    
    return 0;
}
