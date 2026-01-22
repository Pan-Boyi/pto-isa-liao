/**
 * LLaMA Layer Test with Loop Replay
 * 
 * This test verifies that the compact replay mechanism produces
 * IDENTICAL results to direct task creation for LLaMA operations.
 * 
 * Test structure:
 * 1. Run reference implementation (CPU)
 * 2. Run with direct task creation (no replay)
 * 3. Run with loop replay (compact_task array)
 * 4. Verify all three produce identical results
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "../pto_runtime.h"
#include "../pto_runtime.c"

// =============================================================================
// Configuration
// =============================================================================

#define TILE_ROWS       32
#define HIDDEN_DIM      128
#define HEAD_DIM        32

static inline double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// =============================================================================
// InCore Functions - Actual Implementations
// =============================================================================

void rmsnorm_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* weight = (float*)args[1];
    float* output = (float*)args[2];
    float eps = 1e-6f;
    
    for (int row = 0; row < TILE_ROWS; row++) {
        float sum_sq = 0.0f;
        for (int col = 0; col < HIDDEN_DIM; col++) {
            float val = input[row * HIDDEN_DIM + col];
            sum_sq += val * val;
        }
        float inv_rms = 1.0f / sqrtf(sum_sq / HIDDEN_DIM + eps);
        
        for (int col = 0; col < HIDDEN_DIM; col++) {
            output[row * HIDDEN_DIM + col] = 
                input[row * HIDDEN_DIM + col] * inv_rms * weight[col];
        }
    }
}

void linear_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* weight = (float*)args[1];
    float* output = (float*)args[2];
    
    for (int row = 0; row < TILE_ROWS; row++) {
        for (int col = 0; col < HIDDEN_DIM; col++) {
            output[row * HIDDEN_DIM + col] = 
                input[row * HIDDEN_DIM + col] * weight[col];
        }
    }
}

void scale_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input = (float*)args[0];
    float* output = (float*)args[1];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    
    for (int i = 0; i < TILE_ROWS * HIDDEN_DIM; i++) {
        output[i] = input[i] * scale;
    }
}

void residual_add_incore(void** args, int32_t num_args) {
    (void)num_args;
    float* input1 = (float*)args[0];
    float* input2 = (float*)args[1];
    float* output = (float*)args[2];
    
    for (int i = 0; i < TILE_ROWS * HIDDEN_DIM; i++) {
        output[i] = input1[i] + input2[i];
    }
}

// =============================================================================
// User Data
// =============================================================================

typedef struct {
    float* input;
    float* output;
    float* norm_weight;
    float* proj_weight;
    float* temp_norm;
    float* temp_proj;
    float* temp_out;
    int num_tiles;
} TestData;

// =============================================================================
// Orchestration - Direct (no replay)
// =============================================================================

void orchestration_direct(PTORuntime* rt, void* user_data) {
    TestData* data = (TestData*)user_data;
    int tile_size = TILE_ROWS * HIDDEN_DIM;
    
    for (int tile = 0; tile < data->num_tiles; tile++) {
        float* in_ptr = data->input + tile * tile_size;
        float* norm_ptr = data->temp_norm + tile * tile_size;
        float* proj_ptr = data->temp_proj + tile * tile_size;
        float* out_ptr = data->temp_out + tile * tile_size;
        float* final_ptr = data->output + tile * tile_size;
        
        // Task 0: RMSNorm
        int32_t t0 = pto_task_alloc(rt, "rmsnorm", (void*)rmsnorm_incore, 0, 0);
        pto_task_add_input(rt, t0, in_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t0, data->norm_weight, 0, 0, 1, HIDDEN_DIM);
        pto_task_add_output(rt, t0, norm_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t0);
        
        // Task 1: Linear
        int32_t t1 = pto_task_alloc(rt, "linear", (void*)linear_incore, 0, 0);
        pto_task_add_input(rt, t1, norm_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t1, data->proj_weight, 0, 0, 1, HIDDEN_DIM);
        pto_task_add_output(rt, t1, proj_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t1);
        
        // Task 2: Scale
        int32_t t2 = pto_task_alloc(rt, "scale", (void*)scale_incore, 0, 0);
        pto_task_add_input(rt, t2, proj_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_output(rt, t2, out_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t2);
        
        // Task 3: Residual add
        int32_t t3 = pto_task_alloc(rt, "residual", (void*)residual_add_incore, 0, 0);
        pto_task_add_input(rt, t3, in_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t3, out_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_output(rt, t3, final_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t3);
    }
}

// =============================================================================
// Orchestration - With Loop Replay (compact_task array)
// =============================================================================

void orchestration_replay(PTORuntime* rt, void* user_data) {
    TestData* data = (TestData*)user_data;
    int tile_size = TILE_ROWS * HIDDEN_DIM;
    
    // Initialize loop replay context
    // NOTE: ctx must NOT be static when running multiple tests - it retains state
    LoopReplayCtx ctx = {0};
    pto_loop_init(&ctx, "tile_loop", tile_size, OFFSET_NONE);
    
    for (int tile = 0; tile < data->num_tiles; tile++) {
        if (pto_loop_should_record(rt, &ctx, tile)) {
            // First iteration: record the task pattern
            float* in_ptr = data->input + tile * tile_size;
            float* norm_ptr = data->temp_norm + tile * tile_size;
            float* proj_ptr = data->temp_proj + tile * tile_size;
            float* out_ptr = data->temp_out + tile * tile_size;
            float* final_ptr = data->output + tile * tile_size;
            
            // Task 0: RMSNorm
            int32_t t0 = pto_task_alloc(rt, "rmsnorm", (void*)rmsnorm_incore, 0, 0);
            pto_task_add_input(rt, t0, in_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t0, data->norm_weight, 0, 0, 1, HIDDEN_DIM);
            pto_task_add_output(rt, t0, norm_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t0);
            
            // Task 1: Linear
            int32_t t1 = pto_task_alloc(rt, "linear", (void*)linear_incore, 0, 0);
            pto_task_add_input(rt, t1, norm_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t1, data->proj_weight, 0, 0, 1, HIDDEN_DIM);
            pto_task_add_output(rt, t1, proj_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t1);
            
            // Task 2: Scale
            int32_t t2 = pto_task_alloc(rt, "scale", (void*)scale_incore, 0, 0);
            pto_task_add_input(rt, t2, proj_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_output(rt, t2, out_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t2);
            
            // Task 3: Residual add
            int32_t t3 = pto_task_alloc(rt, "residual", (void*)residual_add_incore, 0, 0);
            pto_task_add_input(rt, t3, in_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t3, out_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_output(rt, t3, final_ptr, 0, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t3);
            
            pto_loop_finish_record(rt, &ctx);
        } else {
            // Subsequent iterations: replay with offset
            pto_loop_replay(rt, &ctx, tile);
        }
    }
    
    pto_loop_cleanup(&ctx);
}

// =============================================================================
// Orchestration - With Loop Replay using ROW OFFSET mode
// =============================================================================

void orchestration_replay_rowoffset(PTORuntime* rt, void* user_data) {
    TestData* data = (TestData*)user_data;
    
    // Use OFFSET_ROW mode: template stores base tensor, row_offset adjusted per tile
    // NOTE: ctx must NOT be static when running multiple tests - it retains state
    LoopReplayCtx ctx = {0};
    pto_loop_init(&ctx, "tile_loop_row", TILE_ROWS, OFFSET_ROW);
    
    for (int tile = 0; tile < data->num_tiles; tile++) {
        if (pto_loop_should_record(rt, &ctx, tile)) {
            // First iteration: record with row_offset = tile * TILE_ROWS
            int row_off = tile * TILE_ROWS;
            
            // Task 0: RMSNorm - uses base tensors with row offset
            int32_t t0 = pto_task_alloc(rt, "rmsnorm", (void*)rmsnorm_incore, 0, 0);
            pto_task_add_input(rt, t0, data->input, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t0, data->norm_weight, 0, 0, 1, HIDDEN_DIM);
            pto_task_add_output(rt, t0, data->temp_norm, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t0);
            
            // Task 1: Linear
            int32_t t1 = pto_task_alloc(rt, "linear", (void*)linear_incore, 0, 0);
            pto_task_add_input(rt, t1, data->temp_norm, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t1, data->proj_weight, 0, 0, 1, HIDDEN_DIM);
            pto_task_add_output(rt, t1, data->temp_proj, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t1);
            
            // Task 2: Scale
            int32_t t2 = pto_task_alloc(rt, "scale", (void*)scale_incore, 0, 0);
            pto_task_add_input(rt, t2, data->temp_proj, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_output(rt, t2, data->temp_out, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t2);
            
            // Task 3: Residual add
            int32_t t3 = pto_task_alloc(rt, "residual", (void*)residual_add_incore, 0, 0);
            pto_task_add_input(rt, t3, data->input, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t3, data->temp_out, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_output(rt, t3, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t3);
            
            pto_loop_finish_record(rt, &ctx);
        } else {
            // Subsequent iterations: replay with row offset delta
            pto_loop_replay(rt, &ctx, tile);
        }
    }
    
    pto_loop_cleanup(&ctx);
}

// =============================================================================
// Reference Implementation
// =============================================================================

void reference_impl(TestData* data) {
    float eps = 1e-6f;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    int tile_size = TILE_ROWS * HIDDEN_DIM;
    
    for (int tile = 0; tile < data->num_tiles; tile++) {
        int offset = tile * tile_size;
        
        for (int row = 0; row < TILE_ROWS; row++) {
            int row_offset = offset + row * HIDDEN_DIM;
            
            // RMSNorm
            float sum_sq = 0.0f;
            for (int col = 0; col < HIDDEN_DIM; col++) {
                float val = data->input[row_offset + col];
                sum_sq += val * val;
            }
            float inv_rms = 1.0f / sqrtf(sum_sq / HIDDEN_DIM + eps);
            
            for (int col = 0; col < HIDDEN_DIM; col++) {
                data->temp_norm[row_offset + col] = 
                    data->input[row_offset + col] * inv_rms * data->norm_weight[col];
            }
            
            // Linear + Scale
            for (int col = 0; col < HIDDEN_DIM; col++) {
                data->temp_out[row_offset + col] = 
                    data->temp_norm[row_offset + col] * data->proj_weight[col] * scale;
            }
            
            // Residual add
            for (int col = 0; col < HIDDEN_DIM; col++) {
                data->output[row_offset + col] = 
                    data->input[row_offset + col] + data->temp_out[row_offset + col];
            }
        }
    }
}

// =============================================================================
// Verification
// =============================================================================

int verify(const char* name, float* output, float* reference, int total_elements) {
    int errors = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    float tolerance = 1e-5f;
    
    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(output[i] - reference[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        if (diff > tolerance) {
            if (errors < 3) {
                printf("  [%s] Mismatch at %d: output=%.6f, ref=%.6f, diff=%.2e\n",
                       name, i, output[i], reference[i], diff);
            }
            errors++;
        }
    }
    
    printf("  [%s] Max diff: %.2e at index %d\n", name, max_diff, max_diff_idx);
    return errors;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int num_tiles = 64;  // Default
    if (argc > 1) num_tiles = atoi(argv[1]);
    
    int total_elements = num_tiles * TILE_ROWS * HIDDEN_DIM;
    
    printf("================================================================================\n");
    printf("LLaMA Layer Correctness Test - Loop Replay vs Direct\n");
    printf("================================================================================\n");
    printf("Configuration:\n");
    printf("  Num Tiles:       %d\n", num_tiles);
    printf("  Tile Size:       %d x %d = %d elements\n", TILE_ROWS, HIDDEN_DIM, TILE_ROWS * HIDDEN_DIM);
    printf("  Total Elements:  %d (%.1f KB)\n", total_elements, total_elements * 4.0 / 1024);
    printf("  Tasks per tile:  4\n");
    printf("  Total Tasks:     %d\n", num_tiles * 4);
    printf("================================================================================\n\n");
    
    // Allocate buffers
    float* input = (float*)malloc(total_elements * sizeof(float));
    float* norm_weight = (float*)malloc(HIDDEN_DIM * sizeof(float));
    float* proj_weight = (float*)malloc(HIDDEN_DIM * sizeof(float));
    float* temp_norm = (float*)malloc(total_elements * sizeof(float));
    float* temp_proj = (float*)malloc(total_elements * sizeof(float));
    float* temp_out = (float*)malloc(total_elements * sizeof(float));
    
    // Three output buffers for comparison
    float* output_ref = (float*)malloc(total_elements * sizeof(float));
    float* output_direct = (float*)malloc(total_elements * sizeof(float));
    float* output_replay = (float*)malloc(total_elements * sizeof(float));
    
    // Initialize input with deterministic random values
    srand(12345);
    for (int i = 0; i < total_elements; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        norm_weight[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;
        proj_weight[i] = ((float)rand() / RAND_MAX) * 0.2f + 0.9f;
    }
    
    // ==========================================================================
    // Test 1: Reference Implementation
    // ==========================================================================
    printf("Test 1: Reference Implementation (CPU)\n");
    printf("--------------------------------------\n");
    
    memset(output_ref, 0, total_elements * sizeof(float));
    memset(temp_norm, 0, total_elements * sizeof(float));
    memset(temp_out, 0, total_elements * sizeof(float));
    
    TestData ref_data = {
        .input = input,
        .output = output_ref,
        .norm_weight = norm_weight,
        .proj_weight = proj_weight,
        .temp_norm = temp_norm,
        .temp_proj = temp_proj,
        .temp_out = temp_out,
        .num_tiles = num_tiles
    };
    
    double start = get_time_us();
    reference_impl(&ref_data);
    double ref_time = get_time_us() - start;
    
    printf("  Time: %.2f ms\n", ref_time / 1000.0);
    printf("  Sample output[0..3]: %.6f, %.6f, %.6f, %.6f\n", 
           output_ref[0], output_ref[1], output_ref[2], output_ref[3]);
    printf("\n");
    
    // ==========================================================================
    // Test 2: Direct Task Creation (no replay)
    // ==========================================================================
    printf("Test 2: Direct Task Creation (no replay)\n");
    printf("-----------------------------------------\n");
    
    memset(output_direct, 0, total_elements * sizeof(float));
    memset(temp_norm, 0, total_elements * sizeof(float));
    memset(temp_proj, 0, total_elements * sizeof(float));
    memset(temp_out, 0, total_elements * sizeof(float));
    
    TestData direct_data = {
        .input = input,
        .output = output_direct,
        .norm_weight = norm_weight,
        .proj_weight = proj_weight,
        .temp_norm = temp_norm,
        .temp_proj = temp_proj,
        .temp_out = temp_out,
        .num_tiles = num_tiles
    };
    
    start = get_time_us();
    int result = runtime_entry_arm64(orchestration_direct, &direct_data, 4, 0);
    double direct_time = get_time_us() - start;
    
    if (result != 0) {
        printf("  ERROR: runtime_entry_arm64 failed!\n");
        return 1;
    }
    
    printf("  Time: %.2f ms\n", direct_time / 1000.0);
    printf("  Sample output[0..3]: %.6f, %.6f, %.6f, %.6f\n", 
           output_direct[0], output_direct[1], output_direct[2], output_direct[3]);
    
    int errors_direct = verify("Direct", output_direct, output_ref, total_elements);
    printf("  Verification: %s (%d errors)\n\n", errors_direct == 0 ? "PASS" : "FAIL", errors_direct);
    
    // ==========================================================================
    // Test 3: Loop Replay with ROW OFFSET mode
    // ==========================================================================
    printf("Test 3: Loop Replay (OFFSET_ROW mode)\n");
    printf("-------------------------------------\n");
    
    memset(output_replay, 0, total_elements * sizeof(float));
    memset(temp_norm, 0, total_elements * sizeof(float));
    memset(temp_proj, 0, total_elements * sizeof(float));
    memset(temp_out, 0, total_elements * sizeof(float));
    
    TestData replay_data = {
        .input = input,
        .output = output_replay,
        .norm_weight = norm_weight,
        .proj_weight = proj_weight,
        .temp_norm = temp_norm,
        .temp_proj = temp_proj,
        .temp_out = temp_out,
        .num_tiles = num_tiles
    };
    
    start = get_time_us();
    result = runtime_entry_arm64(orchestration_replay_rowoffset, &replay_data, 4, 0);
    double replay_time = get_time_us() - start;
    
    if (result != 0) {
        printf("  ERROR: runtime_entry_arm64 failed!\n");
        return 1;
    }
    
    printf("  Time: %.2f ms\n", replay_time / 1000.0);
    printf("  Sample output[0..3]: %.6f, %.6f, %.6f, %.6f\n", 
           output_replay[0], output_replay[1], output_replay[2], output_replay[3]);
    
    int errors_replay = verify("Replay", output_replay, output_ref, total_elements);
    printf("  Verification: %s (%d errors)\n\n", errors_replay == 0 ? "PASS" : "FAIL", errors_replay);
    
    // ==========================================================================
    // Summary
    // ==========================================================================
    printf("================================================================================\n");
    printf("Summary\n");
    printf("================================================================================\n");
    printf("  Reference:     %.2f ms\n", ref_time / 1000.0);
    printf("  Direct:        %.2f ms (%.2fx ref)\n", direct_time / 1000.0, direct_time / ref_time);
    printf("  Loop Replay:   %.2f ms (%.2fx direct)\n", replay_time / 1000.0, direct_time / replay_time);
    printf("\n");
    printf("  Direct vs Ref:    %s\n", errors_direct == 0 ? "IDENTICAL ✓" : "MISMATCH ✗");
    printf("  Replay vs Ref:    %s\n", errors_replay == 0 ? "IDENTICAL ✓" : "MISMATCH ✗");
    printf("================================================================================\n");
    
    // Detail comparison of a few tiles
    printf("\nDetailed comparison (tile 0, first 4 elements):\n");
    printf("  Reference: %.6f, %.6f, %.6f, %.6f\n", 
           output_ref[0], output_ref[1], output_ref[2], output_ref[3]);
    printf("  Direct:    %.6f, %.6f, %.6f, %.6f\n", 
           output_direct[0], output_direct[1], output_direct[2], output_direct[3]);
    printf("  Replay:    %.6f, %.6f, %.6f, %.6f\n", 
           output_replay[0], output_replay[1], output_replay[2], output_replay[3]);
    
    if (num_tiles > 1) {
        int tile1_offset = TILE_ROWS * HIDDEN_DIM;
        printf("\nDetailed comparison (tile 1, first 4 elements):\n");
        printf("  Reference: %.6f, %.6f, %.6f, %.6f\n", 
               output_ref[tile1_offset], output_ref[tile1_offset+1], 
               output_ref[tile1_offset+2], output_ref[tile1_offset+3]);
        printf("  Direct:    %.6f, %.6f, %.6f, %.6f\n", 
               output_direct[tile1_offset], output_direct[tile1_offset+1], 
               output_direct[tile1_offset+2], output_direct[tile1_offset+3]);
        printf("  Replay:    %.6f, %.6f, %.6f, %.6f\n", 
               output_replay[tile1_offset], output_replay[tile1_offset+1], 
               output_replay[tile1_offset+2], output_replay[tile1_offset+3]);
    }
    
    // Cleanup
    free(input);
    free(norm_weight);
    free(proj_weight);
    free(temp_norm);
    free(temp_proj);
    free(temp_out);
    free(output_ref);
    free(output_direct);
    free(output_replay);
    
    return (errors_direct == 0 && errors_replay == 0) ? 0 : 1;
}
