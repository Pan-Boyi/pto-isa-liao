/**
 * Test replay validation - verifies arguments are compatible with recorded template
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../pto_runtime.h"
#include "../pto_runtime.c"

void dummy_incore(void** args, int num_args) {}

// Test 1: Valid replay pattern (same tensor, linear offset)
void test_valid_pattern() {
    printf("Test 1: Valid Replay Pattern\n");
    printf("=============================\n");
    
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    pto_runtime_init(rt);
    
    float* Q = (float*)malloc(1024 * 128 * sizeof(float));
    float* O = (float*)malloc(1024 * 128 * sizeof(float));
    
    LoopReplayCtx ctx = {0};
    pto_loop_init(&ctx, "test_loop", 32, OFFSET_ROW);
    
    // Record iteration 0
    pto_loop_should_record(rt, &ctx, 0);
    int32_t t0 = pto_task_alloc(rt, "test_task", (void*)dummy_incore, 0, 0);
    pto_task_add_input(rt, t0, Q, 0, 0, 32, 128);   // Q[0:32, :]
    pto_task_add_output(rt, t0, O, 0, 0, 32, 128);  // O[0:32, :]
    pto_task_submit(rt, t0);
    pto_loop_finish_record(rt, &ctx);
    
    printf("Recorded at iteration 0: Q[0:32, :], O[0:32, :]\n\n");
    
    // Validate iteration 1 (should pass)
    printf("Validating iteration 1 (expect Q[32:64, :], O[32:64, :])...\n");
    TaskArg args_iter1[2] = {
        {.region = {Q, 32, 0, 32, 128}, .is_output = false},  // Q[32:64, :]
        {.region = {O, 32, 0, 32, 128}, .is_output = true}    // O[32:64, :]
    };
    bool valid1 = pto_loop_validate_task(&ctx, 0, 1, args_iter1, 2);
    printf("  Result: %s\n\n", valid1 ? "PASS" : "FAIL");
    
    // Validate iteration 10 (should pass)
    printf("Validating iteration 10 (expect Q[320:352, :], O[320:352, :])...\n");
    TaskArg args_iter10[2] = {
        {.region = {Q, 320, 0, 32, 128}, .is_output = false},
        {.region = {O, 320, 0, 32, 128}, .is_output = true}
    };
    bool valid10 = pto_loop_validate_task(&ctx, 0, 10, args_iter10, 2);
    printf("  Result: %s\n\n", valid10 ? "PASS" : "FAIL");
    
    pto_loop_cleanup(&ctx);
    free(Q); free(O);
    pto_runtime_shutdown(rt);
    free(rt);
}

// Test 2: Invalid - different tensor pointer
void test_invalid_tensor_pointer() {
    printf("\nTest 2: Invalid - Different Tensor Pointer\n");
    printf("============================================\n");
    
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    pto_runtime_init(rt);
    
    float* Q1 = (float*)malloc(1024 * 128 * sizeof(float));
    float* Q2 = (float*)malloc(1024 * 128 * sizeof(float));  // Different buffer!
    float* O = (float*)malloc(1024 * 128 * sizeof(float));
    
    LoopReplayCtx ctx = {0};
    pto_loop_init(&ctx, "test_loop", 32, OFFSET_ROW);
    
    // Record with Q1
    pto_loop_should_record(rt, &ctx, 0);
    int32_t t0 = pto_task_alloc(rt, "test_task", (void*)dummy_incore, 0, 0);
    pto_task_add_input(rt, t0, Q1, 0, 0, 32, 128);
    pto_task_add_output(rt, t0, O, 0, 0, 32, 128);
    pto_task_submit(rt, t0);
    pto_loop_finish_record(rt, &ctx);
    
    printf("Recorded with Q1=%p\n\n", Q1);
    
    // Try to validate with Q2 (should FAIL)
    printf("Validating with Q2=%p (different pointer, should FAIL)...\n", Q2);
    TaskArg args_bad[2] = {
        {.region = {Q2, 32, 0, 32, 128}, .is_output = false},  // Wrong tensor!
        {.region = {O, 32, 0, 32, 128}, .is_output = true}
    };
    bool valid = pto_loop_validate_task(&ctx, 0, 1, args_bad, 2);
    printf("  Result: %s (expected FAIL)\n", valid ? "UNEXPECTED PASS" : "CORRECTLY FAILED");
    
    pto_loop_cleanup(&ctx);
    free(Q1); free(Q2); free(O);
    pto_runtime_shutdown(rt);
    free(rt);
}

// Test 3: Invalid - wrong row offset
void test_invalid_offset() {
    printf("\nTest 3: Invalid - Wrong Row Offset\n");
    printf("===================================\n");
    
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    pto_runtime_init(rt);
    
    float* Q = (float*)malloc(1024 * 128 * sizeof(float));
    float* O = (float*)malloc(1024 * 128 * sizeof(float));
    
    LoopReplayCtx ctx = {0};
    pto_loop_init(&ctx, "test_loop", 32, OFFSET_ROW);  // stride = 32
    
    // Record at iteration 0
    pto_loop_should_record(rt, &ctx, 0);
    int32_t t0 = pto_task_alloc(rt, "test_task", (void*)dummy_incore, 0, 0);
    pto_task_add_input(rt, t0, Q, 0, 0, 32, 128);
    pto_task_add_output(rt, t0, O, 0, 0, 32, 128);
    pto_task_submit(rt, t0);
    pto_loop_finish_record(rt, &ctx);
    
    printf("Recorded at iter 0: row_offset=0, stride=32\n\n");
    
    // Try with wrong offset (40 instead of 32)
    printf("Validating iter 1 with row_offset=40 (expected 32, should FAIL)...\n");
    TaskArg args_bad[2] = {
        {.region = {Q, 40, 0, 32, 128}, .is_output = false},  // Wrong offset!
        {.region = {O, 40, 0, 32, 128}, .is_output = true}
    };
    bool valid = pto_loop_validate_task(&ctx, 0, 1, args_bad, 2);
    printf("  Result: %s (expected FAIL)\n", valid ? "UNEXPECTED PASS" : "CORRECTLY FAILED");
    
    pto_loop_cleanup(&ctx);
    free(Q); free(O);
    pto_runtime_shutdown(rt);
    free(rt);
}

// Test 4: Invalid - different tile shape
void test_invalid_shape() {
    printf("\nTest 4: Invalid - Different Tile Shape\n");
    printf("=======================================\n");
    
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    pto_runtime_init(rt);
    
    float* Q = (float*)malloc(1024 * 128 * sizeof(float));
    float* O = (float*)malloc(1024 * 128 * sizeof(float));
    
    LoopReplayCtx ctx = {0};
    pto_loop_init(&ctx, "test_loop", 32, OFFSET_ROW);
    
    // Record with shape [32, 128]
    pto_loop_should_record(rt, &ctx, 0);
    int32_t t0 = pto_task_alloc(rt, "test_task", (void*)dummy_incore, 0, 0);
    pto_task_add_input(rt, t0, Q, 0, 0, 32, 128);
    pto_task_add_output(rt, t0, O, 0, 0, 32, 128);
    pto_task_submit(rt, t0);
    pto_loop_finish_record(rt, &ctx);
    
    printf("Recorded with shape [32, 128]\n\n");
    
    // Try with different shape [16, 128] (last tile is smaller)
    printf("Validating with shape [16, 128] (should FAIL)...\n");
    TaskArg args_bad[2] = {
        {.region = {Q, 32, 0, 16, 128}, .is_output = false},  // Wrong rows!
        {.region = {O, 32, 0, 16, 128}, .is_output = true}
    };
    bool valid = pto_loop_validate_task(&ctx, 0, 1, args_bad, 2);
    printf("  Result: %s (expected FAIL)\n", valid ? "UNEXPECTED PASS" : "CORRECTLY FAILED");
    
    pto_loop_cleanup(&ctx);
    free(Q); free(O);
    pto_runtime_shutdown(rt);
    free(rt);
}

int main() {
    printf("Replay Argument Validation Tests\n");
    printf("=================================\n\n");
    
    test_valid_pattern();
    test_invalid_tensor_pointer();
    test_invalid_offset();
    test_invalid_shape();
    
    printf("\n=================================\n");
    printf("All tests completed.\n");
    printf("\nKey takeaway: Replay is ONLY correct when:\n");
    printf("  1. Same tensor pointer every iteration\n");
    printf("  2. Same tile shape (rows, cols) every iteration\n");
    printf("  3. Row offset increases by a fixed stride\n");
    printf("  4. Col offset is constant\n");
    
    return 0;
}
