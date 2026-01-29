/**
 * PTO Runtime2 - Test Suite
 * 
 * Tests the core functionality of the PTO Runtime2 system.
 */

#include "../pto_runtime2.h"
#include "../pto_runtime2_sim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// =============================================================================
// Test Utilities
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    printf("TEST: %s... ", #name); \
    if (test_##name()) { \
        printf("PASSED\n"); \
        tests_passed++; \
    } else { \
        printf("FAILED\n"); \
        tests_failed++; \
    }

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("\n  ASSERT FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return false; \
    } \
} while(0)

// =============================================================================
// Test: Basic Runtime Creation
// =============================================================================

static bool test_runtime_create(void) {
    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_SIMULATE);
    ASSERT(rt != NULL);
    ASSERT(rt->sm_handle != NULL);
    ASSERT(rt->gm_heap != NULL);
    
    pto2_runtime_destroy(rt);
    return true;
}

// =============================================================================
// Test: Shared Memory Layout
// =============================================================================

static bool test_shared_memory(void) {
    PTO2SharedMemoryHandle* sm = pto2_sm_create_default();
    ASSERT(sm != NULL);
    ASSERT(pto2_sm_validate(sm));
    
    // Check initial state
    ASSERT(sm->header->current_task_index == 0);
    ASSERT(sm->header->last_task_alive == 0);
    ASSERT(sm->header->heap_top == 0);
    ASSERT(sm->header->heap_tail == 0);
    
    pto2_sm_destroy(sm);
    return true;
}

// =============================================================================
// Test: TensorMap Operations
// =============================================================================

static bool test_tensormap(void) {
    PTO2TensorMap tm;
    ASSERT(pto2_tensormap_init_default(&tm));
    
    // Insert some entries
    int dummy_buffer;
    PTO2TensorRegion region1 = {&dummy_buffer, 0, 0, 1024};
    PTO2TensorRegion region2 = {&dummy_buffer, 1, 0, 1024};
    PTO2TensorRegion region3 = {&dummy_buffer, 2, 0, 1024};
    
    pto2_tensormap_insert(&tm, &region1, 0);
    pto2_tensormap_insert(&tm, &region2, 1);
    pto2_tensormap_insert(&tm, &region3, 2);
    
    // Lookup
    ASSERT(pto2_tensormap_lookup(&tm, &region1) == 0);
    ASSERT(pto2_tensormap_lookup(&tm, &region2) == 1);
    ASSERT(pto2_tensormap_lookup(&tm, &region3) == 2);
    
    // Non-existent lookup
    PTO2TensorRegion region_nx = {&dummy_buffer, 99, 0, 1024};
    ASSERT(pto2_tensormap_lookup(&tm, &region_nx) == -1);
    
    // Test lazy invalidation
    pto2_tensormap_sync_validity(&tm, 2);  // Invalidate tasks 0, 1
    ASSERT(pto2_tensormap_lookup(&tm, &region1) == -1);  // Stale
    ASSERT(pto2_tensormap_lookup(&tm, &region2) == -1);  // Stale
    ASSERT(pto2_tensormap_lookup(&tm, &region3) == 2);   // Still valid
    
    pto2_tensormap_destroy(&tm);
    return true;
}

// =============================================================================
// Test: Ring Buffer Operations
// =============================================================================

static bool test_ring_buffer(void) {
    // Test HeapRing
    void* heap = malloc(1024);
    ASSERT(heap != NULL);
    
    int32_t tail = 0;
    PTO2HeapRing hr;
    pto2_heap_ring_init(&hr, heap, 1024, &tail);
    
    // Allocate some buffers (sizes are aligned to 64 bytes)
    void* buf1 = pto2_heap_ring_try_alloc(&hr, 128);
    ASSERT(buf1 != NULL);
    ASSERT(buf1 == heap);
    
    void* buf2 = pto2_heap_ring_try_alloc(&hr, 256);
    ASSERT(buf2 != NULL);
    
    // Check that space decreased (aligned sizes: 128 and 256 = 384 total)
    int32_t avail = pto2_heap_ring_available(&hr);
    ASSERT(avail > 0);
    ASSERT(avail < 1024);  // Less than total
    
    free(heap);
    
    // Test DepListPool
    PTO2DepListEntry pool[100];
    PTO2DepListPool dp;
    pto2_dep_pool_init(&dp, pool, 100);
    
    // Test prepend
    int32_t head = 0;
    head = pto2_dep_list_prepend(&dp, head, 10);
    head = pto2_dep_list_prepend(&dp, head, 20);
    head = pto2_dep_list_prepend(&dp, head, 30);
    
    ASSERT(pto2_dep_list_count(&dp, head) == 3);
    
    // Verify order (prepend gives reverse order)
    PTO2DepListEntry* e = pto2_dep_pool_get(&dp, head);
    ASSERT(e != NULL && e->task_id == 30);
    
    return true;
}

// =============================================================================
// Test: Scope Management
// =============================================================================

static bool test_scope_management(void) {
    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_GRAPH_ONLY);
    ASSERT(rt != NULL);
    
    // Test scope nesting
    ASSERT(pto2_get_scope_depth(&rt->orchestrator) == 0);
    
    pto2_rt_scope_begin(rt);
    ASSERT(pto2_get_scope_depth(&rt->orchestrator) == 1);
    
    pto2_rt_scope_begin(rt);
    ASSERT(pto2_get_scope_depth(&rt->orchestrator) == 2);
    
    pto2_rt_scope_end(rt);
    ASSERT(pto2_get_scope_depth(&rt->orchestrator) == 1);
    
    pto2_rt_scope_end(rt);
    ASSERT(pto2_get_scope_depth(&rt->orchestrator) == 0);
    
    pto2_runtime_destroy(rt);
    return true;
}

// =============================================================================
// Test: Task Submission
// =============================================================================

static bool test_task_submission(void) {
    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_GRAPH_ONLY);
    ASSERT(rt != NULL);
    
    // Create some dummy buffers
    int buf_A[256], buf_B[256], buf_C[256];
    
    pto2_rt_scope_begin(rt);
    
    // Submit task with inputs and outputs
    PTO2TaskParam params1[] = {
        PTO2_INPUT(buf_A, 0, 1024),
        PTO2_INPUT(buf_B, 0, 1024),
        PTO2_OUTPUT(buf_C, 0, 1024)
    };
    
    int32_t task1 = pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "gemm_tile",
                                         params1, 3);
    ASSERT(task1 >= 0);
    
    // Submit another task that depends on first
    PTO2TaskParam params2[] = {
        PTO2_INPUT(buf_C, 0, 1024),  // Should create dependency on task1
        PTO2_OUTPUT(buf_C, 1, 1024)
    };
    
    int32_t task2 = pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR, NULL, "vector_add",
                                         params2, 2);
    ASSERT(task2 >= 0);
    ASSERT(task2 == task1 + 1);
    
    pto2_rt_scope_end(rt);
    
    // Verify task count
    ASSERT(rt->orchestrator.tasks_submitted == 2);
    
    pto2_runtime_destroy(rt);
    return true;
}

// =============================================================================
// Test: Simple BGEMM Pattern
// =============================================================================

static bool test_bgemm_pattern(void) {
    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_SIMULATE);
    ASSERT(rt != NULL);
    
    // BGEMM: C[m,n] += A[m,k] * B[k,n]
    // Simplified: 2 GEMM tiles + accumulation
    
    int A[4][256], B[4][256], C[2][2][256], P[2][2][256];
    
    pto2_rt_scope_begin(rt);
    
    for (int k = 0; k < 2; k++) {
        for (int m = 0; m < 2; m++) {
            for (int n = 0; n < 2; n++) {
                // P[m,n] = A[m,k] * B[k,n]
                PTO2TaskParam gemm_params[] = {
                    PTO2_INPUT(&A[m*2+k], 0, 1024),
                    PTO2_INPUT(&B[k*2+n], 0, 1024),
                    PTO2_OUTPUT(&P[m][n], k, 1024)
                };
                
                int32_t gemm_task = pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE,
                                                        NULL, "gemm_tile",
                                                        gemm_params, 3);
                ASSERT(gemm_task >= 0);
                
                // C[m,n] += P[m,n]
                PTO2TaskParam add_params[] = {
                    PTO2_INPUT(&P[m][n], k, 1024),
                    PTO2_INOUT(&C[m][n], 0, 1024)
                };
                
                int32_t add_task = pto2_rt_submit_task(rt, 0, PTO2_WORKER_VECTOR,
                                                       NULL, "tile_add",
                                                       add_params, 2);
                ASSERT(add_task >= 0);
            }
        }
    }
    
    pto2_rt_scope_end(rt);
    
    // Mark orchestration done
    pto2_rt_orchestration_done(rt);
    
    // Execute
    pto2_runtime_execute(rt);
    
    // Verify completion
    ASSERT(pto2_runtime_is_done(rt));
    ASSERT(rt->orchestrator.tasks_submitted == 16);  // 2*2*2*2 = 16 tasks
    
    pto2_runtime_destroy(rt);
    return true;
}

// =============================================================================
// Test: Simulation
// =============================================================================

static bool test_simulation(void) {
    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_GRAPH_ONLY);
    ASSERT(rt != NULL);
    
    // Submit some tasks
    int buf[4][256];
    
    pto2_rt_scope_begin(rt);
    
    for (int i = 0; i < 4; i++) {
        PTO2TaskParam params[] = {
            PTO2_OUTPUT(&buf[i], 0, 1024)
        };
        
        int32_t task_id = pto2_rt_submit_task(rt, 0, 
                                              i % 2 ? PTO2_WORKER_CUBE : PTO2_WORKER_VECTOR,
                                              NULL, 
                                              i % 2 ? "gemm_tile" : "vector_op",
                                              params, 1);
        ASSERT(task_id >= 0);
    }
    
    pto2_rt_scope_end(rt);
    pto2_rt_orchestration_done(rt);
    
    // Create simulation state
    PTO2SimState* sim = pto2_sim_create_default();
    ASSERT(sim != NULL);
    
    // Run simulation
    int64_t makespan = pto2_sim_run(sim, rt);
    ASSERT(makespan > 0);
    
    pto2_sim_destroy(sim);
    pto2_runtime_destroy(rt);
    return true;
}

// =============================================================================
// Test: Validation
// =============================================================================

static bool test_validation(void) {
    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_GRAPH_ONLY);
    ASSERT(rt != NULL);
    
    ASSERT(pto2_runtime_validate(rt));
    
    pto2_runtime_destroy(rt);
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(void) {
    printf("\n========== PTO Runtime2 Test Suite ==========\n\n");
    
    TEST(runtime_create);
    TEST(shared_memory);
    TEST(tensormap);
    TEST(ring_buffer);
    TEST(scope_management);
    TEST(task_submission);
    TEST(bgemm_pattern);
    TEST(simulation);
    TEST(validation);
    
    printf("\n==============================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("==============================================\n\n");
    
    return tests_failed > 0 ? 1 : 0;
}
