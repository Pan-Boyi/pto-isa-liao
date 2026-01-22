/**
 * Test: LLaMA Cycle Simulation with Trace Generation
 * 
 * This test runs LLaMA layer orchestration in simulation mode,
 * where InCore functions return cycle counts instead of computing.
 * The trace is saved for visualization in Chrome Tracing.
 * 
 * Build:
 *   cc -O2 -I. -DSEQ_LEN=256 examples/test_llama_cycle_sim.c \
 *      -o test_llama_sim
 * 
 * Run:
 *   ./test_llama_sim
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Include runtime
#include "pto_runtime.h"

// Simulation functions from ascend_a2a3_sim backend
// These return cycle counts instead of computing

// Flash Attention functions
#include "examples/output_ascend_a2a3_sim/llama7b/flash_attn_init_state.c"
#include "examples/output_ascend_a2a3_sim/llama7b/flash_attn_normalize.c"
#include "examples/output_ascend_a2a3_sim/llama7b/flash_attn_output_update.c"
#include "examples/output_ascend_a2a3_sim/llama7b/flash_attn_score_block.c"
#include "examples/output_ascend_a2a3_sim/llama7b/flash_attn_softmax_update.c"

// Residual add variants
#include "examples/output_ascend_a2a3_sim/llama7b/residual_add_tile.c"
#include "examples/output_ascend_a2a3_sim/llama7b/residual_add_tile_64.c"
#include "examples/output_ascend_a2a3_sim/llama7b/residual_add_tile_128.c"
#include "examples/output_ascend_a2a3_sim/llama7b/residual_add_tile_256.c"

// RMSNorm variants
#include "examples/output_ascend_a2a3_sim/llama7b/rmsnorm_tile.c"
#include "examples/output_ascend_a2a3_sim/llama7b/rmsnorm_tile_64.c"
#include "examples/output_ascend_a2a3_sim/llama7b/rmsnorm_tile_128.c"
#include "examples/output_ascend_a2a3_sim/llama7b/rmsnorm_tile_256.c"

// RoPE variants
#include "examples/output_ascend_a2a3_sim/llama7b/rope_tile.c"
#include "examples/output_ascend_a2a3_sim/llama7b/rope_tile_64.c"
#include "examples/output_ascend_a2a3_sim/llama7b/rope_tile_128.c"
#include "examples/output_ascend_a2a3_sim/llama7b/rope_tile_256.c"

// SwiGLU variants
#include "examples/output_ascend_a2a3_sim/llama7b/swiglu_tile.c"
#include "examples/output_ascend_a2a3_sim/llama7b/swiglu_tile_64.c"
#include "examples/output_ascend_a2a3_sim/llama7b/swiglu_tile_128.c"
#include "examples/output_ascend_a2a3_sim/llama7b/swiglu_tile_256.c"

// MatMul variants
#include "examples/output_ascend_a2a3_sim/llama7b/tile_matmul.c"
#include "examples/output_ascend_a2a3_sim/llama7b/tile_matmul_64.c"
#include "examples/output_ascend_a2a3_sim/llama7b/tile_matmul_128.c"
#include "examples/output_ascend_a2a3_sim/llama7b/tile_matmul_256.c"

// Include the orchestration function (which includes pto_runtime.c)
#define PTO_NO_MAIN
#include "examples/output_arm64/llama7b/llama_layer_dynamic_orchestration.c"

// Cycle function registry
typedef struct {
    const char* name;
    CycleCostFunc func;
} CycleFuncEntry;

static CycleFuncEntry cycle_func_registry[] = {
    // Flash Attention
    {"flash_attn_init_state", flash_attn_init_state_cycle_count},
    {"flash_attn_normalize", flash_attn_normalize_cycle_count},
    {"flash_attn_output_update", flash_attn_output_update_cycle_count},
    {"flash_attn_score_block", flash_attn_score_block_cycle_count},
    {"flash_attn_softmax_update", flash_attn_softmax_update_cycle_count},
    
    // Residual add
    {"residual_add_tile", residual_add_tile_cycle_count},
    {"residual_add_tile_64", residual_add_tile_64_cycle_count},
    {"residual_add_tile_128", residual_add_tile_128_cycle_count},
    {"residual_add_tile_256", residual_add_tile_256_cycle_count},
    
    // RMSNorm
    {"rmsnorm_tile", rmsnorm_tile_cycle_count},
    {"rmsnorm_tile_64", rmsnorm_tile_64_cycle_count},
    {"rmsnorm_tile_128", rmsnorm_tile_128_cycle_count},
    {"rmsnorm_tile_256", rmsnorm_tile_256_cycle_count},
    
    // RoPE
    {"rope_tile", rope_tile_cycle_count},
    {"rope_tile_64", rope_tile_64_cycle_count},
    {"rope_tile_128", rope_tile_128_cycle_count},
    {"rope_tile_256", rope_tile_256_cycle_count},
    
    // SwiGLU
    {"swiglu_tile", swiglu_tile_cycle_count},
    {"swiglu_tile_64", swiglu_tile_64_cycle_count},
    {"swiglu_tile_128", swiglu_tile_128_cycle_count},
    {"swiglu_tile_256", swiglu_tile_256_cycle_count},
    
    // MatMul
    {"tile_matmul", tile_matmul_cycle_count},
    {"tile_matmul_64", tile_matmul_64_cycle_count},
    {"tile_matmul_128", tile_matmul_128_cycle_count},
    {"tile_matmul_256", tile_matmul_256_cycle_count},
    
    {NULL, NULL}
};

CycleCostFunc find_cycle_func(const char* name) {
    for (int i = 0; cycle_func_registry[i].name != NULL; i++) {
        if (strcmp(cycle_func_registry[i].name, name) == 0) {
            return cycle_func_registry[i].func;
        }
    }
    return NULL;
}

// LLaMA configuration
#define TILE_ROWS 32
#define TILE_COLS 128
#define HIDDEN_DIM 4096
#define NUM_HEADS 32
#define HEAD_DIM 128

#ifndef SEQ_LEN
#define SEQ_LEN 256
#endif

// Forward declaration of orchestration function
extern void llama_layer_dynamic(PTORuntime* rt, 
    float* input, float* output, 
    float* attn_norm_weights, float* wq, float* wk, float* wv, float* wo,
    float* cos_cache, float* sin_cache,
    float* mlp_norm_weights, float* w_gate, float* w_up, float* w_down,
    float* all_q_tiles, float* all_k_tiles, float* all_v_tiles,
    float* all_q_rope, float* all_k_rope, float* all_attn_out,
    float* all_m_vec, float* all_l_vec, float* all_hidden,
    float* temp_norm, float* temp_scores, float* temp_attn_weights,
    float* temp_scale, float* temp_gate, float* temp_up, float* temp_swiglu, float* temp_mlp_out,
    float* const_zeros_large, float* const_zeros_small, float* const_neg_inf,
    int32_t seq_len, int32_t num_tiles);

// Custom task alloc that also sets cycle function
static int32_t (*original_task_alloc)(PTORuntime*, const char*, void*, int32_t, int32_t) = NULL;

int main(int argc, char** argv) {
    int seq_len = SEQ_LEN;
    int num_tiles = seq_len / TILE_ROWS;
    
    // A2/A3 worker configuration
    int num_vector_workers = 48;  // Vector workers for is_cube=0 tasks
    int num_cube_workers = 24;    // Cube workers for is_cube=1 tasks (matmul)
    int num_workers = num_vector_workers + num_cube_workers;
    
    printf("=== LLaMA Cycle Simulation (A2A3 Mode) ===\n");
    printf("Sequence length: %d\n", seq_len);
    printf("Number of tiles: %d\n", num_tiles);
    printf("Vector workers (is_cube=0): %d\n", num_vector_workers);
    printf("Cube workers (is_cube=1): %d\n", num_cube_workers);
    printf("Total workers: %d\n", num_workers);
    printf("\n");
    
    // Allocate all buffers (dummy data, we only count cycles)
    size_t hidden_size = seq_len * HIDDEN_DIM;
    size_t weight_size = HIDDEN_DIM * HIDDEN_DIM;
    size_t tile_size = TILE_ROWS * TILE_COLS;
    
    float* input = (float*)calloc(hidden_size, sizeof(float));
    float* output = (float*)calloc(hidden_size, sizeof(float));
    float* attn_norm = (float*)calloc(HIDDEN_DIM, sizeof(float));
    float* wq = (float*)calloc(weight_size, sizeof(float));
    float* wk = (float*)calloc(weight_size, sizeof(float));
    float* wv = (float*)calloc(weight_size, sizeof(float));
    float* wo = (float*)calloc(weight_size, sizeof(float));
    float* cos_cache = (float*)calloc(seq_len * HEAD_DIM, sizeof(float));
    float* sin_cache = (float*)calloc(seq_len * HEAD_DIM, sizeof(float));
    float* mlp_norm = (float*)calloc(HIDDEN_DIM, sizeof(float));
    float* w_gate = (float*)calloc(weight_size, sizeof(float));
    float* w_up = (float*)calloc(weight_size, sizeof(float));
    float* w_down = (float*)calloc(weight_size, sizeof(float));
    float* q_tiles = (float*)calloc(hidden_size, sizeof(float));
    float* k_tiles = (float*)calloc(hidden_size, sizeof(float));
    float* v_tiles = (float*)calloc(hidden_size, sizeof(float));
    float* q_rope = (float*)calloc(hidden_size, sizeof(float));
    float* k_rope = (float*)calloc(hidden_size, sizeof(float));
    float* attn_out = (float*)calloc(hidden_size, sizeof(float));
    float* m_vec = (float*)calloc(seq_len * NUM_HEADS, sizeof(float));
    float* l_vec = (float*)calloc(seq_len * NUM_HEADS, sizeof(float));
    float* hidden = (float*)calloc(hidden_size, sizeof(float));
    float* temp_norm_buf = (float*)calloc(hidden_size, sizeof(float));
    float* temp_scores = (float*)calloc(seq_len * seq_len, sizeof(float));
    float* temp_weights = (float*)calloc(seq_len * seq_len, sizeof(float));
    float* temp_scale = (float*)calloc(hidden_size, sizeof(float));
    float* temp_gate = (float*)calloc(hidden_size, sizeof(float));
    float* temp_up = (float*)calloc(hidden_size, sizeof(float));
    float* temp_swiglu = (float*)calloc(hidden_size, sizeof(float));
    float* temp_mlp = (float*)calloc(hidden_size, sizeof(float));
    float* zeros_large = (float*)calloc(hidden_size, sizeof(float));
    float* zeros_small = (float*)calloc(seq_len, sizeof(float));
    float* neg_inf = (float*)calloc(seq_len, sizeof(float));
    
    // Initialize runtime
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    pto_runtime_init(rt);
    
    // Enable A2A3 simulation mode with dual queue
    pto_runtime_enable_a2a3_sim(rt, num_vector_workers, num_cube_workers);
    
    // Disable record and replay mode
    pto_set_record_replay(0);
    printf("Record/Replay mode: DISABLED\n");
    
    printf("Building task graph...\n");
    
    // Run orchestration to build task graph
    llama_layer_dynamic(rt, input, output,
        attn_norm, wq, wk, wv, wo,
        cos_cache, sin_cache,
        mlp_norm, w_gate, w_up, w_down,
        q_tiles, k_tiles, v_tiles,
        q_rope, k_rope, attn_out,
        m_vec, l_vec, hidden,
        temp_norm_buf, temp_scores, temp_weights,
        temp_scale, temp_gate, temp_up, temp_swiglu, temp_mlp,
        zeros_large, zeros_small, neg_inf,
        seq_len, num_tiles);
    
    printf("Task graph built: %lld tasks\n", (long long)rt->total_tasks_scheduled);
    
    // Set cycle functions and is_cube flag for all tasks
    printf("Registering cycle functions and worker types...\n");
    int cycle_funcs_set = 0;
    int cube_task_count = 0;
    for (int i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        if (task->is_active && task->func_name) {
            CycleCostFunc cf = find_cycle_func(task->func_name);
            if (cf) {
                task->cycle_func = cf;
                cycle_funcs_set++;
            }
            
            // Set is_cube=true for matmul functions (they need cube unit)
            // This is a workaround since the orchestration code was generated before is_cube support
            if (strstr(task->func_name, "matmul") != NULL ||
                strstr(task->func_name, "score_block") != NULL) {
                task->is_cube = true;
                cube_task_count++;
            }
        }
        
        // Also check compact tasks
        CompactTask* ct = &rt->compact_task[i];
        if (ct->template_ref && ct->template_ref->func_name) {
            CycleCostFunc cf = find_cycle_func(ct->template_ref->func_name);
            if (cf) {
                // For replay tasks, we need to set it on the template
                // This is a bit awkward but works for simulation
                ((RecordedTask*)ct->template_ref)->cycle_func = cf;
                cycle_funcs_set++;
            }
            
            // Set is_cube for replay tasks too
            if (strstr(ct->template_ref->func_name, "matmul") != NULL ||
                strstr(ct->template_ref->func_name, "score_block") != NULL) {
                ((RecordedTask*)ct->template_ref)->is_cube = true;
                cube_task_count++;
            }
        }
    }
    printf("Cycle functions registered: %d\n", cycle_funcs_set);
    printf("Tasks marked as cube (matmul/score_block): %d\n", cube_task_count);
    
    // Analyze cube task fanin distribution
    int cube_fanin_0 = 0, cube_fanin_1 = 0, cube_fanin_gt1 = 0;
    for (int i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        if (task->is_cube) {
            if (task->fanin == 0) cube_fanin_0++;
            else if (task->fanin == 1) cube_fanin_1++;
            else cube_fanin_gt1++;
        }
    }
    printf("Cube task fanin distribution: fanin=0: %d, fanin=1: %d, fanin>1: %d\n",
           cube_fanin_0, cube_fanin_1, cube_fanin_gt1);
    
    // Analyze vector task fanout to cube tasks
    int vec_fanout_max = 0, vec_fanout_to_cube_max = 0;
    int vec_task_id_max_fanout = -1;
    for (int i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        if (!task->is_cube) {
            if (task->fanout_count > vec_fanout_max) {
                vec_fanout_max = task->fanout_count;
                vec_task_id_max_fanout = i;
            }
            int fanout_to_cube = 0;
            for (int j = 0; j < task->fanout_count; j++) {
                int dep_id = task->fanout[j];
                if (dep_id >= 0 && dep_id < rt->next_task_id && rt->pend_task[dep_id].is_cube) {
                    fanout_to_cube++;
                }
            }
            if (fanout_to_cube > vec_fanout_to_cube_max) {
                vec_fanout_to_cube_max = fanout_to_cube;
            }
        }
    }
    printf("Vector task max fanout: %d (task %d), max fanout to cube: %d\n",
           vec_fanout_max, vec_task_id_max_fanout, vec_fanout_to_cube_max);
    
    // How many vector tasks have fanout to cube?
    int vec_with_cube_fanout = 0;
    int total_cube_fanout = 0;
    for (int i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        if (!task->is_cube) {
            int fanout_to_cube = 0;
            for (int j = 0; j < task->fanout_count; j++) {
                int dep_id = task->fanout[j];
                if (dep_id >= 0 && dep_id < rt->next_task_id && rt->pend_task[dep_id].is_cube) {
                    fanout_to_cube++;
                }
            }
            if (fanout_to_cube > 0) {
                vec_with_cube_fanout++;
                total_cube_fanout += fanout_to_cube;
            }
        }
    }
    printf("Vector tasks with cube fanout: %d, total cube dependencies: %d\n",
           vec_with_cube_fanout, total_cube_fanout);
    
    // Check reverse: cube tasks with vector fanout
    int cube_with_vec_fanout = 0;
    int total_vec_from_cube = 0;
    for (int i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        if (task->is_cube) {
            int fanout_to_vec = 0;
            for (int j = 0; j < task->fanout_count; j++) {
                int dep_id = task->fanout[j];
                if (dep_id >= 0 && dep_id < rt->next_task_id && !rt->pend_task[dep_id].is_cube) {
                    fanout_to_vec++;
                }
            }
            if (fanout_to_vec > 0) {
                cube_with_vec_fanout++;
                total_vec_from_cube += fanout_to_vec;
            }
        }
    }
    printf("Cube tasks with vector fanout: %d, total vector deps from cube: %d\n",
           cube_with_vec_fanout, total_vec_from_cube);
    
    // Direct count from cube fanout (already computed above)
    // total_vec_from_cube tells us how many vector tasks are downstream of cube
    printf("(This equals vector tasks with cube producer)\n");
    
    // Note: Since dual_queue_mode is enabled, tasks are already in the correct queues
    // We just need to check the counts
    printf("Ready queues: vector=%d, cube=%d\n", 
           rt->vector_ready_count, rt->cube_ready_count);
    
    // However, the is_cube flag was set AFTER tasks were queued, so we need to rebuild
    // First, collect all ready tasks from both queues
    int32_t all_ready[PTO_MAX_READY_QUEUE];
    int all_ready_count = 0;
    
    // Drain vector queue
    while (rt->vector_ready_count > 0) {
        all_ready[all_ready_count++] = rt->vector_ready_queue[rt->vector_ready_head];
        rt->vector_ready_head = (rt->vector_ready_head + 1) % PTO_MAX_READY_QUEUE;
        rt->vector_ready_count--;
    }
    
    // Drain cube queue
    while (rt->cube_ready_count > 0) {
        all_ready[all_ready_count++] = rt->cube_ready_queue[rt->cube_ready_head];
        rt->cube_ready_head = (rt->cube_ready_head + 1) % PTO_MAX_READY_QUEUE;
        rt->cube_ready_count--;
    }
    
    // Reset queue pointers
    rt->vector_ready_head = 0;
    rt->vector_ready_tail = 0;
    rt->cube_ready_head = 0;
    rt->cube_ready_tail = 0;
    
    // Re-add tasks to correct queues based on updated is_cube flag
    for (int i = 0; i < all_ready_count; i++) {
        int task_id = all_ready[i];
        bool is_cube = false;
        
        CompactTask* ct = &rt->compact_task[task_id];
        if (ct->template_ref != NULL) {
            is_cube = ct->template_ref->is_cube;
        } else {
            is_cube = rt->pend_task[task_id].is_cube;
        }
        
        if (is_cube) {
            rt->cube_ready_queue[rt->cube_ready_tail++] = task_id;
            rt->cube_ready_count++;
        } else {
            rt->vector_ready_queue[rt->vector_ready_tail++] = task_id;
            rt->vector_ready_count++;
        }
    }
    printf("Ready queues redistributed: total=%d -> vector=%d, cube=%d\n", 
           all_ready_count, rt->vector_ready_count, rt->cube_ready_count);
    
    // Execute tasks in simulation mode with dual queue
    printf("\nRunning simulation with dual queue (vector/cube workers)...\n");
    int32_t task_id;
    int tasks_executed = 0;
    int vector_tasks = 0;
    int cube_tasks = 0;
    int vector_worker_idx = 0;  // Cycle through vector workers 0-47
    int cube_worker_idx = 0;    // Cycle through cube workers 48-71
    
    // Track peak queue sizes
    int peak_vector_queue = rt->vector_ready_count;
    int peak_cube_queue = rt->cube_ready_count;
    
    // Process both queues until both are empty
    while (1) {
        bool did_work = false;
        
        // Track peak sizes
        if (rt->vector_ready_count > peak_vector_queue) peak_vector_queue = rt->vector_ready_count;
        if (rt->cube_ready_count > peak_cube_queue) peak_cube_queue = rt->cube_ready_count;
        
        // Try to get a vector task (is_cube=0)
        task_id = pto_get_ready_task_vector(rt);
        if (task_id >= 0) {
            // Assign to vector worker (IDs 0 to num_vector_workers-1)
            int worker_id = vector_worker_idx % num_vector_workers;
            vector_worker_idx++;
            pto_execute_task_with_worker(rt, task_id, worker_id);
            pto_task_complete(rt, task_id);
            tasks_executed++;
            vector_tasks++;
            did_work = true;
            // Track last vector task
            if (tasks_executed >= 446190) {
                printf("Task #%d: vector task %d (%s)\n", tasks_executed, task_id,
                       rt->pend_task[task_id].func_name ? rt->pend_task[task_id].func_name : "?");
            }
        }
        
        // Try to get a cube task (is_cube=1)
        task_id = pto_get_ready_task_cube(rt);
        if (task_id >= 0) {
            // Assign to cube worker (IDs num_vector_workers to num_workers-1)
            int worker_id = num_vector_workers + (cube_worker_idx % num_cube_workers);
            cube_worker_idx++;
            pto_execute_task_with_worker(rt, task_id, worker_id);
            pto_task_complete(rt, task_id);
            tasks_executed++;
            cube_tasks++;
            did_work = true;
            // Track last cube task  
            if (tasks_executed >= 446190) {
                printf("Task #%d: cube task %d\n", tasks_executed, task_id);
            }
        }
        
        // If no work was done on either queue, we're done
        if (!did_work) {
            printf("\nLoop exit: vector_ready=%d, cube_ready=%d\n",
                   rt->vector_ready_count, rt->cube_ready_count);
            printf("  total_scheduled=%lld, total_completed=%lld\n",
                   (long long)rt->total_tasks_scheduled, (long long)rt->total_tasks_completed);
            
            // Check if there are pending tasks
            int pending_with_fanin = 0;
            int pending_fanin_1 = 0;
            for (int i = 0; i < rt->next_task_id; i++) {
                PendingTask* task = &rt->pend_task[i];
                if (task->is_active && !task->is_complete && task->fanin > 0) {
                    pending_with_fanin++;
                    if (task->fanin == 1) pending_fanin_1++;
                    if (pending_with_fanin <= 3) {
                        printf("  Pending task %d (%s): fanin=%d, is_cube=%d\n",
                               i, task->func_name ? task->func_name : "?", 
                               task->fanin, task->is_cube);
                    }
                }
            }
            printf("  Tasks with unresolved fanin: %d (fanin=1: %d)\n", 
                   pending_with_fanin, pending_fanin_1);
            break;
        }
    }
    
    printf("Simulation complete: %d tasks executed\n", tasks_executed);
    printf("  - Vector tasks (is_cube=0): %d\n", vector_tasks);
    printf("  - Cube tasks (is_cube=1): %d\n", cube_tasks);
    printf("  - Peak vector queue depth: %d\n", peak_vector_queue);
    printf("  - Peak cube queue depth: %d\n", peak_cube_queue);
    
    // Debug: print remaining tasks
    int remaining = rt->total_tasks_scheduled - rt->total_tasks_completed;
    if (remaining > 0) {
        printf("  - Remaining tasks: %d (dependency issue?)\n", remaining);
    }
    
    // Print summary
    pto_trace_print_summary();
    
    // Write trace
    char trace_filename[256];
    snprintf(trace_filename, sizeof(trace_filename), "llama_trace_seq%d.json", seq_len);
    pto_trace_write_json(trace_filename);
    
    // Cleanup
    pto_trace_cleanup();
    pto_runtime_shutdown(rt);
    free(rt);
    
    // Free buffers
    free(input); free(output); free(attn_norm);
    free(wq); free(wk); free(wv); free(wo);
    free(cos_cache); free(sin_cache);
    free(mlp_norm); free(w_gate); free(w_up); free(w_down);
    free(q_tiles); free(k_tiles); free(v_tiles);
    free(q_rope); free(k_rope); free(attn_out);
    free(m_vec); free(l_vec); free(hidden);
    free(temp_norm_buf); free(temp_scores); free(temp_weights);
    free(temp_scale); free(temp_gate); free(temp_up); free(temp_swiglu); free(temp_mlp);
    free(zeros_large); free(zeros_small); free(neg_inf);
    
    printf("\n=== Done ===\n");
    return 0;
}
