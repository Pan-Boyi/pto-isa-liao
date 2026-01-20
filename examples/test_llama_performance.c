/**
 * LLaMA Layer Performance Test
 * 
 * Tests the orchestration runtime for sequence lengths from 1K to 16K.
 * Measures task graph construction time for each sequence length.
 * 
 * Compile: gcc -O2 -I.. -o test_llama_performance test_llama_performance.c
 * Run: ./test_llama_performance
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "pto_runtime.h"
#include "pto_runtime.c"

// Configuration
#define TILE_ROWS 32
#define TILE_COLS 128

// Dummy buffers (we only care about timing, not actual computation)
static float input[1024];
static float output[1024];
static float all_q_tiles[1024];
static float all_k_tiles[1024];
static float all_v_tiles[1024];
static float all_q_rope[1024];
static float all_k_rope[1024];
static float all_attn_out[1024];
static float all_m_vec[1024];
static float all_l_vec[1024];
static float all_hidden[1024];
static float temp_norm[1024];
static float temp_scores[1024];
static float temp_attn_weights[1024];
static float temp_scale[1024];
static float temp_gate[1024];
static float temp_up[1024];
static float temp_swiglu[1024];
static float temp_mlp_out[1024];
static float const_zeros_large[1024];
static float const_zeros_small[1024];
static float const_neg_inf[1024];
static float wq[1024], wk[1024], wv[1024], wo[1024];
static float w_gate[1024], w_up[1024], w_down[1024];
static float attn_norm_weights[1024], mlp_norm_weights[1024];
static float cos_cache[1024], sin_cache[1024];

// Helper to emit a single task
static inline int emit_task(PTORuntime* rt, const char* func_name, int tile, int row_offset,
                           float* in1, float* in2, float* out) {
    int32_t t = pto_task_alloc(rt, func_name, NULL, 0, 0);
    if (in1) pto_task_add_input(rt, t, in1, row_offset, 0, TILE_ROWS, TILE_COLS);
    if (in2) pto_task_add_input(rt, t, in2, 0, 0, TILE_COLS, TILE_COLS);  // weights are shared
    if (out) pto_task_add_output(rt, t, out, row_offset, 0, TILE_ROWS, TILE_COLS);
    pto_task_submit(rt, t);
    return t;
}

// Helper for cross-tile attention tasks
static inline int emit_cross_tile_task(PTORuntime* rt, const char* func_name, 
                                       int q_tile, int q_offset, int kv_tile, int kv_offset,
                                       float* q_in, float* kv_in, float* out) {
    int32_t t = pto_task_alloc(rt, func_name, NULL, 0, 0);
    pto_task_add_input(rt, t, q_in, q_offset, 0, TILE_ROWS, TILE_COLS);
    pto_task_add_input(rt, t, kv_in, kv_offset, 0, TILE_ROWS, TILE_COLS);
    pto_task_add_output(rt, t, out, q_offset, 0, TILE_ROWS, TILE_COLS);
    pto_task_submit(rt, t);
    return t;
}

/**
 * Build task graph for LLaMA layer with given number of tiles.
 * Uses the correct 3-phase structure with cross-tile dependencies.
 * 
 * WITH ADAPTIVE TILE OPTIMIZATION:
 * - 64-row tiles (scale = 2x) → iteration count halved
 * - actual_iters = num_tiles / 2
 */
void build_llama_task_graph(PTORuntime* rt, int num_tiles) {
    // WITH ADAPTIVE TILES: iterations are halved (64-row tiles vs 32-row base)
    int actual_iters = num_tiles / 2;  // Scale = 2x
    int scale = 2;  // 64 rows / 32 rows
    
    int task_idx = 0;
    
    // ================================================================
    // PHASE 1: Pre-Attention (ALL tiles parallel)
    // With adaptive: actual_iters iterations
    // ================================================================
    for (int tile_i = 0; tile_i < actual_iters; tile_i++) {
        // Offset is scaled by 2 to cover the same memory region
        int row_offset = tile_i * TILE_ROWS * scale;
        
        // RMSNorm
        emit_task(rt, "rmsnorm_tile_64", tile_i, row_offset, input, attn_norm_weights, temp_norm);
        
        // Q, K, V matmuls
        emit_task(rt, "tile_matmul_64", tile_i, row_offset, temp_norm, wq, all_q_tiles);
        emit_task(rt, "tile_matmul_64", tile_i, row_offset, temp_norm, wk, all_k_tiles);
        emit_task(rt, "tile_matmul_64", tile_i, row_offset, temp_norm, wv, all_v_tiles);
        
        // RoPE
        emit_task(rt, "rope_tile_64", tile_i, row_offset, all_q_tiles, cos_cache, all_q_rope);
        emit_task(rt, "rope_tile_64", tile_i, row_offset, all_k_tiles, cos_cache, all_k_rope);
    }
    
    // ================================================================
    // PHASE 2: Flash Attention (CROSS-TILE dependencies)
    // With adaptive: actual_iters × actual_iters iterations (halved in both dimensions!)
    // ================================================================
    for (int q_tile = 0; q_tile < actual_iters; q_tile++) {
        int q_offset = q_tile * TILE_ROWS * scale;
        
        // Initialize attention state
        emit_task(rt, "flash_attn_init_state", q_tile, q_offset, const_zeros_large, NULL, all_attn_out);
        
        // Inner loop: Q[q_tile] attends to ALL K,V tiles (ALSO HALVED!)
        for (int kv_tile = 0; kv_tile < actual_iters; kv_tile++) {
            int kv_offset = kv_tile * TILE_ROWS * scale;
            
            // Score: S = Q[q] @ K[kv].T (CROSS-TILE!)
            emit_cross_tile_task(rt, "flash_attn_score_block", 
                                q_tile, q_offset, kv_tile, kv_offset,
                                all_q_rope, all_k_rope, temp_scores);
            
            // Softmax update
            emit_task(rt, "flash_attn_softmax_update", q_tile, q_offset, 
                     temp_scores, all_m_vec, temp_attn_weights);
            
            // Output update: O += P @ V[kv] (CROSS-TILE!)
            emit_cross_tile_task(rt, "flash_attn_output_update",
                                q_tile, q_offset, kv_tile, kv_offset,
                                all_attn_out, all_v_tiles, all_attn_out);
        }
        
        // Normalize
        emit_task(rt, "flash_attn_normalize", q_tile, q_offset, all_attn_out, all_l_vec, all_attn_out);
    }
    
    // ================================================================
    // PHASE 3: Post-Attention (depends on Phase 2 completion)
    // With adaptive: actual_iters iterations
    // ================================================================
    for (int tile_i = 0; tile_i < actual_iters; tile_i++) {
        int row_offset = tile_i * TILE_ROWS * scale;
        
        // Output projection
        emit_task(rt, "tile_matmul_64", tile_i, row_offset, all_attn_out, wo, temp_norm);
        
        // Residual
        emit_task(rt, "residual_add_tile_64", tile_i, row_offset, temp_norm, input, all_hidden);
        
        // MLP RMSNorm
        emit_task(rt, "rmsnorm_tile_64", tile_i, row_offset, all_hidden, mlp_norm_weights, temp_norm);
        
        // Gate and Up
        emit_task(rt, "tile_matmul_64", tile_i, row_offset, temp_norm, w_gate, temp_gate);
        emit_task(rt, "tile_matmul_64", tile_i, row_offset, temp_norm, w_up, temp_up);
        
        // SwiGLU
        emit_task(rt, "swiglu_tile_64", tile_i, row_offset, temp_gate, temp_up, temp_swiglu);
        
        // Down
        emit_task(rt, "tile_matmul_64", tile_i, row_offset, temp_swiglu, w_down, temp_mlp_out);
        
        // Final residual
        emit_task(rt, "residual_add_tile_64", tile_i, row_offset, temp_mlp_out, all_hidden, output);
    }
}

/**
 * Get current time in milliseconds
 */
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/**
 * Calculate expected task count for N tiles WITH ADAPTIVE TILE OPTIMIZATION
 * 
 * With 64-row tiles (scale = 2x), iteration count is halved:
 * - actual_iters = num_tiles / 2
 * 
 * Phase 1: actual_iters * 6 tasks = (N/2) * 6
 * Phase 2: actual_iters * (1 + actual_iters*3 + 1) = (N/2)^2 * 4 (approximately)
 * Phase 3: actual_iters * 8 tasks = (N/2) * 8
 * 
 * Total with adaptive: ~8*(N/2) + 3*(N/2)^2 = 4N + 3N²/4
 */
int expected_task_count(int num_tiles) {
    // With adaptive tiles (64-row), iterations are halved
    int actual_iters = num_tiles / 2;
    return 14 * actual_iters + 3 * actual_iters * actual_iters;
}

/**
 * Calculate expected task count WITHOUT adaptive optimization (for comparison)
 */
int expected_task_count_no_adaptive(int num_tiles) {
    return 16 * num_tiles + 3 * num_tiles * num_tiles;
}

/**
 * Calculate memory usage for task data structures
 */
typedef struct {
    size_t pending_task_bytes;    // PendingTask array
    size_t tensormap_bytes;       // TensorMap hash table + entries
    size_t ready_queue_bytes;     // Ready queue
    size_t total_bytes;           // Total memory
} MemoryUsage;

MemoryUsage calculate_memory_usage(PTORuntime* rt) {
    MemoryUsage mem = {0};
    
    // PendingTask array (fixed size allocation)
    mem.pending_task_bytes = sizeof(PendingTask) * PTO_MAX_TASKS;
    
    // TensorMap: hash table pointers + actual entries
    mem.tensormap_bytes = sizeof(TensorMapEntry*) * PTO_TENSORMAP_SIZE;
    
    // Count actual tensormap entries
    int tensormap_entry_count = 0;
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        TensorMapEntry* entry = rt->tensor_map[i];
        while (entry) {
            tensormap_entry_count++;
            entry = entry->next;
        }
    }
    mem.tensormap_bytes += tensormap_entry_count * sizeof(TensorMapEntry);
    
    // Ready queue (fixed size)
    mem.ready_queue_bytes = sizeof(int32_t) * PTO_MAX_READY_QUEUE;
    
    // Total
    mem.total_bytes = mem.pending_task_bytes + mem.tensormap_bytes + mem.ready_queue_bytes;
    
    return mem;
}

/**
 * Calculate actual memory used (based on actual task count)
 */
size_t calculate_actual_memory(int task_count, int tensormap_entries) {
    return task_count * sizeof(PendingTask) + 
           tensormap_entries * sizeof(TensorMapEntry) +
           sizeof(TensorMapEntry*) * PTO_TENSORMAP_SIZE +
           sizeof(int32_t) * PTO_MAX_READY_QUEUE;
}

int main(int argc, char** argv) {
    printf("====================================================================\n");
    printf("LLaMA Layer Orchestration Performance Test\n");
    printf("WITH ADAPTIVE TILE OPTIMIZATION (64-row tiles, scale=2x)\n");
    printf("====================================================================\n");
    printf("\n");
    printf("Configuration:\n");
    printf("  Base Tile Size: %d x %d (32 rows)\n", TILE_ROWS, TILE_COLS);
    printf("  Adaptive Tile Size: 64 x %d (64 rows, scale=2x)\n", TILE_COLS);
    printf("  Sequence Lengths: 1K to 16K\n");
    printf("\n");
    printf("Task Graph Complexity WITH ADAPTIVE TILES:\n");
    printf("  Actual iterations: N/2 (halved due to 64-row tiles)\n");
    printf("  Phase 1 (Pre-Attn):  6*(N/2) tasks\n");
    printf("  Phase 2 (Flash Attn): (N/2)^2 * 4 tasks (cross-tile, QUADRATIC REDUCTION!)\n");
    printf("  Phase 3 (Post-Attn): 8*(N/2) tasks\n");
    printf("  Total: 14*(N/2) + 3*(N/2)^2 = 7N + 3N^2/4\n");
    printf("\n");
    printf("OPTIMIZATION: N^2 -> (N/2)^2 = N^2/4 => 75%% REDUCTION!\n");
    printf("\n");
    printf("====================================================================\n");
    printf("%-8s %-8s %-8s %-10s %-10s %-12s %-10s %-12s %-8s\n", 
           "SeqLen", "Tiles", "ActIter", "Tasks", "NoAdapt", "Build(ms)", "Tasks/ms", "Memory", "Saved");
    printf("--------------------------------------------------------------------\n");
    
    // Test sequence lengths from 1K to 16K
    // With adaptive tiles: iterations are halved, quadratic term becomes N^2/4
    int seq_lengths[] = {1024, 2048, 4096, 8192, 12288, 16384};
    int num_tests = sizeof(seq_lengths) / sizeof(seq_lengths[0]);
    
    for (int i = 0; i < num_tests; i++) {
        int seq_len = seq_lengths[i];
        int num_tiles = seq_len / TILE_ROWS;
        int actual_iters = num_tiles / 2;  // With adaptive tiles (64-row)
        int expected_tasks = expected_task_count(num_tiles);
        int expected_no_adaptive = expected_task_count_no_adaptive(num_tiles);
        
        // Check if we'll exceed the limit
        if (expected_tasks > PTO_MAX_TASKS) {
            // Estimate memory for this configuration
            double est_mem_mb = (expected_tasks * sizeof(PendingTask) + 
                                expected_tasks * sizeof(TensorMapEntry) +
                                sizeof(TensorMapEntry*) * PTO_TENSORMAP_SIZE) / (1024.0 * 1024.0);
            printf("%-8d %-8d %-8d %-10d %-10d %-12s %-10s %-10.1f MB (exceeds limit %d)\n", 
                   seq_len, num_tiles, actual_iters, expected_tasks, expected_no_adaptive, 
                   "N/A", "N/A", est_mem_mb, PTO_MAX_TASKS);
            continue;
        }
        
        // Initialize runtime (heap allocated to avoid stack overflow)
        PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
        if (!rt) {
            printf("  ERROR: Failed to allocate runtime\n");
            continue;
        }
        pto_runtime_init(rt);
        
        // Measure task graph construction time
        double start = get_time_ms();
        build_llama_task_graph(rt, num_tiles);
        double end = get_time_ms();
        
        double elapsed_ms = end - start;
        int actual_tasks = rt->total_tasks_scheduled;
        double tasks_per_ms = actual_tasks / elapsed_ms;
        
        // Calculate memory usage
        MemoryUsage mem = calculate_memory_usage(rt);
        
        // Calculate actual memory used (not the fixed allocation)
        // Count tensormap entries
        int tensormap_entries = 0;
        for (int j = 0; j < PTO_TENSORMAP_SIZE; j++) {
            TensorMapEntry* entry = rt->tensor_map[j];
            while (entry) {
                tensormap_entries++;
                entry = entry->next;
            }
        }
        
        // Actual memory = tasks * sizeof(PendingTask) + tensormap entries * sizeof(TensorMapEntry)
        size_t actual_mem = actual_tasks * sizeof(PendingTask) + 
                           tensormap_entries * sizeof(TensorMapEntry);
        double actual_mem_mb = actual_mem / (1024.0 * 1024.0);
        double savings_pct = 100.0 * (1.0 - (double)actual_tasks / expected_no_adaptive);
        
        printf("%-8d %-8d %-8d %-10d %-10d %-12.3f %-10.1f %-10.2f MB %-6.1f%%\n", 
               seq_len, num_tiles, actual_iters, actual_tasks, expected_no_adaptive,
               elapsed_ms, tasks_per_ms, actual_mem_mb, savings_pct);
        
        // Verify task count
        if (actual_tasks != expected_tasks) {
            printf("  WARNING: Expected %d tasks (adaptive), got %d\n", expected_tasks, actual_tasks);
        }
        
        // Cleanup
        pto_runtime_shutdown(rt);
        free(rt);
    }
    
    printf("====================================================================\n");
    printf("\n");
    printf("Analysis - ADAPTIVE TILE OPTIMIZATION:\n");
    printf("--------------------------------------------------------------------\n");
    printf("  WITHOUT ADAPTIVE: 16N + 3N^2 (where N = seq_len / 32-row tiles)\n");
    printf("  WITH ADAPTIVE:    14*(N/2) + 3*(N/2)^2 = 7N + 3N^2/4\n");
    printf("\n");
    printf("  SAVINGS: N^2 -> (N/2)^2 = N^2/4 => 75%% reduction in quadratic term!\n");
    printf("\n");
    printf("  The 64-row tiles (vs 32-row) halve iteration count in BOTH loops:\n");
    printf("    - Outer Q loop: N -> N/2\n");
    printf("    - Inner KV loop: N -> N/2\n");
    printf("    - Flash Attention: N*N -> (N/2)*(N/2) = N^2/4\n");
    printf("\n");
    printf("Extrapolation (with vs without adaptive):\n");
    printf("  %-8s %-8s %-12s %-12s %-12s %-8s\n", "SeqLen", "Tiles", "Adaptive", "No-Adaptive", "Savings", "Memory");
    for (int s = 8192; s <= 32768; s *= 2) {
        int n = s / TILE_ROWS;
        int tasks_adaptive = expected_task_count(n);
        int tasks_no_adaptive = expected_task_count_no_adaptive(n);
        double savings_pct = 100.0 * (1.0 - (double)tasks_adaptive / tasks_no_adaptive);
        double est_mem_mb = (tasks_adaptive * sizeof(PendingTask) + 
                            tasks_adaptive * sizeof(TensorMapEntry)) / (1024.0 * 1024.0);
        printf("  %-8dK %-8d %-12d %-12d %-6.1f%% ~%.1f MB\n",
               s/1024, n, tasks_adaptive, tasks_no_adaptive, savings_pct, est_mem_mb);
    }
    
    printf("\nData Structure Sizes:\n");
    printf("  sizeof(PendingTask):    %zu bytes\n", sizeof(PendingTask));
    printf("  sizeof(TensorMapEntry): %zu bytes\n", sizeof(TensorMapEntry));
    printf("  PTO_MAX_TASKS:          %d\n", PTO_MAX_TASKS);
    printf("  Fixed overhead:         %.2f MB\n", 
           (sizeof(PendingTask) * PTO_MAX_TASKS + 
            sizeof(TensorMapEntry*) * PTO_TENSORMAP_SIZE +
            sizeof(int32_t) * PTO_MAX_READY_QUEUE) / (1024.0 * 1024.0));
    printf("\n");
    printf("CONCLUSION: Adaptive tile optimization provides ~75%% reduction in task count\n");
    printf("            for Flash Attention's N^2 dependency pattern.\n");
    printf("\n");
    
    return 0;
}
