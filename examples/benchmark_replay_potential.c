/**
 * Benchmark: Maximum potential of replay optimization
 * 
 * This test uses a simple loop structure where ALL iterations
 * can be replayed (except first), showing the theoretical maximum benefit.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../pto_runtime.h"
#include "../pto_runtime.c"

#define TILE_ROWS   32
#define HIDDEN_DIM  128
#define TASKS_PER_TILE 19  // Similar to LLaMA layer

static inline double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

void dummy_func(void** args, int32_t n) { (void)args; (void)n; }

typedef struct {
    float* input;
    float* output;
    float* weight;
    float* temps[10];
    int num_tiles;
} TestData;

// Orchestration WITHOUT replay (baseline)
void orchestration_no_replay(PTORuntime* rt, void* user_data) {
    TestData* data = (TestData*)user_data;
    
    for (int tile = 0; tile < data->num_tiles; tile++) {
        int row_off = tile * TILE_ROWS;
        
        // Create chain of TASKS_PER_TILE tasks
        int32_t prev_task = -1;
        for (int t = 0; t < TASKS_PER_TILE; t++) {
            int32_t task = pto_task_alloc(rt, "op", (void*)dummy_func, 0, 0);
            
            // Input from previous output or main input
            if (t == 0) {
                pto_task_add_input(rt, task, data->input, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            } else {
                pto_task_add_input(rt, task, data->temps[t % 10], row_off, 0, TILE_ROWS, HIDDEN_DIM);
            }
            pto_task_add_input(rt, task, data->weight, 0, 0, 1, HIDDEN_DIM);
            
            // Output to temp or final output
            if (t == TASKS_PER_TILE - 1) {
                pto_task_add_output(rt, task, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            } else {
                pto_task_add_output(rt, task, data->temps[(t+1) % 10], row_off, 0, TILE_ROWS, HIDDEN_DIM);
            }
            
            pto_task_submit(rt, task);
            prev_task = task;
        }
    }
}

// Orchestration WITH replay (optimized)
void orchestration_with_replay(PTORuntime* rt, void* user_data) {
    TestData* data = (TestData*)user_data;
    
    LoopReplayCtx ctx = {0};
    pto_loop_init(&ctx, "tile_loop", TILE_ROWS, OFFSET_ROW);
    
    for (int tile = 0; tile < data->num_tiles; tile++) {
        if (pto_loop_should_record(rt, &ctx, tile)) {
            int row_off = tile * TILE_ROWS;
            
            int32_t prev_task = -1;
            for (int t = 0; t < TASKS_PER_TILE; t++) {
                int32_t task = pto_task_alloc(rt, "op", (void*)dummy_func, 0, 0);
                
                if (t == 0) {
                    pto_task_add_input(rt, task, data->input, row_off, 0, TILE_ROWS, HIDDEN_DIM);
                } else {
                    pto_task_add_input(rt, task, data->temps[t % 10], row_off, 0, TILE_ROWS, HIDDEN_DIM);
                }
                pto_task_add_input(rt, task, data->weight, 0, 0, 1, HIDDEN_DIM);
                
                if (t == TASKS_PER_TILE - 1) {
                    pto_task_add_output(rt, task, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
                } else {
                    pto_task_add_output(rt, task, data->temps[(t+1) % 10], row_off, 0, TILE_ROWS, HIDDEN_DIM);
                }
                
                pto_task_submit(rt, task);
                prev_task = task;
            }
            
            pto_loop_finish_record(rt, &ctx);
        } else {
            pto_loop_replay(rt, &ctx, tile);
        }
    }
    
    pto_loop_cleanup(&ctx);
}

typedef struct {
    int task_count;
    int direct_count;
    int replay_count;
    double time_us;
} BenchResult;

BenchResult run_test(void (*orch_func)(PTORuntime*, void*), TestData* data) {
    BenchResult result = {0};
    
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    pto_runtime_init(rt);
    
    double start = get_time_us();
    orch_func(rt, data);
    result.time_us = get_time_us() - start;
    
    result.task_count = rt->next_task_id;
    for (int i = 0; i < rt->next_task_id; i++) {
        if (rt->compact_task[i].template_ref != NULL) {
            result.replay_count++;
        }
    }
    result.direct_count = result.task_count - result.replay_count;
    
    pto_runtime_shutdown(rt);
    free(rt);
    
    return result;
}

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("Replay Optimization: Maximum Potential Benchmark\n");
    printf("================================================================================\n\n");
    
    printf("Configuration:\n");
    printf("  Tasks per tile: %d\n", TASKS_PER_TILE);
    printf("  Tile size: %d x %d\n", TILE_ROWS, HIDDEN_DIM);
    printf("  PendingTask: %zu bytes, CompactTask: %zu bytes (%.0fx smaller)\n\n",
           sizeof(PendingTask), sizeof(CompactTask), 
           (double)sizeof(PendingTask) / sizeof(CompactTask));
    
    // Allocate test data
    int max_tiles = 16384 / 32;  // seq_len=16384
    int max_elements = max_tiles * TILE_ROWS * HIDDEN_DIM;
    
    TestData data;
    data.input = (float*)calloc(max_elements, sizeof(float));
    data.output = (float*)calloc(max_elements, sizeof(float));
    data.weight = (float*)calloc(HIDDEN_DIM, sizeof(float));
    for (int i = 0; i < 10; i++) {
        data.temps[i] = (float*)calloc(max_elements, sizeof(float));
    }
    
    printf("%-10s | %-35s | %-35s | %-10s\n",
           "Tiles", "NO REPLAY", "WITH REPLAY", "Speedup");
    printf("%-10s | %-10s %-10s %-12s | %-10s %-10s %-12s | %-10s\n",
           "", "Tasks", "Time(us)", "Tasks/ms", "Tasks", "Time(us)", "Tasks/ms", "");
    printf("--------------------------------------------------------------------------------------------\n");
    
    int tile_counts[] = {8, 16, 32, 64, 128, 256, 512};
    int num_tests = sizeof(tile_counts) / sizeof(tile_counts[0]);
    
    for (int i = 0; i < num_tests; i++) {
        data.num_tiles = tile_counts[i];
        
        // Warmup
        run_test(orchestration_no_replay, &data);
        run_test(orchestration_with_replay, &data);
        
        // Measure
        int runs = 5;
        double no_replay_total = 0, with_replay_total = 0;
        BenchResult no_replay_last, with_replay_last;
        
        for (int r = 0; r < runs; r++) {
            no_replay_last = run_test(orchestration_no_replay, &data);
            no_replay_total += no_replay_last.time_us;
            
            with_replay_last = run_test(orchestration_with_replay, &data);
            with_replay_total += with_replay_last.time_us;
        }
        
        double no_replay_avg = no_replay_total / runs;
        double with_replay_avg = with_replay_total / runs;
        
        double no_replay_thr = no_replay_last.task_count / (no_replay_avg / 1000.0);
        double with_replay_thr = with_replay_last.task_count / (with_replay_avg / 1000.0);
        double speedup = no_replay_avg / with_replay_avg;
        
        printf("%-10d | %-10d %-10.0f %-12.0f | %-10d %-10.0f %-12.0f | %-10.2fx\n",
               tile_counts[i],
               no_replay_last.task_count, no_replay_avg, no_replay_thr,
               with_replay_last.task_count, with_replay_avg, with_replay_thr,
               speedup);
    }
    
    printf("--------------------------------------------------------------------------------------------\n");
    
    // Detailed analysis
    data.num_tiles = 512;
    BenchResult with_replay = run_test(orchestration_with_replay, &data);
    
    printf("\nDetailed Analysis at %d tiles (%d tasks):\n", 512, with_replay.task_count);
    printf("  Direct tasks:  %d (%.1f%%)\n", with_replay.direct_count, 
           100.0 * with_replay.direct_count / with_replay.task_count);
    printf("  Replay tasks:  %d (%.1f%%)\n", with_replay.replay_count,
           100.0 * with_replay.replay_count / with_replay.task_count);
    
    int64_t direct_bytes = (int64_t)with_replay.direct_count * sizeof(PendingTask);
    int64_t replay_bytes = (int64_t)with_replay.replay_count * sizeof(CompactTask);
    int64_t total_bytes = direct_bytes + replay_bytes;
    int64_t baseline_bytes = (int64_t)with_replay.task_count * sizeof(PendingTask);
    
    printf("  Memory touched: %.1f MB (vs %.1f MB baseline)\n",
           total_bytes / (1024.0 * 1024), baseline_bytes / (1024.0 * 1024));
    printf("  Memory reduction: %.1fx\n", (double)baseline_bytes / total_bytes);
    
    printf("================================================================================\n");
    
    // Cleanup
    free(data.input);
    free(data.output);
    free(data.weight);
    for (int i = 0; i < 10; i++) free(data.temps[i]);
    
    return 0;
}
