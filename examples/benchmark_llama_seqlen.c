/**
 * Benchmark: LLaMA Orchestration Time for seq_len 256 to 16K
 * 
 * Uses the actual generated llama_layer_dynamic orchestration function
 * Measures pure orchestration (task graph building) time
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Define PTO_NO_MAIN to avoid duplicate main()
#define PTO_NO_MAIN

// Include the actual generated orchestration
#include "output_arm64/llama7b/llama_layer_dynamic_orchestration.c"

static inline double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// Dummy tensors (only need valid pointers, data doesn't matter for orchestration)
#define MAX_ELEMENTS (16384 * 128)  // 16K seq * 128 cols
static float* tensors[50];

static void alloc_tensors() {
    for (int i = 0; i < 50; i++) {
        tensors[i] = (float*)calloc(MAX_ELEMENTS, sizeof(float));
    }
}

static void free_tensors() {
    for (int i = 0; i < 50; i++) {
        if (tensors[i]) free(tensors[i]);
    }
}

/**
 * Run orchestration and return time in microseconds
 * Also returns task count and compact task count via pointers
 */
double run_orchestration(int seq_len, int* task_count, int* compact_count) {
    int num_tiles = seq_len / 32;
    
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "Failed to allocate runtime\n");
        return -1;
    }
    pto_runtime_init(rt);
    
    double start = get_time_us();
    
    llama_layer_dynamic(rt, 
        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
        tensors[5], tensors[6], tensors[7], tensors[8], tensors[9],
        tensors[10], tensors[11], tensors[12], tensors[13], tensors[14],
        tensors[15], tensors[16], tensors[17], tensors[18], tensors[19],
        tensors[20], tensors[21], tensors[22], tensors[23], tensors[24],
        tensors[25], tensors[26], tensors[27], tensors[28], tensors[29],
        tensors[30], tensors[31], tensors[32],
        seq_len, num_tiles);
    
    double elapsed = get_time_us() - start;
    
    *task_count = rt->next_task_id;
    
    // Count compact (replay) vs direct tasks
    *compact_count = 0;
    for (int i = 0; i < rt->next_task_id; i++) {
        if (rt->compact_task[i].template_ref != NULL) {
            (*compact_count)++;
        }
    }
    
    pto_runtime_shutdown(rt);
    free(rt);
    
    return elapsed;
}

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("LLaMA Orchestration Benchmark (seq_len = 256 to 16K)\n");
    printf("================================================================================\n\n");
    
    printf("Memory Analysis:\n");
    printf("  sizeof(PendingTask):  %zu bytes\n", sizeof(PendingTask));
    printf("  sizeof(CompactTask):  %zu bytes\n", sizeof(CompactTask));
    printf("  Ratio: %.0fx smaller for replay tasks\n\n", (double)sizeof(PendingTask) / sizeof(CompactTask));
    
    alloc_tensors();
    
    // Warmup run
    int dummy_tasks, dummy_compact;
    run_orchestration(256, &dummy_tasks, &dummy_compact);
    
    printf("%-10s %-8s %-10s %-10s %-10s %-12s %-10s\n",
           "SeqLen", "Tiles", "Tasks", "Direct", "Replay", "Time(us)", "Tasks/ms");
    printf("--------------------------------------------------------------------------------\n");
    
    // Test sequence lengths: 256, 512, 1024, 2048, 4096, 8192, 16384
    // NOTE: Static LoopReplayCtx vars persist, so subsequent calls at same or smaller
    // seq_len will reuse fragments. Run in increasing order for consistent results.
    int seq_lens[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    int num_tests = sizeof(seq_lens) / sizeof(seq_lens[0]);
    
    // First pass: measure each seq_len independently
    for (int i = 0; i < num_tests; i++) {
        int seq_len = seq_lens[i];
        int num_tiles = seq_len / 32;
        
        int task_count = 0, compact_count = 0;
        double time_us = run_orchestration(seq_len, &task_count, &compact_count);
        int direct_count = task_count - compact_count;
        
        double tasks_per_ms = task_count / (time_us / 1000.0);
        
        printf("%-10d %-8d %-10d %-10d %-10d %-12.1f %-10.1f\n",
               seq_len, num_tiles, task_count, direct_count, compact_count, time_us, tasks_per_ms);
    }
    
    printf("\n--- Steady-state performance at seq_len=16384 ---\n");
    // After all fragments are recorded, measure steady-state performance
    double total_time = 0;
    int task_count = 0, compact_count = 0;
    int runs = 10;
    for (int r = 0; r < runs; r++) {
        total_time += run_orchestration(16384, &task_count, &compact_count);
    }
    double avg_time = total_time / runs;
    int direct_count = task_count - compact_count;
    double tasks_per_ms = task_count / (avg_time / 1000.0);
    double replay_ratio = 100.0 * compact_count / task_count;
    printf("%-10d %-8d %-10d %-10d %-10d %-12.1f %-10.1f (avg %d runs, %.1f%% replay)\n",
           16384, 512, task_count, direct_count, compact_count, avg_time, tasks_per_ms, runs, replay_ratio);
    
    printf("================================================================================\n");
    printf("\nNOTE: Loop replay uses CompactTask (24 bytes) instead of PendingTask (2864 bytes)\n");
    printf("      This provides ~120x better cache efficiency for replay tasks\n");
    
    free_tensors();
    return 0;
}
