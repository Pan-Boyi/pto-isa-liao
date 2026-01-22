/**
 * Benchmark: Compare orchestration with/without record_replay
 * 
 * Tests the actual LLaMA orchestration function with:
 * 1. Replay DISABLED (baseline - all direct task creation)
 * 2. Replay ENABLED (optimized - uses CompactTask for replay)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define PTO_NO_MAIN
#include "output_arm64/llama7b/llama_layer_dynamic_orchestration.c"

static inline double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

#define MAX_ELEMENTS (16384 * 128)
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

typedef struct {
    int task_count;
    int direct_count;
    int replay_count;
    double time_us;
} BenchResult;

BenchResult run_benchmark(int seq_len, int replay_enabled) {
    BenchResult result = {0};
    int num_tiles = seq_len / 32;
    
    // Set replay mode
    pto_set_record_replay(replay_enabled);
    
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
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
    
    result.time_us = get_time_us() - start;
    result.task_count = rt->next_task_id;
    
    // Count compact (replay) vs direct tasks
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
    printf("LLaMA Orchestration: Replay vs No-Replay Comparison\n");
    printf("================================================================================\n\n");
    
    printf("Memory Analysis:\n");
    printf("  sizeof(PendingTask):  %zu bytes\n", sizeof(PendingTask));
    printf("  sizeof(CompactTask):  %zu bytes\n", sizeof(CompactTask));
    printf("  Ratio: %.0fx smaller for replay tasks\n\n", (double)sizeof(PendingTask) / sizeof(CompactTask));
    
    alloc_tensors();
    
    int seq_lens[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    int num_tests = sizeof(seq_lens) / sizeof(seq_lens[0]);
    
    printf("%-10s | %-40s | %-40s | %-10s\n",
           "SeqLen", "NO REPLAY (baseline)", "WITH REPLAY (optimized)", "Speedup");
    printf("%-10s | %-12s %-12s %-12s | %-12s %-12s %-12s | %-10s\n",
           "", "Tasks", "Time(us)", "Tasks/ms", "Tasks", "Time(us)", "Tasks/ms", "");
    printf("----------------------------------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < num_tests; i++) {
        int seq_len = seq_lens[i];
        
        // Run without replay (baseline)
        BenchResult no_replay = run_benchmark(seq_len, 0);
        double no_replay_thr = no_replay.task_count / (no_replay.time_us / 1000.0);
        
        // Run with replay (optimized)
        BenchResult with_replay = run_benchmark(seq_len, 1);
        double with_replay_thr = with_replay.task_count / (with_replay.time_us / 1000.0);
        
        double speedup = no_replay.time_us / with_replay.time_us;
        
        printf("%-10d | %-12d %-12.0f %-12.0f | %-12d %-12.0f %-12.0f | %-10.2fx\n",
               seq_len, 
               no_replay.task_count, no_replay.time_us, no_replay_thr,
               with_replay.task_count, with_replay.time_us, with_replay_thr,
               speedup);
    }
    
    printf("----------------------------------------------------------------------------------------------------------\n");
    
    // Detailed analysis at seq_len=16384
    printf("\nDetailed Analysis at seq_len=16384:\n");
    printf("-----------------------------------\n");
    
    // Multiple runs for stable measurement
    int runs = 5;
    double no_replay_total = 0, with_replay_total = 0;
    BenchResult no_replay_last, with_replay_last;
    
    for (int r = 0; r < runs; r++) {
        no_replay_last = run_benchmark(16384, 0);
        no_replay_total += no_replay_last.time_us;
        
        with_replay_last = run_benchmark(16384, 1);
        with_replay_total += with_replay_last.time_us;
    }
    
    double no_replay_avg = no_replay_total / runs;
    double with_replay_avg = with_replay_total / runs;
    
    printf("\nNO REPLAY (baseline):\n");
    printf("  Tasks: %d (all direct)\n", no_replay_last.task_count);
    printf("  Time:  %.1f us (avg of %d runs)\n", no_replay_avg, runs);
    printf("  Throughput: %.0f tasks/ms\n", no_replay_last.task_count / (no_replay_avg / 1000.0));
    
    printf("\nWITH REPLAY (optimized):\n");
    printf("  Tasks: %d (direct=%d, replay=%d)\n", 
           with_replay_last.task_count, with_replay_last.direct_count, with_replay_last.replay_count);
    printf("  Time:  %.1f us (avg of %d runs)\n", with_replay_avg, runs);
    printf("  Throughput: %.0f tasks/ms\n", with_replay_last.task_count / (with_replay_avg / 1000.0));
    printf("  Replay ratio: %.1f%%\n", 100.0 * with_replay_last.replay_count / with_replay_last.task_count);
    
    printf("\nSpeedup: %.2fx\n", no_replay_avg / with_replay_avg);
    
    // Cache analysis
    printf("\nCache Analysis:\n");
    int64_t no_replay_bytes = (int64_t)no_replay_last.task_count * sizeof(PendingTask);
    int64_t replay_bytes = (int64_t)with_replay_last.direct_count * sizeof(PendingTask) +
                           (int64_t)with_replay_last.replay_count * sizeof(CompactTask);
    printf("  No replay: %lld MB touched\n", no_replay_bytes / (1024 * 1024));
    printf("  With replay: %lld MB touched\n", replay_bytes / (1024 * 1024));
    printf("  Memory reduction: %.1fx\n", (double)no_replay_bytes / replay_bytes);
    
    printf("================================================================================\n");
    
    free_tensors();
    return 0;
}
