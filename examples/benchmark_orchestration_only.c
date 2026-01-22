/**
 * Benchmark orchestration time ONLY (no task execution)
 * Compares direct task creation vs loop replay
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../pto_runtime.h"
#include "../pto_runtime.c"

#define TILE_ROWS   32
#define HIDDEN_DIM  128

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
    float* temp1;
    float* temp2;
    int num_tiles;
} TestData;

// Direct task creation (no replay)
void orchestration_direct(PTORuntime* rt, void* user_data) {
    TestData* data = (TestData*)user_data;
    
    for (int tile = 0; tile < data->num_tiles; tile++) {
        int row_off = tile * TILE_ROWS;
        
        // Task 0: op1
        int32_t t0 = pto_task_alloc(rt, "op1", (void*)dummy_func, 0, 0);
        pto_task_add_input(rt, t0, data->input, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t0, data->weight, 0, 0, 1, HIDDEN_DIM);
        pto_task_add_output(rt, t0, data->temp1, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t0);
        
        // Task 1: op2
        int32_t t1 = pto_task_alloc(rt, "op2", (void*)dummy_func, 0, 0);
        pto_task_add_input(rt, t1, data->temp1, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t1, data->weight, 0, 0, 1, HIDDEN_DIM);
        pto_task_add_output(rt, t1, data->temp2, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t1);
        
        // Task 2: op3
        int32_t t2 = pto_task_alloc(rt, "op3", (void*)dummy_func, 0, 0);
        pto_task_add_input(rt, t2, data->temp2, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_output(rt, t2, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t2);
        
        // Task 3: op4
        int32_t t3 = pto_task_alloc(rt, "op4", (void*)dummy_func, 0, 0);
        pto_task_add_input(rt, t3, data->input, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_input(rt, t3, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_add_output(rt, t3, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
        pto_task_submit(rt, t3);
    }
}

// Loop replay (compact task array)
void orchestration_replay(PTORuntime* rt, void* user_data) {
    TestData* data = (TestData*)user_data;
    
    LoopReplayCtx ctx = {0};
    pto_loop_init(&ctx, "tile_loop", TILE_ROWS, OFFSET_ROW);
    
    for (int tile = 0; tile < data->num_tiles; tile++) {
        if (pto_loop_should_record(rt, &ctx, tile)) {
            int row_off = tile * TILE_ROWS;
            
            int32_t t0 = pto_task_alloc(rt, "op1", (void*)dummy_func, 0, 0);
            pto_task_add_input(rt, t0, data->input, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t0, data->weight, 0, 0, 1, HIDDEN_DIM);
            pto_task_add_output(rt, t0, data->temp1, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t0);
            
            int32_t t1 = pto_task_alloc(rt, "op2", (void*)dummy_func, 0, 0);
            pto_task_add_input(rt, t1, data->temp1, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t1, data->weight, 0, 0, 1, HIDDEN_DIM);
            pto_task_add_output(rt, t1, data->temp2, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t1);
            
            int32_t t2 = pto_task_alloc(rt, "op3", (void*)dummy_func, 0, 0);
            pto_task_add_input(rt, t2, data->temp2, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_output(rt, t2, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t2);
            
            int32_t t3 = pto_task_alloc(rt, "op4", (void*)dummy_func, 0, 0);
            pto_task_add_input(rt, t3, data->input, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_input(rt, t3, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_add_output(rt, t3, data->output, row_off, 0, TILE_ROWS, HIDDEN_DIM);
            pto_task_submit(rt, t3);
            
            pto_loop_finish_record(rt, &ctx);
        } else {
            pto_loop_replay(rt, &ctx, tile);
        }
    }
    
    pto_loop_cleanup(&ctx);
}

double measure_orchestration(void (*orch_func)(PTORuntime*, void*), TestData* data) {
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    pto_runtime_init(rt);
    
    double start = get_time_us();
    orch_func(rt, data);
    double elapsed = get_time_us() - start;
    
    int32_t total_tasks = rt->next_task_id;
    
    pto_runtime_shutdown(rt);
    free(rt);
    
    return elapsed;
}

int main(int argc, char** argv) {
    int num_tiles = 512;
    if (argc > 1) num_tiles = atoi(argv[1]);
    
    int total_elements = num_tiles * TILE_ROWS * HIDDEN_DIM;
    
    printf("Orchestration-Only Benchmark\n");
    printf("============================\n");
    printf("Tiles: %d, Tasks per tile: 4, Total tasks: %d\n\n", num_tiles, num_tiles * 4);
    
    // Allocate buffers
    float* input = (float*)malloc(total_elements * sizeof(float));
    float* output = (float*)malloc(total_elements * sizeof(float));
    float* weight = (float*)malloc(HIDDEN_DIM * sizeof(float));
    float* temp1 = (float*)malloc(total_elements * sizeof(float));
    float* temp2 = (float*)malloc(total_elements * sizeof(float));
    
    TestData data = {
        .input = input,
        .output = output,
        .weight = weight,
        .temp1 = temp1,
        .temp2 = temp2,
        .num_tiles = num_tiles
    };
    
    // Warmup
    measure_orchestration(orchestration_direct, &data);
    measure_orchestration(orchestration_replay, &data);
    
    // Benchmark
    int runs = 10;
    double direct_sum = 0, replay_sum = 0;
    
    for (int i = 0; i < runs; i++) {
        direct_sum += measure_orchestration(orchestration_direct, &data);
        replay_sum += measure_orchestration(orchestration_replay, &data);
    }
    
    double direct_avg = direct_sum / runs;
    double replay_avg = replay_sum / runs;
    int total_tasks = num_tiles * 4;
    
    printf("Results (average of %d runs):\n", runs);
    printf("  Direct:       %.2f us (%.1f tasks/ms)\n", direct_avg, total_tasks / (direct_avg / 1000.0));
    printf("  Loop Replay:  %.2f us (%.1f tasks/ms)\n", replay_avg, total_tasks / (replay_avg / 1000.0));
    printf("  Speedup:      %.2fx\n", direct_avg / replay_avg);
    
    free(input);
    free(output);
    free(weight);
    free(temp1);
    free(temp2);
    
    return 0;
}
