
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <signal.h>
#include <setjmp.h>
#include "pto_runtime.h"
#include "pto_runtime_common.h"

// Signal handler for debugging crashes
static jmp_buf crash_jmp;
static void crash_handler(int sig) {
    fprintf(stderr, "ERROR: Signal %d received (likely crash)\n", sig);
    longjmp(crash_jmp, 1);
}

// Forward declarations of InCore functions
void qk_matmul(float* input_q, float* input_k, float* output_s);
void scale_scores(float* input_s, float* output, float scale);
void rowmax(float* input, float* output);
void rowexpandsub(float* input_a, float* input_b, float* output);
void elem_exp(float* input, float* output);
void rowsum(float* input, float* output);
void maximum(float* input_a, float* input_b, float* output);
void sub(float* input_a, float* input_b, float* output);
void mul(float* input_a, float* input_b, float* output);
void add(float* input_a, float* input_b, float* output);
void pv_matmul(float* input_p, float* input_v, float* output);
void rowexpandmul(float* input_a, float* input_b, float* output);
void rowexpanddiv(float* input_a, float* input_b, float* output);

// Wrapper functions to convert PTOInCoreFunc signature

void qk_matmul_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        float* q = (float*)args[0];
        float* k = (float*)args[1];
        float* s = (float*)args[2];
        if (!q || !k || !s) {
            fprintf(stderr, "ERROR: qk_matmul received NULL pointer\n");
            return;
        }
        qk_matmul(q, k, s);
    }
}

void scale_scores_wrapper(void** args, int32_t num_args) {
    if (num_args >= 2) {
        float scale = 0.0883883476f;  // SOFTMAX_SCALE = 1/sqrt(128)
        scale_scores((float*)args[0], (float*)args[1], scale);
    }
}

void rowmax_wrapper(void** args, int32_t num_args) {
    if (num_args >= 2) {
        rowmax((float*)args[0], (float*)args[1]);
    }
}

void rowexpandsub_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        rowexpandsub((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void elem_exp_wrapper(void** args, int32_t num_args) {
    if (num_args >= 2) {
        elem_exp((float*)args[0], (float*)args[1]);
    }
}

void rowsum_wrapper(void** args, int32_t num_args) {
    if (num_args >= 2) {
        rowsum((float*)args[0], (float*)args[1]);
    }
}

void maximum_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        maximum((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void sub_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        sub((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void mul_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        mul((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void add_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        add((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void pv_matmul_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        pv_matmul((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void rowexpandmul_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        rowexpandmul((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

void rowexpanddiv_wrapper(void** args, int32_t num_args) {
    if (num_args >= 3) {
        rowexpanddiv((float*)args[0], (float*)args[1], (float*)args[2]);
    }
}

// Forward declaration of orchestration function
void attention_fusion_block(PTORuntime* rt, 
    float* input_q, float* input_k, float* input_v, 
    float* output_o, float* state_o, float* state_l, float* state_m,
    float* temp_s, float* temp_s_scaled, float* temp_m_new, float* temp_m_local,
    float* temp_s_shifted, float* temp_p, float* temp_l_local, float* temp_m_diff,
    float* temp_scale, float* temp_l_scaled, float* temp_o_scaled, float* temp_o_local);

int main(int argc, char** argv) {
    // Set up signal handlers for debugging
    signal(SIGABRT, crash_handler);
    signal(SIGSEGV, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp) != 0) {
        fprintf(stderr, "Program crashed. Attempting cleanup...\n");
        return 1;
    }
    
    if (argc < 5) {
        fprintf(stderr, "Usage: test_wrapper <q_file> <k_file> <v_file> <output_file>\n");
        return 1;
    }
    
    const char* q_file = argv[1];
    const char* k_file = argv[2];
    const char* v_file = argv[3];
    const char* output_file = argv[4];
    
    // Dimensions
    int q_rows = 8;
    int q_cols = 128;
    int kv_rows = 128;
    int kv_cols = 128;
    
    // Initialize runtime
    PTORuntime* rt = (PTORuntime*)calloc(1, sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "Failed to allocate PTORuntime\n");
        return 1;
    }
    pto_runtime_init(rt);
    
    // Allocate buffers
    float* input_q = (float*)malloc(q_rows * q_cols * sizeof(float));
    float* input_k = (float*)malloc(kv_rows * kv_cols * sizeof(float));
    float* input_v = (float*)malloc(kv_rows * kv_cols * sizeof(float));
    float* output_o = (float*)calloc(q_rows * q_cols, sizeof(float));
    float* state_o = (float*)calloc(q_rows * q_cols, sizeof(float));
    float* state_l = (float*)calloc(q_rows * 1, sizeof(float));
    // state_m: allocated as [q_rows, 128] to match codegen expectation (stride 128)
    // Only the first column is used, but we need stride 128 for codegen compatibility
    float* state_m = (float*)calloc(q_rows * 128, sizeof(float));
    
    // Temporary buffers - initialize to zero to avoid garbage values
    float* temp_s = (float*)calloc(q_rows * kv_rows, sizeof(float));
    float* temp_s_scaled = (float*)calloc(q_rows * kv_rows, sizeof(float));
    float* temp_m_new = (float*)calloc(q_rows * 1, sizeof(float));
    float* temp_m_local = (float*)calloc(q_rows * 1, sizeof(float));
    float* temp_s_shifted = (float*)calloc(q_rows * kv_rows, sizeof(float));
    float* temp_p = (float*)calloc(q_rows * kv_rows, sizeof(float));
    float* temp_l_local = (float*)calloc(q_rows * 1, sizeof(float));
    float* temp_m_diff = (float*)calloc(q_rows * 1, sizeof(float));
    float* temp_scale = (float*)calloc(q_rows * 1, sizeof(float));
    float* temp_l_scaled = (float*)calloc(q_rows * 1, sizeof(float));
    float* temp_o_scaled = (float*)calloc(q_rows * q_cols, sizeof(float));
    float* temp_o_local = (float*)calloc(q_rows * q_cols, sizeof(float));
    
    // Load input data
    FILE* fp = fopen(q_file, "rb");
    if (!fp || fread(input_q, sizeof(float), q_rows * q_cols, fp) != q_rows * q_cols) {
        fprintf(stderr, "Failed to read Q file\n");
        return 1;
    }
    fclose(fp);
    
    fp = fopen(k_file, "rb");
    if (!fp || fread(input_k, sizeof(float), kv_rows * kv_cols, fp) != kv_rows * kv_cols) {
        fprintf(stderr, "Failed to read K file\n");
        return 1;
    }
    fclose(fp);
    
    fp = fopen(v_file, "rb");
    if (!fp || fread(input_v, sizeof(float), kv_rows * kv_cols, fp) != kv_rows * kv_cols) {
        fprintf(stderr, "Failed to read V file\n");
        return 1;
    }
    fclose(fp);
    
    // Initialize state (for first block)
    // state_o: accumulated output, starts at 0
    memset(state_o, 0, q_rows * q_cols * sizeof(float));
    // state_l: sum of exp values, starts at 0
    memset(state_l, 0, q_rows * 1 * sizeof(float));
    // state_m: max values, starts at -inf (for first block, this ensures correct behavior)
    // state_m is allocated as [q_rows, 128] to match codegen stride, but only first column is used
    memset(state_m, 0, q_rows * 128 * sizeof(float));
    for (int i = 0; i < q_rows; i++) {
        state_m[i * 128 + 0] = -INFINITY;
    }
    
    // Call Attention Fusion block
    attention_fusion_block(rt, input_q, input_k, input_v, output_o,
        state_o, state_l, state_m,
        temp_s, temp_s_scaled, temp_m_new, temp_m_local,
        temp_s_shifted, temp_p, temp_l_local, temp_m_diff,
        temp_scale, temp_l_scaled, temp_o_scaled, temp_o_local);
    
    // Set function pointers for all tasks
    for (int32_t task_id = 0; task_id < rt->next_task_id; task_id++) {
        int32_t slot = PTO_TASK_SLOT(task_id);
        PendingTask* task = &rt->pend_task[slot];
        if (!task->is_active) continue;
        
        // Map function name to wrapper function
        if (strcmp(task->func_name, "qk_matmul") == 0) {
            task->func_ptr = (void*)qk_matmul_wrapper;
        } else if (strcmp(task->func_name, "scale_scores") == 0) {
            task->func_ptr = (void*)scale_scores_wrapper;
        } else if (strcmp(task->func_name, "rowmax") == 0) {
            task->func_ptr = (void*)rowmax_wrapper;
        } else if (strcmp(task->func_name, "rowexpandsub") == 0) {
            task->func_ptr = (void*)rowexpandsub_wrapper;
        } else if (strcmp(task->func_name, "elem_exp") == 0) {
            task->func_ptr = (void*)elem_exp_wrapper;
        } else if (strcmp(task->func_name, "rowsum") == 0) {
            task->func_ptr = (void*)rowsum_wrapper;
        } else if (strcmp(task->func_name, "maximum") == 0) {
            task->func_ptr = (void*)maximum_wrapper;
        } else if (strcmp(task->func_name, "sub") == 0) {
            task->func_ptr = (void*)sub_wrapper;
        } else if (strcmp(task->func_name, "mul") == 0) {
            task->func_ptr = (void*)mul_wrapper;
        } else if (strcmp(task->func_name, "add") == 0) {
            task->func_ptr = (void*)add_wrapper;
        } else if (strcmp(task->func_name, "pv_matmul") == 0) {
            task->func_ptr = (void*)pv_matmul_wrapper;
        } else if (strcmp(task->func_name, "rowexpandmul") == 0) {
            task->func_ptr = (void*)rowexpandmul_wrapper;
        } else if (strcmp(task->func_name, "rowexpanddiv") == 0) {
            task->func_ptr = (void*)rowexpanddiv_wrapper;
        } else {
            fprintf(stderr, "WARNING: Unknown function name: %s\n", task->func_name);
        }
    }
    
    // Execute all tasks
    printf("Executing tasks...\n");
    fflush(stdout);
    int executed_count = 0;
    int max_iterations = 10000;
    int iteration = 0;
    
    while ((rt->ready_count > 0 || rt->active_task_count > (int32_t)rt->total_tasks_completed) && iteration < max_iterations) {
        iteration++;
        int32_t task_id = pto_get_ready_task(rt);
        
        if (task_id < 0) {
            if (rt->active_task_count > (int32_t)rt->total_tasks_completed) {
                usleep(100);
                continue;
            }
            break;
        }
        
        int32_t slot = PTO_TASK_SLOT(task_id);
        PendingTask* task = &rt->pend_task[slot];
        
        // Build argument array
        void* args[64];
        int arg_idx = 0;
        
        for (int i = 0; i < task->num_args; i++) {
            TaskArg* arg = &task->args[i];
            if (!arg || !arg->region.raw_tensor) {
                fprintf(stderr, "ERROR: Invalid task argument %d for task %d\n", i, task_id);
                return 1;
            }
            float* base_ptr = (float*)arg->region.raw_tensor;
            // Calculate offset: row_offset * stride + col_offset
            // The stride is determined by the actual tensor layout, not the region dimensions
            // For state_m: allocated as [q_rows, 1] but codegen expects stride 128
            // For state_l: allocated as [q_rows, 1], stride is 1
            // For other tensors: use region.cols as stride (which matches the actual tensor layout)
            int64_t stride;
            if (base_ptr == state_m) {
                // state_m is allocated as [q_rows, 1] but codegen generates stride 128
                // We need to map [q_rows, 1] to [q_rows, 128] layout
                // For row i, col 0: actual offset = i * 1 + 0, but codegen expects i * 128 + 0
                // So we need to create a wrapper or adjust the pointer
                // Actually, the generated code uses stride 128, so we need to allocate with stride 128
                // But we allocated as [q_rows, 1], so we need to handle this differently
                // The simplest fix: allocate state_m as [q_rows, 128] and only use first column
                stride = 128;  // Codegen expects this stride
            } else if (base_ptr == state_l) {
                // state_l is allocated as [q_rows, 1]
                stride = 1;
            } else {
                // Use region.cols as stride (matches actual tensor layout)
                stride = arg->region.cols;
            }
            int64_t offset = arg->region.row_offset * stride + arg->region.col_offset;
            void* ptr = (void*)(base_ptr + offset);
            args[arg_idx++] = ptr;
        }
        
        // Execute the task
        if (task->func_ptr) {
            PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
            if (!func) {
                fprintf(stderr, "ERROR: NULL function pointer for task %d: %s\n", 
                        task_id, task->func_name);
                return 1;
            }
            // Check for NULL arguments
            for (int i = 0; i < task->num_args; i++) {
                if (!args[i]) {
                    fprintf(stderr, "ERROR: NULL argument %d for task %d: %s\n", 
                            i, task_id, task->func_name);
                    return 1;
                }
            }
            // printf("    Executing task %d: %s\n", task_id, task->func_name);
            // fflush(stdout);
            func(args, task->num_args);
            executed_count++;
            // printf("    Task %d completed\n", task_id);
            // fflush(stdout);
        } else {
            fprintf(stderr, "ERROR: No function pointer for task %d: %s\n", 
                    task_id, task->func_name);
            // Don't mark as complete if function pointer is missing
            continue;
        }
        
        pto_task_complete(rt, task_id);
    }
    
    printf("Execution complete! Executed %d tasks\n", executed_count);
    fflush(stdout);
    
    // Execute post-task operations from attention_fusion_block
    // The generated code contains TLOAD/TSTORE operations that need to be executed
    printf("\n  Executing post-task operations from attention_fusion_block...\n");
    
    // Use dynamic allocation to avoid stack overflow for large tile arrays
    float (*tile_o_scaled)[128] = (float(*)[128])malloc(8 * 128 * sizeof(float));
    float (*tile_o_local)[128] = (float(*)[128])malloc(8 * 128 * sizeof(float));
    float (*tile_o_sum)[128] = (float(*)[128])malloc(8 * 128 * sizeof(float));
    float (*tile_m_copy)[1] = (float(*)[1])malloc(8 * 1 * sizeof(float));
    if (!tile_o_scaled || !tile_o_local || !tile_o_sum || !tile_m_copy) {
        fprintf(stderr, "ERROR: Failed to allocate tile arrays\n");
        return 1;
    }
    
    // Debug: Check temp buffer values before post-task operations (commented out for performance)
    // float temp_s_sum = 0.0f;
    // float temp_s_scaled_sum = 0.0f;
    // float temp_m_local_sum = 0.0f;
    // float temp_m_new_sum = 0.0f;
    // float temp_s_shifted_sum = 0.0f;
    // float temp_p_sum = 0.0f;
    // float temp_o_scaled_sum = 0.0f;
    // float temp_o_local_sum = 0.0f;
    // for (int i = 0; i < q_rows * kv_rows; i++) {
    //     temp_s_sum += temp_s[i];
    //     temp_s_scaled_sum += temp_s_scaled[i];
    //     temp_s_shifted_sum += temp_s_shifted[i];
    //     temp_p_sum += temp_p[i];
    // }
    // for (int i = 0; i < q_rows; i++) {
    //     temp_m_local_sum += temp_m_local[i];
    //     temp_m_new_sum += temp_m_new[i];
    // }
    // for (int i = 0; i < q_rows * q_cols; i++) {
    //     temp_o_scaled_sum += temp_o_scaled[i];
    //     temp_o_local_sum += temp_o_local[i];
    // }
    // printf("    temp_s sum: %f\n", temp_s_sum);
    // printf("    temp_s_scaled sum: %f\n", temp_s_scaled_sum);
    // printf("    temp_m_local sum: %f\n", temp_m_local_sum);
    // printf("    temp_m_new sum: %f\n", temp_m_new_sum);
    // printf("    temp_s_shifted sum: %f\n", temp_s_shifted_sum);
    // printf("    temp_p sum: %f\n", temp_p_sum);
    // printf("    temp_o_scaled sum: %f\n", temp_o_scaled_sum);
    // printf("    temp_o_local sum: %f\n", temp_o_local_sum);
    // fflush(stdout);
    
    // TLOAD: tile_o_scaled = load(temp_o_scaled[0, 0])
    for (int _row = 0; _row < q_rows; _row++) {
        for (int _col = 0; _col < q_cols; _col++) {
            tile_o_scaled[_row][_col] = temp_o_scaled[_row * q_cols + _col];
        }
    }
    
    // TLOAD: tile_o_local = load(temp_o_local[0, 0])
    for (int _row = 0; _row < q_rows; _row++) {
        for (int _col = 0; _col < q_cols; _col++) {
            tile_o_local[_row][_col] = temp_o_local[_row * q_cols + _col];
        }
    }
    
    // Fused loop: tile_o_sum = tile_o_scaled + tile_o_local
    // This implements O_new = scale * O + O_local
    for (int _row = 0; _row < q_rows; _row++) {
        for (int _col = 0; _col < q_cols; _col++) {
            tile_o_sum[_row][_col] = tile_o_scaled[_row][_col] + tile_o_local[_row][_col];
        }
    }
    
    // TSTORE: store(tile_o_sum) -> state_o[0, 0]
    for (int _row = 0; _row < q_rows; _row++) {
        for (int _col = 0; _col < q_cols; _col++) {
            state_o[_row * q_cols + _col] = tile_o_sum[_row][_col];
        }
    }
    
    // TLOAD: tile_m_copy = load(temp_m_new[0, 0])
    for (int _row = 0; _row < q_rows; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            tile_m_copy[_row][_col] = temp_m_new[_row * 1 + _col];
        }
    }
    
    // TSTORE: store(tile_m_copy) -> state_m[0, 0]
    // Note: state_m is declared as [q_rows, 128] in attention_fusion_block.c
    // but only the first column is used, so we use stride 128
    for (int _row = 0; _row < q_rows; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            state_m[_row * 128 + _col] = tile_m_copy[_row][_col];
        }
    }
    
    // Free dynamically allocated tile arrays
    free(tile_o_scaled);
    free(tile_o_local);
    free(tile_o_sum);
    free(tile_m_copy);
    
    printf("  Post-task operations completed.\n");
    fflush(stdout);
    
    // Debug: Check state values after post-task operations (commented out for performance)
    // printf("  Debug: Checking state values after post-task operations...\n");
    // float state_o_sum = 0.0f;
    // float state_l_sum = 0.0f;
    // float state_m_sum = 0.0f;
    // for (int i = 0; i < q_rows; i++) {
    //     for (int j = 0; j < q_cols; j++) {
    //         state_o_sum += state_o[i * q_cols + j];
    //     }
    //     state_l_sum += state_l[i * 128 + 0];
    //     state_m_sum += state_m[i * 128 + 0];
    // }
    // printf("    state_o sum: %f\n", state_o_sum);
    // printf("    state_l sum: %f\n", state_l_sum);
    // printf("    state_m sum: %f\n", state_m_sum);
    // fflush(stdout);
    
    // Execute Task 13: rowexpanddiv (final normalization)
    // This task should execute after post-task operations
    // Note: rowexpanddiv uses state_o and state_l which were just updated
    int32_t t13 = -1;
    for (int32_t task_id = 0; task_id < rt->next_task_id; task_id++) {
        int32_t slot = PTO_TASK_SLOT(task_id);
        PendingTask* task = &rt->pend_task[slot];
        if (task->is_active && strcmp(task->func_name, "rowexpanddiv") == 0) {
            t13 = task_id;
            break;
        }
    }
    
    if (t13 >= 0) {
        // Set function pointer
        int32_t slot = PTO_TASK_SLOT(t13);
        PendingTask* task = &rt->pend_task[slot];
        task->func_ptr = (void*)rowexpanddiv_wrapper;
        
        // Build argument array
        void* args[64];
        int arg_idx = 0;
        for (int i = 0; i < task->num_args; i++) {
            TaskArg* arg = &task->args[i];
            if (!arg || !arg->region.raw_tensor) {
                fprintf(stderr, "ERROR: Invalid task argument %d for rowexpanddiv task\n", i);
                return 1;
            }
            float* base_ptr = (float*)arg->region.raw_tensor;
            int64_t stride;
            if (base_ptr == state_l) {
                stride = 1;  // state_l is [q_rows, 1]
            } else {
                stride = arg->region.cols;
            }
            int64_t offset = arg->region.row_offset * stride + arg->region.col_offset;
            void* ptr = (void*)(base_ptr + offset);
            args[arg_idx++] = ptr;
        }
        
        // Execute rowexpanddiv
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
        pto_task_complete(rt, t13);
        printf("  Executed rowexpanddiv task\n");
        fflush(stdout);
    } else {
        // Fallback: manual normalization
        printf("  WARNING: rowexpanddiv task not found, using manual normalization\n");
        fflush(stdout);
        for (int i = 0; i < q_rows; i++) {
            float l_val = state_l[i * 1 + 0];  // state_l is [q_rows, 1], stride is 1
            if (l_val <= 0) {
                printf("    WARNING: state_l[%d] = %f (should be > 0), using 1e-8\n", i, l_val);
                fflush(stdout);
                l_val = 1e-8f;
            }
            for (int j = 0; j < q_cols; j++) {
                float o_val = state_o[i * q_cols + j];
                if (!isfinite(o_val) || !isfinite(l_val)) {
                    fprintf(stderr, "ERROR: Invalid value at [%d][%d]: o_val=%f, l_val=%f\n", 
                            i, j, o_val, l_val);
                    return 1;
                }
                output_o[i * q_cols + j] = o_val / l_val;
            }
        }
        printf("  Normalization completed.\n");
        fflush(stdout);
    }
    
    // Save output
    fp = fopen(output_file, "wb");
    if (!fp || fwrite(output_o, sizeof(float), q_rows * q_cols, fp) != q_rows * q_cols) {
        fprintf(stderr, "Failed to write output file\n");
        return 1;
    }
    fclose(fp);
    
    // Cleanup
    pto_runtime_shutdown(rt);
    free(rt);
    free(input_q);
    free(input_k);
    free(input_v);
    free(output_o);
    free(state_o);
    free(state_l);
    free(state_m);  // state_m is allocated as q_rows * 128
    free(temp_s);
    free(temp_s_scaled);
    free(temp_m_new);
    free(temp_m_local);
    free(temp_s_shifted);
    free(temp_p);
    free(temp_l_local);
    free(temp_m_diff);
    free(temp_scale);
    free(temp_l_scaled);
    free(temp_o_scaled);
    free(temp_o_local);
    
    return 0;
}
