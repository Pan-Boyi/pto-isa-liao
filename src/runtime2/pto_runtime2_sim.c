/**
 * PTO Runtime2 - Simulation Implementation
 * 
 * Implements cycle-accurate simulation with optional A2A3 core model integration.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_runtime2_sim.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Check if core model is available
#ifdef A2A3_CORE_SIM_AVAILABLE
#include "runtime_a2a3_sim/core_model/a2a3_core_model.h"
#endif

// =============================================================================
// Cycle Cost Estimation
// =============================================================================

/**
 * Default cycle costs (when core model not available)
 */
#define DEFAULT_CUBE_CYCLES      100   // Matrix multiply
#define DEFAULT_VECTOR_CYCLES    50    // Element-wise ops
#define DEFAULT_DMA_CYCLES       80    // DMA transfer
#define DEFAULT_AICPU_CYCLES     200   // AI_CPU operations

int64_t pto2_sim_estimate_cycles_by_name(const char* func_name, int64_t data_size) {
    if (!func_name) {
        return DEFAULT_VECTOR_CYCLES;
    }
    
    // Matrix operations (Cube)
    if (strstr(func_name, "gemm") || strstr(func_name, "matmul") ||
        strstr(func_name, "conv")) {
        // Base cost + data-dependent cost
        int64_t base = DEFAULT_CUBE_CYCLES;
        int64_t data_cost = data_size / 1024;  // 1 cycle per KB
        return base + data_cost;
    }
    
    // DMA / Copy operations
    if (strstr(func_name, "dma") || strstr(func_name, "copy") ||
        strstr(func_name, "transfer")) {
        int64_t base = DEFAULT_DMA_CYCLES;
        int64_t data_cost = data_size / 512;  // 2 cycles per KB
        return base + data_cost;
    }
    
    // Vector operations
    if (strstr(func_name, "add") || strstr(func_name, "mul") ||
        strstr(func_name, "relu") || strstr(func_name, "sigmoid") ||
        strstr(func_name, "vector")) {
        int64_t base = DEFAULT_VECTOR_CYCLES;
        int64_t data_cost = data_size / 2048;  // 0.5 cycles per KB
        return base + data_cost;
    }
    
    // Reduction operations
    if (strstr(func_name, "reduce") || strstr(func_name, "sum") ||
        strstr(func_name, "max") || strstr(func_name, "min")) {
        int64_t base = DEFAULT_VECTOR_CYCLES * 2;
        int64_t data_cost = data_size / 1024;
        return base + data_cost;
    }
    
    // Default: vector operation
    return DEFAULT_VECTOR_CYCLES + data_size / 2048;
}

int64_t pto2_sim_estimate_cycles(PTO2TaskDescriptor* task) {
    // Calculate total data size from packed buffer
    int64_t data_size = 0;
    if (task->packed_buffer_end && task->packed_buffer_base) {
        data_size = (int64_t)((char*)task->packed_buffer_end - 
                              (char*)task->packed_buffer_base);
    }
    
    return pto2_sim_estimate_cycles_by_name(task->func_name, data_size);
}

// =============================================================================
// Simulation State Management
// =============================================================================

PTO2SimState* pto2_sim_create(const PTO2SimConfig* config) {
    PTO2SimState* sim = (PTO2SimState*)calloc(1, sizeof(PTO2SimState));
    if (!sim) {
        return NULL;
    }
    
    sim->config = *config;
    
    // Calculate total workers
    sim->num_workers = config->num_cube_cores + config->num_vector_cores;
    
    // Allocate worker states
    sim->workers = (PTO2SimWorker*)calloc(sim->num_workers, sizeof(PTO2SimWorker));
    if (!sim->workers) {
        free(sim);
        return NULL;
    }
    
    // Initialize cube workers (first set)
    for (int i = 0; i < config->num_cube_cores; i++) {
        sim->workers[i].worker_id = i;
        sim->workers[i].type = PTO2_WORKER_CUBE;
        sim->workers[i].current_cycle = 0;
        sim->workers[i].core = NULL;
        
        #ifdef A2A3_CORE_SIM_AVAILABLE
        if (config->use_core_model) {
            sim->workers[i].core = a2a3_core_create(CORE_TYPE_CUBE, i);
        }
        #endif
    }
    
    // Initialize vector workers (second set)
    for (int i = 0; i < config->num_vector_cores; i++) {
        int idx = config->num_cube_cores + i;
        sim->workers[idx].worker_id = idx;
        sim->workers[idx].type = PTO2_WORKER_VECTOR;
        sim->workers[idx].current_cycle = 0;
        sim->workers[idx].core = NULL;
        
        #ifdef A2A3_CORE_SIM_AVAILABLE
        if (config->use_core_model) {
            sim->workers[idx].core = a2a3_core_create(CORE_TYPE_VECTOR, i);
        }
        #endif
    }
    
    // Enable trace if configured
    if (config->trace_enabled && config->trace_filename) {
        pto2_sim_enable_trace(sim, config->trace_filename);
    }
    
    return sim;
}

PTO2SimState* pto2_sim_create_default(void) {
    PTO2SimConfig config = PTO2_SIM_CONFIG_DEFAULT;
    return pto2_sim_create(&config);
}

void pto2_sim_destroy(PTO2SimState* sim) {
    if (!sim) return;
    
    // Disable trace
    pto2_sim_disable_trace(sim);
    
    // Destroy core models
    #ifdef A2A3_CORE_SIM_AVAILABLE
    for (int i = 0; i < sim->num_workers; i++) {
        if (sim->workers[i].core) {
            a2a3_core_destroy(sim->workers[i].core);
        }
    }
    #endif
    
    if (sim->workers) {
        free(sim->workers);
    }
    
    free(sim);
}

void pto2_sim_reset(PTO2SimState* sim) {
    if (!sim) return;
    
    sim->global_cycle = 0;
    sim->total_task_cycles = 0;
    sim->makespan = 0;
    sim->trace_entries = 0;
    
    for (int i = 0; i < sim->num_workers; i++) {
        sim->workers[i].current_cycle = 0;
        sim->workers[i].total_compute_cycles = 0;
        sim->workers[i].total_stall_cycles = 0;
        sim->workers[i].tasks_executed = 0;
        
        #ifdef A2A3_CORE_SIM_AVAILABLE
        if (sim->workers[i].core) {
            a2a3_core_reset(sim->workers[i].core);
        }
        #endif
    }
}

// =============================================================================
// Simulation Execution
// =============================================================================

/**
 * Find an available worker of the given type
 */
static int find_available_worker(PTO2SimState* sim, PTO2WorkerType type,
                                  int64_t* min_cycle) {
    int best = -1;
    *min_cycle = INT64_MAX;
    
    for (int i = 0; i < sim->num_workers; i++) {
        if (sim->workers[i].type == type) {
            if (sim->workers[i].current_cycle < *min_cycle) {
                *min_cycle = sim->workers[i].current_cycle;
                best = i;
            }
        }
    }
    
    return best;
}

/**
 * Simulate a single task
 */
static void sim_execute_task(PTO2SimState* sim, PTO2Runtime* rt, int32_t task_id) {
    PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, task_id);
    
    // Find earliest start time based on dependencies
    int64_t earliest_start = 0;
    
    // Check fanin tasks for completion time
    int32_t current = task->fanin_head;
    while (current > 0) {
        PTO2DepListEntry* entry = pto2_dep_pool_get(&rt->orchestrator.dep_pool, current);
        if (!entry) break;
        
        // In a full implementation, we'd track end_cycle per task and use it here
        // For now, dependencies are handled by the ready queue ordering
        
        current = entry->next_offset;
    }
    
    // Find available worker
    int64_t worker_free_cycle;
    PTO2WorkerType worker_type = task->worker_type;
    
    // Map AI_CPU and ACCELERATOR to appropriate workers for simulation
    if (worker_type == PTO2_WORKER_AI_CPU || worker_type == PTO2_WORKER_ACCELERATOR) {
        worker_type = PTO2_WORKER_VECTOR;  // Use vector cores for simulation
    }
    
    int worker_id = find_available_worker(sim, worker_type, &worker_free_cycle);
    if (worker_id < 0) {
        // No worker of this type, use first available
        worker_id = 0;
        worker_free_cycle = sim->workers[0].current_cycle;
    }
    
    PTO2SimWorker* worker = &sim->workers[worker_id];
    
    // Calculate start time (max of dependency and worker availability)
    int64_t start_cycle = earliest_start > worker_free_cycle ? 
                          earliest_start : worker_free_cycle;
    
    // Estimate execution time
    int64_t exec_cycles = pto2_sim_estimate_cycles(task);
    
    #ifdef A2A3_CORE_SIM_AVAILABLE
    // Use core model if available
    if (worker->core && task->func_name) {
        int64_t data_size = 0;
        if (task->packed_buffer_end && task->packed_buffer_base) {
            data_size = (int64_t)((char*)task->packed_buffer_end - 
                                  (char*)task->packed_buffer_base);
        }
        exec_cycles = a2a3_core_issue_compute(worker->core, task->func_name, 
                                               data_size);
    }
    #endif
    
    // Update worker state
    int64_t stall_cycles = start_cycle - worker_free_cycle;
    worker->total_stall_cycles += stall_cycles;
    worker->total_compute_cycles += exec_cycles;
    worker->current_cycle = start_cycle + exec_cycles;
    worker->tasks_executed++;
    
    // Update global statistics
    sim->total_task_cycles += exec_cycles;
    if (worker->current_cycle > sim->makespan) {
        sim->makespan = worker->current_cycle;
    }
    
    // Record trace entry
    if (sim->trace_file) {
        pto2_sim_trace_task(sim, worker_id, task_id, task->func_name,
                           start_cycle, start_cycle + exec_cycles);
    }
}

int64_t pto2_sim_run(PTO2SimState* sim, PTO2Runtime* rt) {
    pto2_sim_reset(sim);
    
    // Make sure orchestration is done
    if (!rt->sm_handle->header->orchestrator_done) {
        pto2_rt_orchestration_done(rt);
    }
    
    // Process tasks in dependency order
    while (!pto2_scheduler_is_done(&rt->scheduler)) {
        // Process any new tasks
        pto2_scheduler_process_new_tasks(&rt->scheduler);
        
        bool dispatched = false;
        
        // Try to dispatch ready tasks
        for (int wtype = 0; wtype < PTO2_NUM_WORKER_TYPES; wtype++) {
            int32_t task_id = pto2_scheduler_get_ready_task(&rt->scheduler, wtype);
            
            if (task_id >= 0) {
                // Mark as running
                pto2_scheduler_mark_running(&rt->scheduler, task_id);
                
                // Simulate task execution
                sim_execute_task(sim, rt, task_id);
                
                // Mark task complete
                pto2_scheduler_on_task_complete(&rt->scheduler, task_id);
                
                dispatched = true;
            }
        }
        
        if (!dispatched) {
            // No tasks ready, wait for next event
            PTO2_SPIN_PAUSE();
        }
    }
    
    // Drain any remaining work on core models
    #ifdef A2A3_CORE_SIM_AVAILABLE
    for (int i = 0; i < sim->num_workers; i++) {
        if (sim->workers[i].core) {
            int64_t drain_cycles = a2a3_core_drain(sim->workers[i].core);
            if (drain_cycles > sim->makespan) {
                sim->makespan = drain_cycles;
            }
        }
    }
    #endif
    
    return sim->makespan;
}

// =============================================================================
// Trace API
// =============================================================================

void pto2_sim_enable_trace(PTO2SimState* sim, const char* filename) {
    if (sim->trace_file) {
        fclose(sim->trace_file);
    }
    
    sim->trace_file = fopen(filename, "w");
    if (sim->trace_file) {
        // Write trace header
        fprintf(sim->trace_file, "[\n");
        sim->trace_entries = 0;
    }
}

void pto2_sim_disable_trace(PTO2SimState* sim) {
    if (sim->trace_file) {
        // Close trace array
        fprintf(sim->trace_file, "]\n");
        fclose(sim->trace_file);
        sim->trace_file = NULL;
    }
}

void pto2_sim_trace_task(PTO2SimState* sim, int32_t worker_id, 
                          int32_t task_id, const char* func_name,
                          int64_t start_cycle, int64_t end_cycle) {
    if (!sim->trace_file) return;
    
    const char* name = func_name ? func_name : "task";
    
    // Write Chrome Tracing JSON format entry
    if (sim->trace_entries > 0) {
        fprintf(sim->trace_file, ",\n");
    }
    
    fprintf(sim->trace_file,
            "  {\"name\": \"%s\", \"cat\": \"task\", \"ph\": \"X\", "
            "\"pid\": 0, \"tid\": %d, "
            "\"ts\": %lld, \"dur\": %lld, "
            "\"args\": {\"task_id\": %d}}",
            name,
            worker_id,
            (long long)(start_cycle * 1000 / 1000),  // Scale to microseconds
            (long long)((end_cycle - start_cycle) * 1000 / 1000),
            task_id);
    
    sim->trace_entries++;
}

void pto2_sim_write_trace(PTO2SimState* sim, const char* filename) {
    if (!sim || !filename) return;
    
    // If trace was being written to different file, finalize it
    if (sim->trace_file) {
        pto2_sim_disable_trace(sim);
    }
    
    // If filename is different from config, enable trace to new file
    // For now, trace should have been captured during sim_run
}

// =============================================================================
// Statistics API
// =============================================================================

void pto2_sim_print_stats(PTO2SimState* sim) {
    printf("\n========== Simulation Statistics ==========\n\n");
    
    printf("Configuration:\n");
    printf("  Cube cores:    %d\n", sim->config.num_cube_cores);
    printf("  Vector cores:  %d\n", sim->config.num_vector_cores);
    printf("  Core model:    %s\n", sim->config.use_core_model ? "enabled" : "disabled");
    printf("\n");
    
    printf("Results:\n");
    printf("  Makespan:        %lld cycles\n", (long long)sim->makespan);
    printf("  Total task cycles: %lld\n", (long long)sim->total_task_cycles);
    printf("\n");
    
    // Per-worker-type statistics
    int64_t cube_compute = 0, cube_stall = 0, cube_tasks = 0;
    int64_t vec_compute = 0, vec_stall = 0, vec_tasks = 0;
    
    for (int i = 0; i < sim->num_workers; i++) {
        PTO2SimWorker* w = &sim->workers[i];
        if (w->type == PTO2_WORKER_CUBE) {
            cube_compute += w->total_compute_cycles;
            cube_stall += w->total_stall_cycles;
            cube_tasks += w->tasks_executed;
        } else {
            vec_compute += w->total_compute_cycles;
            vec_stall += w->total_stall_cycles;
            vec_tasks += w->tasks_executed;
        }
    }
    
    printf("Cube workers:\n");
    printf("  Tasks executed:  %lld\n", (long long)cube_tasks);
    printf("  Compute cycles:  %lld\n", (long long)cube_compute);
    printf("  Stall cycles:    %lld\n", (long long)cube_stall);
    if (sim->makespan > 0) {
        float util = (float)cube_compute / (sim->makespan * sim->config.num_cube_cores) * 100;
        printf("  Utilization:     %.1f%%\n", util);
    }
    printf("\n");
    
    printf("Vector workers:\n");
    printf("  Tasks executed:  %lld\n", (long long)vec_tasks);
    printf("  Compute cycles:  %lld\n", (long long)vec_compute);
    printf("  Stall cycles:    %lld\n", (long long)vec_stall);
    if (sim->makespan > 0) {
        float util = (float)vec_compute / (sim->makespan * sim->config.num_vector_cores) * 100;
        printf("  Utilization:     %.1f%%\n", util);
    }
    
    printf("\n==========================================\n");
}

int64_t pto2_sim_get_makespan(PTO2SimState* sim) {
    return sim ? sim->makespan : 0;
}

int64_t pto2_sim_get_total_cycles(PTO2SimState* sim) {
    return sim ? sim->total_task_cycles : 0;
}

float pto2_sim_get_utilization(PTO2SimState* sim, PTO2WorkerType type) {
    if (!sim || sim->makespan <= 0) return 0.0f;
    
    int64_t compute_cycles = 0;
    int32_t num_workers = 0;
    
    for (int i = 0; i < sim->num_workers; i++) {
        if (sim->workers[i].type == type) {
            compute_cycles += sim->workers[i].total_compute_cycles;
            num_workers++;
        }
    }
    
    if (num_workers == 0) return 0.0f;
    
    return (float)compute_cycles / (sim->makespan * num_workers);
}
