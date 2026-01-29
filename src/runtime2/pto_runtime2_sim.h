/**
 * PTO Runtime2 - Simulation Integration
 * 
 * Provides integration between PTO Runtime2 and the A2A3 Core Simulator
 * for cycle-accurate simulation of task execution.
 * 
 * Features:
 * - Cycle cost estimation based on function names
 * - Multi-core simulation with per-worker-type cores
 * - Trace generation for visualization
 * - Dependency-aware scheduling simulation
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_RUNTIME2_SIM_H
#define PTO_RUNTIME2_SIM_H

#include "pto_runtime2.h"
#include <stdio.h>

// Forward declaration for core model
struct A2A3Core;

// =============================================================================
// Simulation Configuration
// =============================================================================

/**
 * Simulation configuration
 */
typedef struct {
    int32_t num_cube_cores;       // Number of cube cores (default: 24)
    int32_t num_vector_cores;     // Number of vector cores (default: 48)
    bool    use_core_model;       // Use A2A3 core model (if available)
    bool    trace_enabled;        // Enable trace generation
    const char* trace_filename;   // Trace output file
} PTO2SimConfig;

/**
 * Default simulation configuration
 */
#define PTO2_SIM_CONFIG_DEFAULT { \
    .num_cube_cores = 24,         \
    .num_vector_cores = 48,       \
    .use_core_model = true,       \
    .trace_enabled = false,       \
    .trace_filename = NULL        \
}

// =============================================================================
// Simulation State
// =============================================================================

/**
 * Per-worker simulation state
 */
typedef struct {
    int32_t worker_id;
    PTO2WorkerType type;
    int64_t current_cycle;        // Current cycle for this worker
    int64_t total_compute_cycles; // Total cycles spent computing
    int64_t total_stall_cycles;   // Total cycles spent waiting
    int32_t tasks_executed;       // Number of tasks executed
    
    // Core model (if enabled)
    struct A2A3Core* core;
} PTO2SimWorker;

/**
 * Simulation runtime extension
 */
typedef struct {
    PTO2SimConfig config;
    
    // Per-worker state
    PTO2SimWorker* workers;
    int32_t num_workers;
    
    // Global simulation state
    int64_t global_cycle;         // Global simulation time
    int64_t total_task_cycles;    // Total cycles across all tasks
    int64_t makespan;             // Critical path length
    
    // Trace state
    FILE* trace_file;
    int32_t trace_entries;
    
} PTO2SimState;

// =============================================================================
// Simulation API
// =============================================================================

/**
 * Create simulation state
 * 
 * @param config Simulation configuration
 * @return Simulation state, or NULL on failure
 */
PTO2SimState* pto2_sim_create(const PTO2SimConfig* config);

/**
 * Create simulation state with default configuration
 */
PTO2SimState* pto2_sim_create_default(void);

/**
 * Destroy simulation state
 */
void pto2_sim_destroy(PTO2SimState* sim);

/**
 * Reset simulation state for reuse
 */
void pto2_sim_reset(PTO2SimState* sim);

/**
 * Run simulation on a runtime
 * 
 * Simulates task execution with cycle-accurate timing.
 * 
 * @param sim Simulation state
 * @param rt  Runtime with submitted tasks
 * @return Total cycles for execution
 */
int64_t pto2_sim_run(PTO2SimState* sim, PTO2Runtime* rt);

/**
 * Estimate cycle cost for a task
 * 
 * @param task Task descriptor
 * @return Estimated cycles
 */
int64_t pto2_sim_estimate_cycles(PTO2TaskDescriptor* task);

/**
 * Estimate cycle cost based on function name
 * 
 * @param func_name Function name
 * @param data_size Total data size (for bandwidth estimation)
 * @return Estimated cycles
 */
int64_t pto2_sim_estimate_cycles_by_name(const char* func_name, int64_t data_size);

// =============================================================================
// Trace API
// =============================================================================

/**
 * Enable trace generation
 */
void pto2_sim_enable_trace(PTO2SimState* sim, const char* filename);

/**
 * Disable trace generation
 */
void pto2_sim_disable_trace(PTO2SimState* sim);

/**
 * Record task execution in trace
 */
void pto2_sim_trace_task(PTO2SimState* sim, int32_t worker_id, 
                          int32_t task_id, const char* func_name,
                          int64_t start_cycle, int64_t end_cycle);

/**
 * Write trace to file in Chrome Tracing JSON format
 */
void pto2_sim_write_trace(PTO2SimState* sim, const char* filename);

// =============================================================================
// Statistics API
// =============================================================================

/**
 * Print simulation statistics
 */
void pto2_sim_print_stats(PTO2SimState* sim);

/**
 * Get simulation makespan (critical path length)
 */
int64_t pto2_sim_get_makespan(PTO2SimState* sim);

/**
 * Get total compute cycles
 */
int64_t pto2_sim_get_total_cycles(PTO2SimState* sim);

/**
 * Get utilization for a worker type
 */
float pto2_sim_get_utilization(PTO2SimState* sim, PTO2WorkerType type);

#endif // PTO_RUNTIME2_SIM_H
