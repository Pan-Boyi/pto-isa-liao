/**
 * PTO Runtime System - Header
 * 
 * This runtime manages task scheduling for PTO programs.
 * 
 * When an Orchestration function calls InCore functions:
 * 1. Each InCore call becomes a pending task with a task_id
 * 2. Tasks track producer-consumer dependencies via fanin/fanout
 * 3. TensorMap tracks which task produces each tensor region
 * 
 * Execution model:
 * - Orchestration functions run on host CPU
 * - InCore functions are scheduled as tasks with data dependencies
 * - Tasks with fanin==0 are ready to execute
 */

#ifndef PTO_RUNTIME_H
#define PTO_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// =============================================================================
// Configuration
// =============================================================================

#define PTO_MAX_TASKS          262144  // Maximum pending tasks (256K for 16K+ seq)
#define PTO_MAX_FANOUT         512     // Maximum fanout per task
#define PTO_MAX_ARGS           16      // Maximum arguments per task
#define PTO_TENSORMAP_SIZE     16384   // Hash table size for tensor map
#define PTO_MAX_READY_QUEUE    4096    // Ready queue size

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Tensor region identifier
 * Uniquely identifies a tensor region by base pointer, offset, and shape
 */
typedef struct {
    void*    raw_tensor;     // Base pointer to tensor data
    int64_t  row_offset;     // Row offset within tensor
    int64_t  col_offset;     // Column offset within tensor
    int64_t  rows;           // Number of rows in this region
    int64_t  cols;           // Number of columns in this region
} TensorRegion;

/**
 * Task argument - either input or output tensor
 */
typedef struct {
    TensorRegion region;     // Tensor region
    bool         is_output;  // True if this is an output argument
} TaskArg;

/**
 * Pending task entry
 */
typedef struct {
    int32_t      task_id;                    // Unique task identifier
    const char*  func_name;                  // InCore function to call
    void*        func_ptr;                   // Function pointer
    
    // Arguments
    TaskArg      args[PTO_MAX_ARGS];         // Input/output arguments
    int32_t      num_args;                   // Number of arguments
    
    // Buffer size estimation
    int32_t      buffer_size_bytes;          // Estimated InCore tile buffer size
    int32_t      buffer_size_with_reuse;     // Buffer size with reuse optimization
    
    // Dependency tracking
    int32_t      fanin;                      // Number of input dependencies remaining
    int32_t      fanout[PTO_MAX_FANOUT];     // Task IDs that depend on this task
    int32_t      fanout_count;               // Number of dependent tasks
    
    // Status
    bool         is_active;                  // Task slot is in use
    bool         is_complete;                // Task has finished execution
} PendingTask;

/**
 * TensorMap entry - maps tensor region to producing task
 */
typedef struct TensorMapEntry {
    TensorRegion           region;       // Tensor region key
    int32_t                producer_id;  // Task that produces this region
    struct TensorMapEntry* next;         // Next entry in hash chain
} TensorMapEntry;

/**
 * PTO Runtime context
 */
typedef struct {
    // Task management
    PendingTask  pend_task[PTO_MAX_TASKS];   // Pending task table
    int32_t      next_task_id;               // Next available task ID
    int32_t      active_task_count;          // Number of active tasks
    
    // TensorMap for dependency tracking
    TensorMapEntry* tensor_map[PTO_TENSORMAP_SIZE];
    
    // Ready queue (tasks with fanin == 0)
    int32_t      ready_queue[PTO_MAX_READY_QUEUE];
    int32_t      ready_head;
    int32_t      ready_tail;
    int32_t      ready_count;
    
    // Statistics
    int64_t      total_tasks_scheduled;
    int64_t      total_tasks_completed;
} PTORuntime;

// =============================================================================
// Runtime API
// =============================================================================

/**
 * Initialize the PTO runtime
 */
void pto_runtime_init(PTORuntime* rt);

/**
 * Shutdown the PTO runtime and free resources
 */
void pto_runtime_shutdown(PTORuntime* rt);

/**
 * Allocate a new task ID and initialize task entry
 * @param rt            Runtime context
 * @param func_name     InCore function name
 * @param func_ptr      Function pointer (can be NULL)
 * @param buffer_bytes  Estimated tile buffer size in bytes (without reuse)
 * @param reuse_bytes   Estimated tile buffer size with reuse optimization
 * Returns task_id or -1 on failure
 */
int32_t pto_task_alloc(PTORuntime* rt, const char* func_name, void* func_ptr,
                       int32_t buffer_bytes, int32_t reuse_bytes);

/**
 * Add an input argument to a task
 * Looks up producer in TensorMap and updates dependencies
 */
void pto_task_add_input(PTORuntime* rt, int32_t task_id,
                        void* tensor, int64_t row_off, int64_t col_off,
                        int64_t rows, int64_t cols);

/**
 * Add an output argument to a task
 * Registers the output in TensorMap
 */
void pto_task_add_output(PTORuntime* rt, int32_t task_id,
                         void* tensor, int64_t row_off, int64_t col_off,
                         int64_t rows, int64_t cols);

/**
 * Finalize task setup and add to pending queue
 * If fanin == 0, task is added to ready queue
 */
void pto_task_submit(PTORuntime* rt, int32_t task_id);

/**
 * Mark a task as complete and update dependencies
 * Decrements fanin of dependent tasks, adds newly ready tasks to queue
 */
void pto_task_complete(PTORuntime* rt, int32_t task_id);

/**
 * Get next ready task from queue
 * Returns task_id or -1 if no tasks ready
 */
int32_t pto_get_ready_task(PTORuntime* rt);

/**
 * Execute all pending tasks until completion
 */
void pto_execute_all(PTORuntime* rt);

/**
 * Print runtime statistics
 */
void pto_runtime_stats(PTORuntime* rt);

/**
 * Dump runtime state to a text file
 * Includes: task table, fanout lists, fanin counters, ready queue, tensor map
 * @param rt       Runtime context
 * @param filename Output filename
 * @return 0 on success, -1 on failure
 */
int pto_runtime_dump(PTORuntime* rt, const char* filename);

/**
 * Dump runtime state to stdout (condensed format)
 * @param rt Runtime context
 * @return 0 on success, -1 on failure
 */
int pto_runtime_dump_stdout(PTORuntime* rt);

// =============================================================================
// TensorMap API (internal)
// =============================================================================

/**
 * Compute hash for tensor region
 */
uint32_t pto_tensormap_hash(TensorRegion* region);

/**
 * Check if two tensor regions match
 */
bool pto_region_match(TensorRegion* a, TensorRegion* b);

/**
 * Lookup producer task for a tensor region
 * Returns task_id or -1 if not found
 */
int32_t pto_tensormap_lookup(PTORuntime* rt, TensorRegion* region);

/**
 * Insert/update tensor region -> task mapping
 */
void pto_tensormap_insert(PTORuntime* rt, TensorRegion* region, int32_t task_id);

/**
 * Clear the tensor map
 */
void pto_tensormap_clear(PTORuntime* rt);

#endif // PTO_RUNTIME_H
