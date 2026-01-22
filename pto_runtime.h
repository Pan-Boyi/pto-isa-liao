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
#include <pthread.h>

// =============================================================================
// Configuration
// =============================================================================

#define PTO_MAX_TASKS          524288  // Maximum pending tasks (512K for non-aligned seq)
#define PTO_MAX_FANOUT         512     // Maximum fanout per task
#define PTO_MAX_ARGS           16      // Maximum arguments per task
#define PTO_TENSORMAP_SIZE     16384   // Hash table size for tensor map
#define PTO_MAX_READY_QUEUE    262144  // Ready queue size (256K for large non-aligned sequences)
#define PTO_MAX_WORKERS        64      // Maximum worker threads

// =============================================================================
// Compact Replay Task (cache-optimized)
// =============================================================================
// 
// PendingTask is ~2.8KB which causes severe cache thrashing during replay
// (16K tasks = 735K cache lines touched). CompactTask is 24 bytes, fitting
// ~2.7 entries per cache line (16K tasks = 6K cache lines = 120x better).
//
// For replay tasks, we use CompactTask array instead of PendingTask array.
// This dramatically improves replay performance (from ~5K to ~15K+ tasks/ms).

struct RecordedTask;  // Forward declaration

/**
 * Compact task entry for replay (32 bytes - cache friendly)
 */
typedef struct {
    struct RecordedTask* template_ref;  // 8 bytes: immutable template
    int32_t  resolved_fanin;            // 4 bytes: incremented when deps complete
    int32_t  offset_delta;              // 4 bytes: row offset for this replay
    int64_t  earliest_start_cycle;      // 8 bytes: for dependency-aware scheduling
    bool     is_complete;               // 1 byte
    bool     is_active;                 // 1 byte  
    int16_t  _padding;                  // 2 bytes alignment
} CompactTask;

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
/**
 * Cycle cost function pointer type
 * Returns estimated cycle count for the InCore function
 */
typedef int64_t (*CycleCostFunc)(void** args, int32_t num_args);

typedef struct {
    int32_t      task_id;                    // Unique task identifier
    const char*  func_name;                  // InCore function to call
    void*        func_ptr;                   // Function pointer
    CycleCostFunc cycle_func;                // Cycle cost function (for simulation mode)
    
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
    
    // Worker type hint (for a2a3_sim backend)
    bool         is_cube;                    // True if requires cube unit (matmul)
    
    // Timing (for dependency-aware simulation)
    int64_t      earliest_start_cycle;       // Earliest time this task can start (after deps)
    int64_t      end_cycle;                  // Time when this task finished
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
    // Task management - dual arrays for optimal cache behavior
    PendingTask  pend_task[PTO_MAX_TASKS];    // Full task table (for direct tasks)
    CompactTask  compact_task[PTO_MAX_TASKS]; // Compact array for replay tasks (24 bytes each)
    int32_t      next_task_id;                // Next available task ID
    int32_t      active_task_count;           // Number of active tasks
    
    // TensorMap for dependency tracking
    TensorMapEntry* tensor_map[PTO_TENSORMAP_SIZE];
    
    // Ready queue (tasks with fanin == 0) - legacy single queue for backward compat
    int32_t      ready_queue[PTO_MAX_READY_QUEUE];
    int32_t      ready_head;
    int32_t      ready_tail;
    int32_t      ready_count;
    
    // ==========================================================================
    // Dual ready queues for a2a3_sim: vector (is_cube=0) and cube (is_cube=1)
    // ==========================================================================
    int32_t      vector_ready_queue[PTO_MAX_READY_QUEUE];  // is_cube=0 tasks
    int32_t      vector_ready_head;
    int32_t      vector_ready_tail;
    int32_t      vector_ready_count;
    
    int32_t      cube_ready_queue[PTO_MAX_READY_QUEUE];    // is_cube=1 tasks
    int32_t      cube_ready_head;
    int32_t      cube_ready_tail;
    int32_t      cube_ready_count;
    
    pthread_cond_t    vector_queue_not_empty;  // Signaled when vector task added
    pthread_cond_t    cube_queue_not_empty;    // Signaled when cube task added
    
    // Statistics
    int64_t      total_tasks_scheduled;
    int64_t      total_tasks_completed;
    
    // ==========================================================================
    // Multi-threaded execution support
    // ==========================================================================
    
    // Thread synchronization
    pthread_mutex_t   queue_mutex;           // Protects ready_queue access
    pthread_mutex_t   task_mutex;            // Protects task state updates
    pthread_cond_t    queue_not_empty;       // Signaled when task added to queue
    pthread_cond_t    all_done;              // Signaled when all tasks complete
    
    // Worker threads
    pthread_t         workers[PTO_MAX_WORKERS];
    int32_t           num_workers;           // Total number of worker threads
    int32_t           num_vector_workers;    // Number of vector workers (is_cube=0)
    int32_t           num_cube_workers;      // Number of cube workers (is_cube=1)
    volatile bool     shutdown_requested;    // Signal workers to exit
    volatile bool     execution_started;     // Orchestration has submitted all tasks
    int32_t           execution_task_threshold;  // Start workers when task_count > threshold
    
    // Simulation mode (uses cycle_func instead of func_ptr)
    bool              simulation_mode;       // If true, call cycle_func and record traces
    bool              dual_queue_mode;       // If true, use separate cube/vector queues
    
    // InCore function registry (maps func_name to actual function pointer)
    // This is populated before execution starts
    void*             func_registry[PTO_MAX_TASKS];  // Indexed by task_id after lookup
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
 * Enable simulation mode with cycle tracing
 * In simulation mode:
 * - Tasks call their cycle_func instead of func_ptr
 * - Cycle counts are recorded to the global trace
 * - Results can be visualized in Chrome Tracing
 * 
 * @param rt            Runtime context
 * @param num_workers   Number of simulated worker threads for trace
 */
void pto_runtime_enable_simulation(PTORuntime* rt, int32_t num_workers);

/**
 * Enable simulation mode with separate cube and vector workers (a2a3_sim mode)
 * 
 * In a2a3_sim mode:
 * - Creates num_vector_workers workers for is_cube=0 tasks (vector operations)
 * - Creates num_cube_workers workers for is_cube=1 tasks (matmul operations)
 * - Uses dual ready queues for separate scheduling
 * 
 * Typical configuration for A2/A3:
 * - 48 vector workers
 * - 24 cube workers
 * 
 * @param rt                 Runtime context
 * @param num_vector_workers Number of vector workers (for is_cube=0 tasks)
 * @param num_cube_workers   Number of cube workers (for is_cube=1 tasks)
 */
void pto_runtime_enable_a2a3_sim(PTORuntime* rt, int32_t num_vector_workers, int32_t num_cube_workers);

/**
 * Get next ready task for a vector worker (is_cube=0 tasks only)
 * Returns task_id or -1 if no tasks ready
 */
int32_t pto_get_ready_task_vector(PTORuntime* rt);

/**
 * Get next ready task for a cube worker (is_cube=1 tasks only)
 * Returns task_id or -1 if no tasks ready
 */
int32_t pto_get_ready_task_cube(PTORuntime* rt);

/**
 * Thread-safe get ready task for vector worker (blocking)
 */
int32_t pto_get_ready_task_vector_blocking(PTORuntime* rt);

/**
 * Thread-safe get ready task for cube worker (blocking)
 */
int32_t pto_get_ready_task_cube_blocking(PTORuntime* rt);

/**
 * Allocate a new task ID and initialize task entry (internal implementation)
 * @param rt            Runtime context
 * @param func_name     InCore function name
 * @param func_ptr      Function pointer (can be NULL)
 * @param buffer_bytes  Estimated tile buffer size in bytes (without reuse)
 * @param reuse_bytes   Estimated tile buffer size with reuse optimization
 * @param is_cube       If true, task requires cube unit (scheduled on cube workers)
 * Returns task_id or -1 on failure
 */
int32_t pto_task_alloc_impl(PTORuntime* rt, const char* func_name, void* func_ptr,
                            int32_t buffer_bytes, int32_t reuse_bytes, bool is_cube);

/**
 * Backward compatible task alloc - accepts 5 or 6 arguments
 * If is_cube is not provided, defaults to false (vector worker)
 */
static inline int32_t pto_task_alloc_5(PTORuntime* rt, const char* func_name, 
                                       void* func_ptr, int32_t buffer_bytes, 
                                       int32_t reuse_bytes) {
    return pto_task_alloc_impl(rt, func_name, func_ptr, buffer_bytes, reuse_bytes, false);
}

static inline int32_t pto_task_alloc_6(PTORuntime* rt, const char* func_name, 
                                       void* func_ptr, int32_t buffer_bytes, 
                                       int32_t reuse_bytes, bool is_cube) {
    return pto_task_alloc_impl(rt, func_name, func_ptr, buffer_bytes, reuse_bytes, is_cube);
}

/**
 * Macro to select correct overload based on argument count
 */
#define _PTO_TASK_ALLOC_NARG(...) _PTO_TASK_ALLOC_NARG_(__VA_ARGS__, 6, 5, 4, 3, 2, 1, 0)
#define _PTO_TASK_ALLOC_NARG_(_1, _2, _3, _4, _5, _6, N, ...) N

#define _PTO_TASK_ALLOC_DISPATCH(N) _PTO_TASK_ALLOC_DISPATCH_(N)
#define _PTO_TASK_ALLOC_DISPATCH_(N) pto_task_alloc_##N

#define pto_task_alloc(...) _PTO_TASK_ALLOC_DISPATCH(_PTO_TASK_ALLOC_NARG(__VA_ARGS__))(__VA_ARGS__)

/**
 * Set the cycle cost function for a task (for simulation mode)
 */
void pto_task_set_cycle_func(PTORuntime* rt, int32_t task_id, CycleCostFunc cycle_func);

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
 * Execute all pending tasks until completion (single-threaded)
 */
void pto_execute_all(PTORuntime* rt);

/**
 * Execute a single task with specified worker ID (for simulation mode)
 * Records cycle trace when simulation mode is enabled
 */
void pto_execute_task_with_worker(PTORuntime* rt, int32_t task_id, int32_t worker_id);

// =============================================================================
// Multi-threaded Execution API
// =============================================================================

/**
 * InCore function signature for ARM64
 * All InCore functions must match this signature
 */
typedef void (*PTOInCoreFunc)(void** args, int32_t num_args);

/**
 * Orchestration function signature
 * Called to build the task graph
 */
typedef void (*PTOOrchFunc)(PTORuntime* rt, void* user_data);

/**
 * Register an InCore function with the runtime
 * @param rt        Runtime context
 * @param func_name Name of the InCore function
 * @param func_ptr  Function pointer (must match PTOInCoreFunc signature)
 */
void pto_register_incore_func(PTORuntime* rt, const char* func_name, PTOInCoreFunc func_ptr);

/**
 * ARM64 Runtime Entry Point - Multi-threaded task execution
 * 
 * This is the main entry point for executing PTO programs on ARM64.
 * 
 * Execution flow:
 * 1. Initialize runtime and spawn worker threads
 * 2. Call orchestration function to build task graph
 * 3. Workers execute tasks from ready queue in parallel
 * 4. Wait for all tasks to complete
 * 5. Shutdown workers and cleanup
 * 
 * @param orch_func               Orchestration function that builds the task graph
 * @param user_data               User data passed to orchestration function
 * @param num_workers             Number of worker threads (1-PTO_MAX_WORKERS)
 * @param execution_task_threshold  Task threshold to start execution:
 *                                  - 0: Wait until orchestration completes (default, safe)
 *                                  - >0: Start when active_task_count > threshold OR orchestration done
 *                                  This enables pipelining task graph building with execution.
 * @return 0 on success, -1 on failure
 */
int runtime_entry_arm64(PTOOrchFunc orch_func, void* user_data, int num_workers, 
                        int execution_task_threshold);

/**
 * Thread-safe version of pto_get_ready_task
 * Blocks until a task is available or shutdown is requested
 * @param rt Runtime context
 * @return task_id or -1 if shutdown
 */
int32_t pto_get_ready_task_blocking(PTORuntime* rt);

/**
 * Thread-safe version of pto_task_complete
 * Updates dependencies and signals waiting workers
 */
void pto_task_complete_threadsafe(PTORuntime* rt, int32_t task_id);

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

// =============================================================================
// Record & Replay API (for loop optimization)
// =============================================================================

/**
 * Recorded task entry - immutable template for replay
 * Contains all data needed to replay a task without re-analyzing dependencies
 */
typedef struct RecordedTask {
    const char*  func_name;                    // InCore function name
    void*        func_ptr;                     // Function pointer
    CycleCostFunc cycle_func;                  // Cycle cost function (for simulation)
    int32_t      buffer_size_bytes;            // Buffer size estimation
    int32_t      buffer_size_with_reuse;       // Buffer size with reuse
    int32_t      fanin;                        // Initial fanin count (immutable)
    int32_t      internal_fanin;               // Fanin from within same fragment (for replay)
    int32_t      fanout[PTO_MAX_FANOUT];       // Relative fanout offsets
    int32_t      fanout_count;                 // Number of fanouts
    TaskArg      args[PTO_MAX_ARGS];           // Arguments (with template regions)
    int32_t      num_args;                     // Number of arguments
    bool         is_cube;                      // True if requires cube unit (matmul)
} RecordedTask;

/**
 * Recorded output - for TensorMap replay
 */
typedef struct {
    TensorRegion region;              // Tensor region (with template offsets)
    int32_t      relative_producer;   // Offset from fragment base
} RecordedOutput;

/**
 * Recorded fragment - a replayable task graph fragment
 */
typedef struct {
    RecordedTask*   tasks;            // Array of recorded tasks
    int32_t         task_count;       // Number of tasks
    RecordedOutput* outputs;          // Array of output registrations
    int32_t         output_count;     // Number of outputs
    const char*     fragment_name;    // Human-readable name
    int32_t         checksum;         // Simple checksum for validation
} RecordedFragment;

/**
 * Offset mode for loop replay
 */
typedef enum {
    OFFSET_NONE = 0,    // No offset adjustment
    OFFSET_ROW,         // Adjust row_offset only
    OFFSET_COL,         // Adjust col_offset only
    OFFSET_ROW_COL      // Adjust both row and col offset
} OffsetMode;

/**
 * Loop replay context - manages record/replay for a single loop
 */
typedef struct {
    RecordedFragment* fragment;       // Recorded fragment (NULL before first record)
    int32_t           record_start;   // Start task_id for recording (-1 if not recording)
    int32_t           base_offset;    // Base offset from first recorded iteration
    int32_t           stride;         // Offset stride per iteration
    OffsetMode        offset_mode;    // How to adjust offsets during replay
    const char*       loop_name;      // For debugging
} LoopReplayCtx;

/**
 * Global flag to enable/disable loop replay optimization
 * When disabled (0), pto_loop_should_record always returns true (direct task creation)
 * When enabled (1, default), uses record/replay for cache efficiency
 */
extern int pto_record_replay_enabled;

/**
 * Enable or disable loop replay optimization
 */
void pto_set_record_replay(int enabled);

/**
 * Initialize loop replay context
 */
void pto_loop_init(LoopReplayCtx* ctx, const char* name, int32_t stride, OffsetMode mode);

/**
 * Check if we should record this iteration (returns true) or replay (returns false)
 * If pto_record_replay_enabled is 0, always returns true (no replay)
 */
bool pto_loop_should_record(PTORuntime* rt, LoopReplayCtx* ctx, int32_t loop_idx);

/**
 * Finish recording the current iteration
 */
void pto_loop_finish_record(PTORuntime* rt, LoopReplayCtx* ctx);

/**
 * Replay the recorded fragment for this iteration
 * Uses compact_task array for cache-efficient replay
 */
void pto_loop_replay(PTORuntime* rt, LoopReplayCtx* ctx, int32_t loop_idx);

/**
 * Cleanup loop replay context
 */
void pto_loop_cleanup(LoopReplayCtx* ctx);

/**
 * Validate that a task matches the recorded template (for debugging)
 * Call this during development to verify replay correctness
 * Returns true if the task arguments are compatible with replay
 */
bool pto_loop_validate_task(LoopReplayCtx* ctx, int32_t task_idx_in_fragment,
                            int32_t loop_idx, TaskArg* args, int32_t num_args);

/**
 * Record a range of tasks as a fragment
 */
RecordedFragment* pto_fragment_record(PTORuntime* rt, int32_t start_id, int32_t end_id,
                                      const char* name);

/**
 * Free a recorded fragment
 */
void pto_fragment_free(RecordedFragment* fragment);

/**
 * Get fragment size in bytes
 */
size_t pto_fragment_size(RecordedFragment* fragment);

// =============================================================================
// Cycle Trace Recording for Performance Analysis
// =============================================================================

#define PTO_MAX_TRACE_ENTRIES 524288  // 512K for large task graphs
#define PTO_MAX_FUNC_NAME_LEN 64

/**
 * Single trace entry recording one task execution
 */
typedef struct {
    char func_name[PTO_MAX_FUNC_NAME_LEN];
    int32_t worker_id;
    int64_t start_cycle;
    int64_t end_cycle;
} CycleTraceEntry;

/**
 * Cycle trace buffer for recording task execution timing
 */
typedef struct {
    CycleTraceEntry entries[PTO_MAX_TRACE_ENTRIES];
    int32_t count;
    int32_t num_workers;
    int64_t per_worker_cycle[PTO_MAX_WORKERS];  // Current cycle per worker
    bool enabled;
} CycleTrace;

/**
 * Global cycle trace (for single-trace use case)
 */
extern CycleTrace* pto_global_trace;

/**
 * Initialize cycle tracing
 */
void pto_trace_init(int32_t num_workers);

/**
 * Record a task execution (simple version - no dependency tracking)
 */
void pto_trace_record(int32_t worker_id, const char* func_name, int64_t cycle_cost);

/**
 * Record a task execution with explicit timing (dependency-aware)
 */
void pto_trace_record_with_time(int32_t worker_id, const char* func_name, 
                                 int64_t start_cycle, int64_t end_cycle);

/**
 * Get the current cycle for a worker
 */
int64_t pto_trace_get_cycle(int32_t worker_id);

/**
 * Cleanup trace resources
 */
void pto_trace_cleanup(void);

/**
 * Generate Chrome Tracing JSON format (for chrome://tracing visualization)
 * Returns newly allocated string that caller must free
 */
char* pto_trace_to_chrome_json(void);

/**
 * Write trace to file in Chrome Tracing JSON format
 */
void pto_trace_write_json(const char* filename);

/**
 * Print trace summary statistics
 */
void pto_trace_print_summary(void);

#endif // PTO_RUNTIME_H
