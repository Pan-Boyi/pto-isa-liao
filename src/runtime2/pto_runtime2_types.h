/**
 * PTO Runtime2 - Core Type Definitions
 * 
 * This header defines all fundamental types used by the PTO Runtime2 system:
 * - Configuration constants
 * - Worker types and task states
 * - Tensor regions and task parameters
 * - Task descriptors with fanin/fanout tracking
 * - Dependency list entries
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_RUNTIME2_TYPES_H
#define PTO_RUNTIME2_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// =============================================================================
// Configuration Constants
// =============================================================================

// Task management
#define PTO2_TASK_WINDOW_SIZE     1024    // Power of 2 for efficient modulo
#define PTO2_TASK_SLOT(task_id)   ((task_id) & (PTO2_TASK_WINDOW_SIZE - 1))

// Memory pools
#define PTO2_HEAP_SIZE            (64 * 1024 * 1024)  // 64MB default heap
#define PTO2_DEP_LIST_POOL_SIZE   8192    // Dependency list pool entries
#define PTO2_TENSORMAP_POOL_SIZE  4096    // TensorMap entry pool
#define PTO2_TENSORMAP_NUM_BUCKETS 1024   // Power of 2 for fast hash

// Task parameters
#define PTO2_MAX_OUTPUTS          16      // Maximum outputs per task
#define PTO2_MAX_INPUTS           16      // Maximum inputs per task
#define PTO2_MAX_INOUTS           8       // Maximum in-out params per task

// Scope management
#define PTO2_MAX_SCOPE_DEPTH      64      // Maximum nesting depth

// Ready queue
#define PTO2_READY_QUEUE_SIZE     4096    // Per-worker-type queue size

// Memory alignment
#define PTO2_ALIGN_SIZE           64      // Cache line alignment
#define PTO2_ALIGN_UP(x, align)   (((x) + (align) - 1) & ~((align) - 1))

// TensorMap cleanup interval
#define PTO2_TENSORMAP_CLEANUP_INTERVAL 64  // Cleanup every N retired tasks

// =============================================================================
// Worker Types
// =============================================================================

/**
 * Worker type enumeration
 * Each worker type has its own ready queue for load balancing
 */
typedef enum {
    PTO2_WORKER_CUBE = 0,       // AICore CUBE unit (matrix ops)
    PTO2_WORKER_VECTOR = 1,     // AICore VECTOR unit (element-wise ops)
    PTO2_WORKER_AI_CPU = 2,     // AI_CPU (scalar ops, control flow)
    PTO2_WORKER_ACCELERATOR = 3,// Fixed-function accelerators (DMA, etc.)
    PTO2_NUM_WORKER_TYPES = 4
} PTO2WorkerType;

// =============================================================================
// Task States
// =============================================================================

/**
 * Task state enumeration
 * 
 * State transitions:
 *   PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED
 * 
 * Conditions:
 *   PENDING->READY:     fanin_refcount == fanin_count
 *   COMPLETED->CONSUMED: fanout_refcount == fanout_count && state == COMPLETED
 */
typedef enum {
    PTO2_TASK_PENDING = 0,    // Waiting for dependencies (fanin_refcount < fanin_count)
    PTO2_TASK_READY = 1,      // All dependencies satisfied, waiting in ready queue
    PTO2_TASK_RUNNING = 2,    // Currently executing on a worker
    PTO2_TASK_COMPLETED = 3,  // Execution finished, output may still be in use
    PTO2_TASK_CONSUMED = 4    // Output fully consumed, buffers can be released
} PTO2TaskState;

// =============================================================================
// Tensor Region
// =============================================================================

/**
 * Tensor region identifier
 * Uniquely identifies a region within a tensor buffer
 */
typedef struct {
    void*    base_ptr;        // Buffer base pointer
    int32_t  tile_index;      // Tile index within buffer
    int32_t  offset;          // Byte offset within tile
    int32_t  size;            // Region size in bytes
} PTO2TensorRegion;

// =============================================================================
// Task Parameter
// =============================================================================

/**
 * Task parameter type enumeration
 */
typedef enum {
    PTO2_PARAM_INPUT = 0,     // Read-only input
    PTO2_PARAM_OUTPUT = 1,    // Write-only output
    PTO2_PARAM_INOUT = 2      // Read-write (accumulation)
} PTO2ParamType;

/**
 * Task parameter descriptor
 * Describes one input/output/inout buffer for a task
 */
typedef struct {
    PTO2ParamType type;       // Parameter type
    void*         buffer;     // Buffer base pointer
    int32_t       tile_index; // Tile index
    int32_t       size;       // Size in bytes
} PTO2TaskParam;

// =============================================================================
// Dependency List Entry
// =============================================================================

/**
 * Dependency list entry (singly-linked list node)
 * Stored in DepListPool ring buffer
 * 
 * Used for both fanin_list and fanout_list
 */
typedef struct {
    int32_t task_id;          // The dependent/dependency task ID
    int32_t next_offset;      // Offset to next entry (0 = end of list)
} PTO2DepListEntry;

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure
 * 
 * Stored in the TaskDescriptor ring buffer in shared memory.
 * Contains both static info (set at submission) and dynamic state.
 * 
 * Concurrency notes:
 * - fanout_head, fanout_count protected by fanout_lock (per-task spinlock)
 * - fanin_head, fanin_count set once at submission, read-only after
 * - Other fields set by Orchestrator, read by Scheduler
 */
typedef struct {
    // Task identification
    int32_t task_id;              // Unique task identifier (absolute, not wrapped)
    int32_t kernel_id;            // InCore function to execute
    int32_t worker_type;          // Target: CUBE, VECTOR, AI_CPU, ACCELERATOR
    int32_t scope_depth;          // Depth of scope when task was created
    
    // Dependency lists (linked list heads - offsets into DepListPool)
    // Fanin: producers this task depends on (set once at submission)
    int32_t fanin_head;           // Offset to first fanin entry (0 = empty)
    int32_t fanin_count;          // Number of producer dependencies
    
    // Fanout: consumers that depend on this task (grows as consumers submit)
    // PROTECTED BY fanout_lock
    volatile int32_t fanout_lock; // Per-task spinlock (0=unlocked, 1=locked)
    volatile int32_t fanout_head; // Offset to first fanout entry (0 = empty)
    volatile int32_t fanout_count;// Total consumers + scope_depth (for lifecycle)
    
    // Packed output buffer (all outputs packed into single contiguous buffer)
    void*    packed_buffer_base;  // Start of packed buffer in GM Heap
    void*    packed_buffer_end;   // End of packed buffer (for heap reclamation)
    int32_t  output_offsets[PTO2_MAX_OUTPUTS]; // Offset of each output within packed buffer
    int32_t  num_outputs;         // Number of output buffers
    
    // Input buffer pointers (for dependency resolution)
    int32_t  num_inputs;          // Number of input buffers
    
    // Function pointer (for execution)
    void*    func_ptr;            // InCore function pointer
    const char* func_name;        // Function name (for debugging/tracing)
    
    // Status flags
    bool     is_active;           // Task slot is in use
    
} PTO2TaskDescriptor;

// =============================================================================
// TensorMap Entry
// =============================================================================

/**
 * TensorMap entry structure
 * Maps tensor region -> producer task ID
 * 
 * Stored in ring buffer pool with lazy invalidation:
 * - Entry is valid only if producer_task_id >= last_task_alive
 * - Stale entries ignored during lookup
 * - Pool wraps around, overwriting stale entries
 * 
 * Chain truncation optimization:
 * - Entries in bucket chains sorted by task_id (newest first)
 * - When lookup hits stale entry, truncate rest of chain
 */
typedef struct {
    PTO2TensorRegion region;      // Tensor region key
    int32_t producer_task_id;     // Task that produces this region
    int32_t next_in_bucket;       // Offset to next entry in hash bucket (-1 = end)
    int32_t next_in_task;         // Offset to next entry for same task (-1 = end)
    bool    in_bucket;            // True if entry is linked in a bucket chain
                                  // CRITICAL: Must be set false before overwriting!
} PTO2TensorMapEntry;

// =============================================================================
// Cycle Cost Function Type
// =============================================================================

/**
 * Cycle cost function pointer type
 * Returns estimated cycle count for the InCore function
 */
typedef int64_t (*PTO2CycleCostFunc)(void** args, int32_t num_args);

// =============================================================================
// InCore Function Type
// =============================================================================

/**
 * InCore function signature
 * All InCore functions must match this signature
 */
typedef void (*PTO2InCoreFunc)(void** args, int32_t num_args);

// =============================================================================
// Utility Macros
// =============================================================================

/**
 * Memory barrier macros for different architectures
 */
#if defined(__aarch64__)
    #define PTO2_MEMORY_BARRIER()     __asm__ __volatile__("dmb sy" ::: "memory")
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#elif defined(__x86_64__)
    #define PTO2_MEMORY_BARRIER()     __asm__ __volatile__("mfence" ::: "memory")
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#else
    #define PTO2_MEMORY_BARRIER()     __sync_synchronize()
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#endif

/**
 * Pause instruction for spin-wait loops
 */
#if defined(__aarch64__)
    #define PTO2_SPIN_PAUSE()         __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__)
    #define PTO2_SPIN_PAUSE()         __builtin_ia32_pause()
#else
    #define PTO2_SPIN_PAUSE()         ((void)0)
#endif

/**
 * Atomic compare-and-swap
 */
#define PTO2_CAS(ptr, expected, desired) \
    __atomic_compare_exchange_n(ptr, expected, desired, false, \
                                __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)

/**
 * Atomic fetch-and-add
 */
#define PTO2_FETCH_ADD(ptr, val) \
    __atomic_fetch_add(ptr, val, __ATOMIC_ACQ_REL)

/**
 * Atomic exchange
 */
#define PTO2_EXCHANGE(ptr, val) \
    __atomic_exchange_n(ptr, val, __ATOMIC_ACQ_REL)

#endif // PTO_RUNTIME2_TYPES_H
