/**
 * PTO Runtime System - Implementation
 * 
 * Task scheduling runtime for PTO programs.
 * Manages dependencies between InCore function calls in Orchestration functions.
 */

#include "pto_runtime.h"
#include <time.h>  // For nanosleep, clock_gettime

// Debug output control - set to 0 to disable debug prints
#ifndef PTO_DEBUG
#define PTO_DEBUG 0
#endif

#if PTO_DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) ((void)0)
#endif

// =============================================================================
// Runtime Initialization
// =============================================================================

void pto_runtime_init(PTORuntime* rt) {
    if (!rt) return;
    
    // Initialize task table
    memset(rt->pend_task, 0, sizeof(rt->pend_task));
    
    // Initialize compact task array (for replay tasks)
    // Setting template_ref to NULL marks entries as "not compact"
    memset(rt->compact_task, 0, sizeof(rt->compact_task));
    
    rt->next_task_id = 0;
    rt->active_task_count = 0;
    
    // Initialize tensor map
    memset(rt->tensor_map, 0, sizeof(rt->tensor_map));
    
    // Initialize legacy ready queue
    memset(rt->ready_queue, 0, sizeof(rt->ready_queue));
    rt->ready_head = 0;
    rt->ready_tail = 0;
    rt->ready_count = 0;
    
    // Initialize dual ready queues (for a2a3_sim mode)
    memset(rt->vector_ready_queue, 0, sizeof(rt->vector_ready_queue));
    rt->vector_ready_head = 0;
    rt->vector_ready_tail = 0;
    rt->vector_ready_count = 0;
    
    memset(rt->cube_ready_queue, 0, sizeof(rt->cube_ready_queue));
    rt->cube_ready_head = 0;
    rt->cube_ready_tail = 0;
    rt->cube_ready_count = 0;
    
    // Initialize statistics
    rt->total_tasks_scheduled = 0;
    rt->total_tasks_completed = 0;
    
    // Initialize thread synchronization primitives
    pthread_mutex_init(&rt->queue_mutex, NULL);
    pthread_mutex_init(&rt->task_mutex, NULL);
    pthread_cond_init(&rt->queue_not_empty, NULL);
    pthread_cond_init(&rt->all_done, NULL);
    pthread_cond_init(&rt->vector_queue_not_empty, NULL);
    pthread_cond_init(&rt->cube_queue_not_empty, NULL);
    
    // Initialize worker state
    rt->num_workers = 0;
    rt->num_vector_workers = 0;
    rt->num_cube_workers = 0;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    rt->execution_task_threshold = 0;
    rt->simulation_mode = false;
    rt->dual_queue_mode = false;
    memset(rt->workers, 0, sizeof(rt->workers));
    memset(rt->func_registry, 0, sizeof(rt->func_registry));
    
    DEBUG_PRINT("[PTO Runtime] Initialized (max_tasks=%d, tensormap_size=%d)\n",
           PTO_MAX_TASKS, PTO_TENSORMAP_SIZE);
}

void pto_runtime_enable_simulation(PTORuntime* rt, int32_t num_workers) {
    if (!rt) return;
    rt->simulation_mode = true;
    rt->dual_queue_mode = false;
    pto_trace_init(num_workers > 0 ? num_workers : 1);
    DEBUG_PRINT("[PTO Runtime] Simulation mode enabled with %d workers\n", num_workers);
}

void pto_runtime_enable_a2a3_sim(PTORuntime* rt, int32_t num_vector_workers, int32_t num_cube_workers) {
    if (!rt) return;
    rt->simulation_mode = true;
    rt->dual_queue_mode = true;
    rt->num_vector_workers = num_vector_workers;
    rt->num_cube_workers = num_cube_workers;
    int total_workers = num_vector_workers + num_cube_workers;
    pto_trace_init(total_workers > 0 ? total_workers : 1);
    DEBUG_PRINT("[PTO Runtime] A2A3 simulation mode enabled: %d vector workers, %d cube workers\n", 
           num_vector_workers, num_cube_workers);
}

void pto_runtime_shutdown(PTORuntime* rt) {
    if (!rt) return;
    
    // Free tensor map entries
    pto_tensormap_clear(rt);
    
    // Destroy thread synchronization primitives
    pthread_mutex_destroy(&rt->queue_mutex);
    pthread_mutex_destroy(&rt->task_mutex);
    pthread_cond_destroy(&rt->queue_not_empty);
    pthread_cond_destroy(&rt->all_done);
    pthread_cond_destroy(&rt->vector_queue_not_empty);
    pthread_cond_destroy(&rt->cube_queue_not_empty);
    
    DEBUG_PRINT("[PTO Runtime] Shutdown (scheduled=%lld, completed=%lld)\n",
           (long long)rt->total_tasks_scheduled,
           (long long)rt->total_tasks_completed);
}

// =============================================================================
// TensorMap Implementation
// =============================================================================

uint32_t pto_tensormap_hash(TensorRegion* region) {
    // Simple hash combining pointer and offsets
    uint64_t ptr_val = (uint64_t)region->raw_tensor;
    uint32_t hash = (uint32_t)(ptr_val ^ (ptr_val >> 32));
    hash ^= (uint32_t)region->row_offset * 31;
    hash ^= (uint32_t)region->col_offset * 17;
    hash ^= (uint32_t)region->rows * 13;
    hash ^= (uint32_t)region->cols * 7;
    return hash % PTO_TENSORMAP_SIZE;
}

bool pto_region_match(TensorRegion* a, TensorRegion* b) {
    return a->raw_tensor == b->raw_tensor &&
           a->row_offset == b->row_offset &&
           a->col_offset == b->col_offset &&
           a->rows == b->rows &&
           a->cols == b->cols;
}

int32_t pto_tensormap_lookup(PTORuntime* rt, TensorRegion* region) {
    uint32_t hash = pto_tensormap_hash(region);
    TensorMapEntry* entry = rt->tensor_map[hash];
    
    while (entry) {
        if (pto_region_match(&entry->region, region)) {
            return entry->producer_id;
        }
        entry = entry->next;
    }
    
    return -1; // Not found
}

void pto_tensormap_insert(PTORuntime* rt, TensorRegion* region, int32_t task_id) {
    uint32_t hash = pto_tensormap_hash(region);
    
    // Check if entry already exists
    TensorMapEntry* entry = rt->tensor_map[hash];
    while (entry) {
        if (pto_region_match(&entry->region, region)) {
            // Update existing entry
            entry->producer_id = task_id;
            return;
        }
        entry = entry->next;
    }
    
    // Create new entry
    TensorMapEntry* new_entry = (TensorMapEntry*)malloc(sizeof(TensorMapEntry));
    if (!new_entry) {
        fprintf(stderr, "[PTO Runtime] ERROR: Failed to allocate TensorMapEntry\n");
        return;
    }
    
    new_entry->region = *region;
    new_entry->producer_id = task_id;
    new_entry->next = rt->tensor_map[hash];
    rt->tensor_map[hash] = new_entry;
}

void pto_tensormap_clear(PTORuntime* rt) {
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        TensorMapEntry* entry = rt->tensor_map[i];
        while (entry) {
            TensorMapEntry* next = entry->next;
            free(entry);
            entry = next;
        }
        rt->tensor_map[i] = NULL;
    }
}

// =============================================================================
// Ready Queue Implementation
// =============================================================================

static void ready_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime] ERROR: Ready queue overflow\n");
        return;
    }
    
    rt->ready_queue[rt->ready_tail] = task_id;
    rt->ready_tail = (rt->ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count++;
}

static int32_t ready_queue_pop(PTORuntime* rt) {
    if (rt->ready_count == 0) {
        return -1;
    }
    
    int32_t task_id = rt->ready_queue[rt->ready_head];
    rt->ready_head = (rt->ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count--;
    return task_id;
}

// =============================================================================
// Dual Ready Queue Implementation (for a2a3_sim mode)
// =============================================================================

static void vector_ready_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->vector_ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime] ERROR: Vector ready queue overflow\n");
        return;
    }
    
    rt->vector_ready_queue[rt->vector_ready_tail] = task_id;
    rt->vector_ready_tail = (rt->vector_ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count++;
}

static int32_t vector_ready_queue_pop(PTORuntime* rt) {
    if (rt->vector_ready_count == 0) {
        return -1;
    }
    
    int32_t task_id = rt->vector_ready_queue[rt->vector_ready_head];
    rt->vector_ready_head = (rt->vector_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count--;
    return task_id;
}

static void cube_ready_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->cube_ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime] ERROR: Cube ready queue overflow\n");
        return;
    }
    
    rt->cube_ready_queue[rt->cube_ready_tail] = task_id;
    rt->cube_ready_tail = (rt->cube_ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count++;
}

static int32_t cube_ready_queue_pop(PTORuntime* rt) {
    if (rt->cube_ready_count == 0) {
        return -1;
    }
    
    int32_t task_id = rt->cube_ready_queue[rt->cube_ready_head];
    rt->cube_ready_head = (rt->cube_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count--;
    return task_id;
}

// Push to appropriate queue based on task's is_cube flag
static void dual_ready_queue_push(PTORuntime* rt, int32_t task_id) {
    // Get is_cube from task or compact task
    bool is_cube = false;
    CompactTask* ct = &rt->compact_task[task_id];
    if (ct->template_ref != NULL) {
        is_cube = ct->template_ref->is_cube;
    } else {
        is_cube = rt->pend_task[task_id].is_cube;
    }
    
    if (is_cube) {
        cube_ready_queue_push(rt, task_id);
    } else {
        vector_ready_queue_push(rt, task_id);
    }
}

// =============================================================================
// Thread-safe Ready Queue Operations
// =============================================================================

static void ready_queue_push_threadsafe(PTORuntime* rt, int32_t task_id) {
    DEBUG_PRINT("[Queue] push_threadsafe: trying to lock for task %d\n", task_id);
    fflush(stdout);
    
    pthread_mutex_lock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] push_threadsafe: got lock for task %d, ready_count=%d\n", task_id, rt->ready_count);
    fflush(stdout);
    
    if (rt->ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime] ERROR: Ready queue overflow\n");
        pthread_mutex_unlock(&rt->queue_mutex);
        return;
    }
    
    rt->ready_queue[rt->ready_tail] = task_id;
    rt->ready_tail = (rt->ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count++;
    
    // Broadcast to wake up all waiting workers (more responsive than signal)
    pthread_cond_broadcast(&rt->queue_not_empty);
    
    pthread_mutex_unlock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] push_threadsafe: released lock, task %d queued\n", task_id);
    fflush(stdout);
}

int32_t pto_get_ready_task_blocking(PTORuntime* rt) {
    DEBUG_PRINT("[Queue] get_blocking: trying to lock\n");
    fflush(stdout);
    
    pthread_mutex_lock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] get_blocking: got lock, ready_count=%d, started=%d, threshold=%d\n", 
           rt->ready_count, rt->execution_started, rt->execution_task_threshold);
    fflush(stdout);
    
    // Determine if we can start execution:
    // - threshold == 0: must wait for execution_started (orchestration complete)
    // - threshold > 0: can start when active_task_count > threshold OR execution_started
    bool can_execute = rt->execution_started || 
                       (rt->execution_task_threshold > 0 && 
                        rt->total_tasks_scheduled > rt->execution_task_threshold);
    
    // Wait until: (can_execute AND task available) OR shutdown OR all done
    while ((rt->ready_count == 0 || !can_execute) && !rt->shutdown_requested) {
        // Check if all tasks are completed
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;  // All done
        }
        
        // Reduce debug spam - only print occasionally
        static __thread int wait_count = 0;
        if (++wait_count % 1000 == 1) {
            DEBUG_PRINT("[Queue] get_blocking: waiting (ready=%d, can_exec=%d, shutdown=%d, count=%d)\n", 
                   rt->ready_count, can_execute, rt->shutdown_requested, wait_count);
            fflush(stdout);
        }
        
        // Wait for signal (with short timeout for responsiveness)
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 100000;  // 100µs timeout
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&rt->queue_not_empty, &rt->queue_mutex, &timeout);
        
        // Re-check execution condition after wakeup
        can_execute = rt->execution_started || 
                      (rt->execution_task_threshold > 0 && 
                       rt->total_tasks_scheduled > rt->execution_task_threshold);
    }
    
    if (rt->shutdown_requested || rt->ready_count == 0) {
        pthread_mutex_unlock(&rt->queue_mutex);
        return -1;
    }
    
    int32_t task_id = rt->ready_queue[rt->ready_head];
    rt->ready_head = (rt->ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count--;
    
    pthread_mutex_unlock(&rt->queue_mutex);
    return task_id;
}

// =============================================================================
// Thread-safe Dual Queue Operations (for a2a3_sim mode)
// =============================================================================

static void dual_ready_queue_push_threadsafe(PTORuntime* rt, int32_t task_id) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    // Get is_cube from task or compact task
    bool is_cube = false;
    CompactTask* ct = &rt->compact_task[task_id];
    if (ct->template_ref != NULL) {
        is_cube = ct->template_ref->is_cube;
    } else {
        is_cube = rt->pend_task[task_id].is_cube;
    }
    
    if (is_cube) {
        if (rt->cube_ready_count >= PTO_MAX_READY_QUEUE) {
            fprintf(stderr, "[PTO Runtime] ERROR: Cube ready queue overflow\n");
            pthread_mutex_unlock(&rt->queue_mutex);
            return;
        }
        rt->cube_ready_queue[rt->cube_ready_tail] = task_id;
        rt->cube_ready_tail = (rt->cube_ready_tail + 1) % PTO_MAX_READY_QUEUE;
        rt->cube_ready_count++;
        pthread_cond_broadcast(&rt->cube_queue_not_empty);
        DEBUG_PRINT("[Queue] task %d pushed to cube queue (count=%d)\n", task_id, rt->cube_ready_count);
    } else {
        if (rt->vector_ready_count >= PTO_MAX_READY_QUEUE) {
            fprintf(stderr, "[PTO Runtime] ERROR: Vector ready queue overflow\n");
            pthread_mutex_unlock(&rt->queue_mutex);
            return;
        }
        rt->vector_ready_queue[rt->vector_ready_tail] = task_id;
        rt->vector_ready_tail = (rt->vector_ready_tail + 1) % PTO_MAX_READY_QUEUE;
        rt->vector_ready_count++;
        pthread_cond_broadcast(&rt->vector_queue_not_empty);
        DEBUG_PRINT("[Queue] task %d pushed to vector queue (count=%d)\n", task_id, rt->vector_ready_count);
    }
    
    pthread_mutex_unlock(&rt->queue_mutex);
}

int32_t pto_get_ready_task_vector(PTORuntime* rt) {
    return vector_ready_queue_pop(rt);
}

int32_t pto_get_ready_task_cube(PTORuntime* rt) {
    return cube_ready_queue_pop(rt);
}

int32_t pto_get_ready_task_vector_blocking(PTORuntime* rt) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    bool can_execute = rt->execution_started || 
                       (rt->execution_task_threshold > 0 && 
                        rt->total_tasks_scheduled > rt->execution_task_threshold);
    
    while ((rt->vector_ready_count == 0 || !can_execute) && !rt->shutdown_requested) {
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;
        }
        
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 100000;
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&rt->vector_queue_not_empty, &rt->queue_mutex, &timeout);
        
        can_execute = rt->execution_started || 
                      (rt->execution_task_threshold > 0 && 
                       rt->total_tasks_scheduled > rt->execution_task_threshold);
    }
    
    if (rt->shutdown_requested || rt->vector_ready_count == 0) {
        pthread_mutex_unlock(&rt->queue_mutex);
        return -1;
    }
    
    int32_t task_id = rt->vector_ready_queue[rt->vector_ready_head];
    rt->vector_ready_head = (rt->vector_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->vector_ready_count--;
    
    pthread_mutex_unlock(&rt->queue_mutex);
    return task_id;
}

int32_t pto_get_ready_task_cube_blocking(PTORuntime* rt) {
    pthread_mutex_lock(&rt->queue_mutex);
    
    bool can_execute = rt->execution_started || 
                       (rt->execution_task_threshold > 0 && 
                        rt->total_tasks_scheduled > rt->execution_task_threshold);
    
    while ((rt->cube_ready_count == 0 || !can_execute) && !rt->shutdown_requested) {
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;
        }
        
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 100000;
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&rt->cube_queue_not_empty, &rt->queue_mutex, &timeout);
        
        can_execute = rt->execution_started || 
                      (rt->execution_task_threshold > 0 && 
                       rt->total_tasks_scheduled > rt->execution_task_threshold);
    }
    
    if (rt->shutdown_requested || rt->cube_ready_count == 0) {
        pthread_mutex_unlock(&rt->queue_mutex);
        return -1;
    }
    
    int32_t task_id = rt->cube_ready_queue[rt->cube_ready_head];
    rt->cube_ready_head = (rt->cube_ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->cube_ready_count--;
    
    pthread_mutex_unlock(&rt->queue_mutex);
    return task_id;
}

// =============================================================================
// Task Management
// =============================================================================

int32_t pto_task_alloc_impl(PTORuntime* rt, const char* func_name, void* func_ptr,
                            int32_t buffer_bytes, int32_t reuse_bytes, bool is_cube) {
    if (rt->next_task_id >= PTO_MAX_TASKS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Task table full\n");
        return -1;
    }
    
    int32_t task_id = rt->next_task_id++;
    PendingTask* task = &rt->pend_task[task_id];
    
    // Initialize task
    task->task_id = task_id;
    task->func_name = func_name;
    task->func_ptr = func_ptr;
    task->cycle_func = NULL;  // Set via pto_task_set_cycle_func if needed
    task->num_args = 0;
    task->buffer_size_bytes = buffer_bytes;
    task->buffer_size_with_reuse = reuse_bytes;
    task->fanin = 0;
    task->fanout_count = 0;
    task->is_active = true;
    task->is_complete = false;
    task->is_cube = is_cube;
    task->earliest_start_cycle = 0;  // No deps yet, can start immediately
    task->end_cycle = 0;
    
    // Clear fanout list
    memset(task->fanout, 0, sizeof(task->fanout));
    
    rt->active_task_count++;
    rt->total_tasks_scheduled++;
    
    DEBUG_PRINT("[PTO Runtime] Allocated task %d: %s (buf=%d B, reuse=%d B, is_cube=%d)\n", 
           task_id, func_name, buffer_bytes, reuse_bytes, is_cube);
    
    return task_id;
}

void pto_task_set_cycle_func(PTORuntime* rt, int32_t task_id, CycleCostFunc cycle_func) {
    if (!rt || task_id < 0 || task_id >= rt->next_task_id) return;
    rt->pend_task[task_id].cycle_func = cycle_func;
}

void pto_task_add_input(PTORuntime* rt, int32_t task_id,
                        void* tensor, int64_t row_off, int64_t col_off,
                        int64_t rows, int64_t cols) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[task_id];
    
    if (task->num_args >= PTO_MAX_ARGS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Too many arguments for task %d\n", task_id);
        return;
    }
    
    // Create tensor region
    TensorRegion region = {
        .raw_tensor = tensor,
        .row_offset = row_off,
        .col_offset = col_off,
        .rows = rows,
        .cols = cols
    };
    
    // Add argument
    TaskArg* arg = &task->args[task->num_args++];
    arg->region = region;
    arg->is_output = false;
    
    // Look up producer in TensorMap
    int32_t producer_id = pto_tensormap_lookup(rt, &region);
    
    if (producer_id >= 0 && producer_id != task_id) {
        // Found producer - add dependency (needs mutex for pipelined execution)
        pthread_mutex_lock(&rt->task_mutex);
        
        PendingTask* producer = &rt->pend_task[producer_id];
        
        // Check if producer is already complete (pipelined execution race condition)
        if (producer->is_complete) {
            // Producer already done - no need to add dependency
            pthread_mutex_unlock(&rt->task_mutex);
            DEBUG_PRINT("[PTO Runtime] Task %d: producer %d already complete, no dependency added\n",
                   task_id, producer_id);
        } else {
            // Add current task to producer's fanout
            if (producer->fanout_count < PTO_MAX_FANOUT) {
                producer->fanout[producer->fanout_count++] = task_id;
            } else {
                fprintf(stderr, "[PTO Runtime] WARNING: Fanout overflow for task %d\n", producer_id);
            }
            
            // Increment fanin (dependency count)
            task->fanin++;
            
            pthread_mutex_unlock(&rt->task_mutex);
            DEBUG_PRINT("[PTO Runtime] Task %d depends on task %d (tensor=%p, offset=[%lld,%lld])\n",
                   task_id, producer_id, tensor, (long long)row_off, (long long)col_off);
        }
    } else {
        DEBUG_PRINT("[PTO Runtime] Task %d input (tensor=%p, offset=[%lld,%lld]) - no producer\n",
               task_id, tensor, (long long)row_off, (long long)col_off);
    }
}

void pto_task_add_output(PTORuntime* rt, int32_t task_id,
                         void* tensor, int64_t row_off, int64_t col_off,
                         int64_t rows, int64_t cols) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[task_id];
    
    if (task->num_args >= PTO_MAX_ARGS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Too many arguments for task %d\n", task_id);
        return;
    }
    
    // Create tensor region
    TensorRegion region = {
        .raw_tensor = tensor,
        .row_offset = row_off,
        .col_offset = col_off,
        .rows = rows,
        .cols = cols
    };
    
    // Add argument
    TaskArg* arg = &task->args[task->num_args++];
    arg->region = region;
    arg->is_output = true;
    
    // Register in TensorMap (this task produces this region)
    pto_tensormap_insert(rt, &region, task_id);
    
    DEBUG_PRINT("[PTO Runtime] Task %d output (tensor=%p, offset=[%lld,%lld], shape=[%lld,%lld])\n",
           task_id, tensor, (long long)row_off, (long long)col_off,
           (long long)rows, (long long)cols);
}

void pto_task_submit(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[task_id];
    
    DEBUG_PRINT("[PTO Runtime] Submitted task %d: %s (fanin=%d, fanout=%d, is_cube=%d)\n",
           task_id, task->func_name, task->fanin, task->fanout_count, task->is_cube);
    
    // If no dependencies, add directly to ready queue (thread-safe)
    // This allows workers to start executing immediately
    if (task->fanin == 0) {
        if (rt->dual_queue_mode) {
            dual_ready_queue_push_threadsafe(rt, task_id);
        } else {
            ready_queue_push_threadsafe(rt, task_id);
        }
        DEBUG_PRINT("[PTO Runtime] Task %d is ready (no dependencies)\n", task_id);
    }
    // Tasks with fanin > 0 stay in pend_task until dependencies complete
}

void pto_task_complete(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    // Check if this is a compact task (replay) or a regular task
    CompactTask* ct = &rt->compact_task[task_id];
    bool is_compact = (ct->template_ref != NULL);
    
    if (is_compact) {
        ct->is_complete = true;
    } else {
        rt->pend_task[task_id].is_complete = true;
    }
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    // Get fanout info and end_cycle from appropriate source
    int32_t fanout_count;
    int32_t* fanout;
    const char* func_name;
    int64_t producer_end_cycle;
    
    if (is_compact) {
        fanout_count = ct->template_ref->fanout_count;
        fanout = ct->template_ref->fanout;  // Relative offsets
        func_name = ct->template_ref->func_name;
        producer_end_cycle = ct->earliest_start_cycle;  // Will be updated in execute_task
    } else {
        fanout_count = rt->pend_task[task_id].fanout_count;
        fanout = rt->pend_task[task_id].fanout;  // Absolute IDs
        func_name = rt->pend_task[task_id].func_name;
        producer_end_cycle = rt->pend_task[task_id].end_cycle;
    }
    
    DEBUG_PRINT("[PTO Runtime] Completed task %d: %s\n", task_id, func_name);
    
    // Update dependent tasks
    for (int i = 0; i < fanout_count; i++) {
        // Compute dependent task ID
        int32_t dep_id = is_compact ? (task_id + fanout[i]) : fanout[i];
        
        // Check if dependent is compact or regular
        CompactTask* dep_ct = &rt->compact_task[dep_id];
        bool dep_is_compact = (dep_ct->template_ref != NULL);
        
        // Update earliest_start_cycle of dependent task
        // It can't start until after this producer finishes
        if (dep_is_compact) {
            if (producer_end_cycle > dep_ct->earliest_start_cycle) {
                dep_ct->earliest_start_cycle = producer_end_cycle;
            }
        } else {
            if (producer_end_cycle > rt->pend_task[dep_id].earliest_start_cycle) {
                rt->pend_task[dep_id].earliest_start_cycle = producer_end_cycle;
            }
        }
        
        bool is_ready = false;
        if (dep_is_compact) {
            // Compact task: INCREMENT resolved_fanin, compare with template->internal_fanin
            // We use internal_fanin (deps within fragment) for replay tasks
            dep_ct->resolved_fanin++;
            is_ready = (dep_ct->resolved_fanin >= dep_ct->template_ref->internal_fanin);
            DEBUG_PRINT("[PTO Runtime] Task %d resolved_fanin=%d/%d\n", 
                   dep_id, dep_ct->resolved_fanin, dep_ct->template_ref->internal_fanin);
        } else {
            // Regular task: DECREMENT fanin
            rt->pend_task[dep_id].fanin--;
            is_ready = (rt->pend_task[dep_id].fanin == 0);
            DEBUG_PRINT("[PTO Runtime] Task %d fanin decremented to %d\n", 
                   dep_id, rt->pend_task[dep_id].fanin);
        }
        
        // Check completion status
        bool dep_complete = dep_is_compact ? dep_ct->is_complete : rt->pend_task[dep_id].is_complete;
        
        if (is_ready && !dep_complete) {
            if (rt->dual_queue_mode) {
                dual_ready_queue_push(rt, dep_id);
            } else {
                ready_queue_push(rt, dep_id);
            }
            DEBUG_PRINT("[PTO Runtime] Task %d is now ready\n", dep_id);
        }
    }
}

void pto_task_complete_threadsafe(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    pthread_mutex_lock(&rt->task_mutex);
    
    // Check if this is a compact task (replay) or a regular task
    CompactTask* ct = &rt->compact_task[task_id];
    bool is_compact = (ct->template_ref != NULL);
    
    if (is_compact) {
        ct->is_complete = true;
    } else {
        rt->pend_task[task_id].is_complete = true;
    }
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    // Get fanout info from appropriate source
    int32_t fanout_count;
    int32_t* fanout;
    const char* func_name;
    
    if (is_compact) {
        fanout_count = ct->template_ref->fanout_count;
        fanout = ct->template_ref->fanout;  // Relative offsets
        func_name = ct->template_ref->func_name;
    } else {
        fanout_count = rt->pend_task[task_id].fanout_count;
        fanout = rt->pend_task[task_id].fanout;  // Absolute IDs
        func_name = rt->pend_task[task_id].func_name;
    }
    
    DEBUG_PRINT("[PTO Runtime] Completed task %d: %s (completed=%lld/%lld)\n", 
           task_id, func_name, 
           (long long)rt->total_tasks_completed, 
           (long long)rt->total_tasks_scheduled);
    
    // Collect tasks that become ready
    int32_t newly_ready[PTO_MAX_FANOUT];
    int32_t newly_ready_count = 0;
    
    for (int i = 0; i < fanout_count; i++) {
        // Compute dependent task ID
        int32_t dep_id = is_compact ? (task_id + fanout[i]) : fanout[i];
        
        // Check if dependent is compact or regular
        CompactTask* dep_ct = &rt->compact_task[dep_id];
        bool dep_is_compact = (dep_ct->template_ref != NULL);
        
        bool is_ready = false;
        bool dep_complete = false;
        
        if (dep_is_compact) {
            dep_ct->resolved_fanin++;
            is_ready = (dep_ct->resolved_fanin >= dep_ct->template_ref->internal_fanin);
            dep_complete = dep_ct->is_complete;
            DEBUG_PRINT("[PTO Runtime] Task %d resolved_fanin=%d/%d\n", 
                   dep_id, dep_ct->resolved_fanin, dep_ct->template_ref->internal_fanin);
        } else {
            rt->pend_task[dep_id].fanin--;
            is_ready = (rt->pend_task[dep_id].fanin == 0);
            dep_complete = rt->pend_task[dep_id].is_complete;
            DEBUG_PRINT("[PTO Runtime] Task %d fanin decremented to %d\n", 
                   dep_id, rt->pend_task[dep_id].fanin);
        }
        
        if (is_ready && !dep_complete) {
            newly_ready[newly_ready_count++] = dep_id;
        }
    }
    
    // Check if all tasks completed
    bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
    
    pthread_mutex_unlock(&rt->task_mutex);
    
    // Add newly ready tasks to queue (outside task_mutex to avoid deadlock)
    for (int i = 0; i < newly_ready_count; i++) {
        if (rt->dual_queue_mode) {
            dual_ready_queue_push_threadsafe(rt, newly_ready[i]);
        } else {
            ready_queue_push_threadsafe(rt, newly_ready[i]);
        }
        DEBUG_PRINT("[PTO Runtime] Task %d is now ready\n", newly_ready[i]);
    }
    
    // Signal if all tasks are done
    if (all_done) {
        pthread_mutex_lock(&rt->queue_mutex);
        pthread_cond_broadcast(&rt->all_done);
        pthread_cond_broadcast(&rt->queue_not_empty);
        pthread_cond_broadcast(&rt->vector_queue_not_empty);
        pthread_cond_broadcast(&rt->cube_queue_not_empty);
        pthread_mutex_unlock(&rt->queue_mutex);
    }
}

int32_t pto_get_ready_task(PTORuntime* rt) {
    return ready_queue_pop(rt);
}

// =============================================================================
// Execution
// =============================================================================

// Generic function pointer type for InCore functions
typedef void (*InCoreFuncPtr)(void);

void pto_execute_all(PTORuntime* rt) {
    printf("\n[PTO Runtime] ======== Executing all tasks ========\n");
    
    while (rt->ready_count > 0 || rt->active_task_count > (int32_t)rt->total_tasks_completed) {
        int32_t task_id = pto_get_ready_task(rt);
        
        if (task_id < 0) {
            // No ready tasks - check for deadlock
            if (rt->active_task_count > (int32_t)rt->total_tasks_completed) {
                fprintf(stderr, "[PTO Runtime] WARNING: No ready tasks but %d tasks pending - possible deadlock\n",
                        rt->active_task_count - (int32_t)rt->total_tasks_completed);
                break;
            }
            continue;
        }
        
        PendingTask* task = &rt->pend_task[task_id];
        
        DEBUG_PRINT("[PTO Runtime] Executing task %d: %s\n", task_id, task->func_name);
        
        // Execute the task (in a real implementation, this would call the actual function)
        if (task->func_ptr) {
            // For now, just call the function pointer
            // In a real implementation, we would pass the arguments
            InCoreFuncPtr func __attribute__((unused)) = (InCoreFuncPtr)task->func_ptr;
            // Note: actual argument passing would require more sophisticated handling
            // func();  // Uncomment when function signatures are properly handled
            DEBUG_PRINT("[PTO Runtime] (Simulated execution of %s)\n", task->func_name);
        }
        
        // Mark complete and update dependencies
        pto_task_complete(rt, task_id);
    }
    
    DEBUG_PRINT("[PTO Runtime] ======== Execution complete ========\n\n");
}

void pto_runtime_stats(PTORuntime* rt) {
    printf("\n[PTO Runtime Statistics]\n");
    printf("  Total tasks scheduled: %lld\n", (long long)rt->total_tasks_scheduled);
    printf("  Total tasks completed: %lld\n", (long long)rt->total_tasks_completed);
    printf("  Active tasks:          %d\n", rt->active_task_count);
    printf("  Ready queue size:      %d\n", rt->ready_count);
    printf("\n");
}

// =============================================================================
// Dump Function - Export Runtime State to Text File
// =============================================================================

int pto_runtime_dump(PTORuntime* rt, const char* filename) {
    if (!rt || !filename) return -1;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "[PTO Runtime] ERROR: Cannot open file %s for writing\n", filename);
        return -1;
    }
    
    // Header
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "PTO RUNTIME DUMP\n");
    fprintf(fp, "================================================================================\n\n");
    
    // Summary statistics
    fprintf(fp, "SUMMARY\n");
    fprintf(fp, "--------------------------------------------------------------------------------\n");
    fprintf(fp, "  Total tasks scheduled:  %lld\n", (long long)rt->total_tasks_scheduled);
    fprintf(fp, "  Total tasks completed:  %lld\n", (long long)rt->total_tasks_completed);
    fprintf(fp, "  Active task count:      %d\n", rt->active_task_count);
    fprintf(fp, "  Next task ID:           %d\n", rt->next_task_id);
    fprintf(fp, "  Ready queue size:       %d\n", rt->ready_count);
    fprintf(fp, "\n");
    
    // Task Table
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "TASK TABLE (pend_task)\n");
    fprintf(fp, "================================================================================\n\n");
    
    for (int32_t i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        
        fprintf(fp, "--------------------------------------------------------------------------------\n");
        fprintf(fp, "TASK %d\n", task->task_id);
        fprintf(fp, "--------------------------------------------------------------------------------\n");
        fprintf(fp, "  Function:     %s\n", task->func_name ? task->func_name : "(null)");
        fprintf(fp, "  Func Ptr:     %p\n", task->func_ptr);
        fprintf(fp, "  Is Active:    %s\n", task->is_active ? "true" : "false");
        fprintf(fp, "  Is Complete:  %s\n", task->is_complete ? "true" : "false");
        fprintf(fp, "\n");
        
        // Buffer size estimation
        fprintf(fp, "  BUFFER SIZE (InCore Tile Buffers)\n");
        fprintf(fp, "  ----------------------------------\n");
        fprintf(fp, "    without_reuse = %d bytes (%.2f KB)\n", 
                task->buffer_size_bytes, task->buffer_size_bytes / 1024.0);
        fprintf(fp, "    with_reuse    = %d bytes (%.2f KB)\n", 
                task->buffer_size_with_reuse, task->buffer_size_with_reuse / 1024.0);
        if (task->buffer_size_bytes > 0) {
            int savings = task->buffer_size_bytes - task->buffer_size_with_reuse;
            float pct = 100.0 * savings / task->buffer_size_bytes;
            fprintf(fp, "    savings       = %d bytes (%.1f%%)\n", savings, pct);
        }
        fprintf(fp, "\n");
        
        // Fanin counter
        fprintf(fp, "  FANIN COUNTER\n");
        fprintf(fp, "  -------------\n");
        fprintf(fp, "    fanin = %d\n", task->fanin);
        fprintf(fp, "\n");
        
        // Fanout list
        fprintf(fp, "  FANOUT LIST (consumers that depend on this task)\n");
        fprintf(fp, "  ------------------------------------------------\n");
        fprintf(fp, "    fanout_count = %d\n", task->fanout_count);
        if (task->fanout_count > 0) {
            fprintf(fp, "    fanout[] = [");
            for (int j = 0; j < task->fanout_count; j++) {
                fprintf(fp, "%d", task->fanout[j]);
                if (j < task->fanout_count - 1) fprintf(fp, ", ");
            }
            fprintf(fp, "]\n");
            
            // Detailed fanout info
            fprintf(fp, "    Consumers:\n");
            for (int j = 0; j < task->fanout_count; j++) {
                int32_t consumer_id = task->fanout[j];
                PendingTask* consumer = &rt->pend_task[consumer_id];
                fprintf(fp, "      -> Task %d (%s)\n", consumer_id, 
                        consumer->func_name ? consumer->func_name : "(null)");
            }
        } else {
            fprintf(fp, "    fanout[] = [] (no consumers)\n");
        }
        fprintf(fp, "\n");
        
        // Arguments
        fprintf(fp, "  ARGUMENTS (num_args = %d)\n", task->num_args);
        fprintf(fp, "  -------------------------\n");
        for (int j = 0; j < task->num_args; j++) {
            TaskArg* arg = &task->args[j];
            fprintf(fp, "    [%d] %s:\n", j, arg->is_output ? "OUTPUT" : "INPUT");
            fprintf(fp, "        tensor:     %p\n", arg->region.raw_tensor);
            fprintf(fp, "        row_offset: %lld\n", (long long)arg->region.row_offset);
            fprintf(fp, "        col_offset: %lld\n", (long long)arg->region.col_offset);
            fprintf(fp, "        rows:       %lld\n", (long long)arg->region.rows);
            fprintf(fp, "        cols:       %lld\n", (long long)arg->region.cols);
        }
        fprintf(fp, "\n");
    }
    
    // Ready Queue
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "READY QUEUE\n");
    fprintf(fp, "================================================================================\n\n");
    fprintf(fp, "  Head:  %d\n", rt->ready_head);
    fprintf(fp, "  Tail:  %d\n", rt->ready_tail);
    fprintf(fp, "  Count: %d\n", rt->ready_count);
    if (rt->ready_count > 0) {
        fprintf(fp, "  Queue: [");
        int idx = rt->ready_head;
        for (int i = 0; i < rt->ready_count; i++) {
            fprintf(fp, "%d", rt->ready_queue[idx]);
            if (i < rt->ready_count - 1) fprintf(fp, ", ");
            idx = (idx + 1) % PTO_MAX_READY_QUEUE;
        }
        fprintf(fp, "]\n");
    } else {
        fprintf(fp, "  Queue: [] (empty)\n");
    }
    fprintf(fp, "\n");
    
    // TensorMap (active entries)
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "TENSOR MAP (non-empty buckets)\n");
    fprintf(fp, "================================================================================\n\n");
    int tensor_count = 0;
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        TensorMapEntry* entry = rt->tensor_map[i];
        while (entry) {
            fprintf(fp, "  [bucket %d] tensor=%p, offset=[%lld,%lld], shape=[%lld,%lld] -> producer: Task %d\n",
                    i,
                    entry->region.raw_tensor,
                    (long long)entry->region.row_offset,
                    (long long)entry->region.col_offset,
                    (long long)entry->region.rows,
                    (long long)entry->region.cols,
                    entry->producer_id);
            tensor_count++;
            entry = entry->next;
        }
    }
    if (tensor_count == 0) {
        fprintf(fp, "  (empty)\n");
    }
    fprintf(fp, "\n  Total tensor entries: %d\n", tensor_count);
    fprintf(fp, "\n");
    
    // Dependency Graph (ASCII representation)
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "DEPENDENCY GRAPH (Producer -> Consumer)\n");
    fprintf(fp, "================================================================================\n\n");
    for (int32_t i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        if (!task->is_active) continue;
        
        // Status indicator
        const char* status = task->is_complete ? "[DONE]" : 
                            (task->fanin == 0 ? "[READY]" : "[WAIT]");
        
        fprintf(fp, "  Task %d (%s) %s\n", i, 
                task->func_name ? task->func_name : "?", status);
        
        if (task->fanout_count > 0) {
            for (int j = 0; j < task->fanout_count; j++) {
                int32_t consumer_id = task->fanout[j];
                PendingTask* consumer = &rt->pend_task[consumer_id];
                fprintf(fp, "    └──> Task %d (%s)\n", consumer_id,
                        consumer->func_name ? consumer->func_name : "?");
            }
        }
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "END OF DUMP\n");
    fprintf(fp, "================================================================================\n");
    
    fclose(fp);
    DEBUG_PRINT("[PTO Runtime] Dumped runtime state to %s\n", filename);
    return 0;
}

int pto_runtime_dump_stdout(PTORuntime* rt) {
    if (!rt) return -1;
    
    // Reuse dump logic but write to stdout
    printf("================================================================================\n");
    printf("PTO RUNTIME DUMP\n");
    printf("================================================================================\n\n");
    
    printf("SUMMARY\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  Total tasks scheduled:  %lld\n", (long long)rt->total_tasks_scheduled);
    printf("  Total tasks completed:  %lld\n", (long long)rt->total_tasks_completed);
    printf("  Active task count:      %d\n", rt->active_task_count);
    printf("  Next task ID:           %d\n", rt->next_task_id);
    printf("  Ready queue size:       %d\n", rt->ready_count);
    printf("\n");
    
    printf("TASK TABLE\n");
    printf("--------------------------------------------------------------------------------\n");
    for (int32_t i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        const char* status = task->is_complete ? "DONE" : 
                            (task->fanin == 0 ? "READY" : "WAIT");
        
        printf("  Task %d: %-20s [%s] fanin=%d buf=%.1fKB fanout=[",
               i, task->func_name ? task->func_name : "?", status, task->fanin,
               task->buffer_size_with_reuse / 1024.0);
        for (int j = 0; j < task->fanout_count; j++) {
            printf("%d", task->fanout[j]);
            if (j < task->fanout_count - 1) printf(",");
        }
        printf("]\n");
    }
    printf("\n");
    
    return 0;
}

// =============================================================================
// Record & Replay Implementation (using compact_task for cache efficiency)
// =============================================================================

RecordedFragment* pto_fragment_record(PTORuntime* rt, int32_t start_id, int32_t end_id,
                                       const char* name) {
    if (!rt || start_id < 0 || end_id <= start_id || end_id > rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid fragment range [%d, %d)\n", start_id, end_id);
        return NULL;
    }
    
    int32_t task_count = end_id - start_id;
    
    // Allocate fragment structure
    RecordedFragment* frag = (RecordedFragment*)malloc(sizeof(RecordedFragment));
    if (!frag) return NULL;
    
    frag->tasks = (RecordedTask*)malloc(task_count * sizeof(RecordedTask));
    if (!frag->tasks) {
        free(frag);
        return NULL;
    }
    
    // Count outputs first
    int32_t output_count = 0;
    for (int32_t i = start_id; i < end_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        for (int j = 0; j < task->num_args; j++) {
            if (task->args[j].is_output) {
                output_count++;
            }
        }
    }
    
    frag->outputs = (RecordedOutput*)malloc(output_count * sizeof(RecordedOutput));
    if (!frag->outputs && output_count > 0) {
        free(frag->tasks);
        free(frag);
        return NULL;
    }
    
    frag->task_count = task_count;
    frag->output_count = 0;
    frag->fragment_name = name;
    frag->checksum = 0;
    
    // Record each task - convert absolute fanout to relative offsets
    for (int32_t i = 0; i < task_count; i++) {
        PendingTask* src = &rt->pend_task[start_id + i];
        RecordedTask* dst = &frag->tasks[i];
        
        dst->func_name = src->func_name;
        dst->func_ptr = src->func_ptr;
        dst->cycle_func = src->cycle_func;  // For simulation mode
        dst->buffer_size_bytes = src->buffer_size_bytes;
        dst->buffer_size_with_reuse = src->buffer_size_with_reuse;
        dst->fanin = src->fanin;
        dst->internal_fanin = 0;  // Will be computed in second pass
        dst->num_args = src->num_args;
        dst->is_cube = src->is_cube;  // Copy cube flag for a2a3_sim mode
        
        // Convert fanout to relative offsets for position-independence
        // Only include fanouts that point to tasks within the fragment
        dst->fanout_count = 0;
        for (int j = 0; j < src->fanout_count; j++) {
            int32_t abs_target = src->fanout[j];
            // Check if target is within the fragment
            if (abs_target >= start_id && abs_target < start_id + task_count) {
                dst->fanout[dst->fanout_count++] = abs_target - (start_id + i);  // relative offset
            }
        }
        
        // Copy arguments
        memcpy(dst->args, src->args, src->num_args * sizeof(TaskArg));
        
        // Record outputs for TensorMap replay
        for (int j = 0; j < src->num_args; j++) {
            if (src->args[j].is_output) {
                RecordedOutput* out = &frag->outputs[frag->output_count++];
                out->region = src->args[j].region;
                out->relative_producer = i;
            }
        }
        
        // Simple checksum
        for (int j = 0; j < dst->fanout_count; j++) {
            frag->checksum ^= dst->fanout[j] * (i + 1);
        }
    }
    
    // Second pass: compute internal_fanin by counting how many times each task
    // is referenced by internal fanouts
    for (int32_t i = 0; i < task_count; i++) {
        RecordedTask* src_task = &frag->tasks[i];
        for (int j = 0; j < src_task->fanout_count; j++) {
            int32_t rel_target = src_task->fanout[j];
            int32_t abs_target = i + rel_target;
            if (abs_target >= 0 && abs_target < task_count) {
                frag->tasks[abs_target].internal_fanin++;
            }
        }
    }
    
    DEBUG_PRINT("[PTO Runtime] Recorded fragment '%s': %d tasks, %d outputs\n",
           name ? name : "(unnamed)", task_count, frag->output_count);
    
    return frag;
}

void pto_fragment_free(RecordedFragment* fragment) {
    if (!fragment) return;
    if (fragment->tasks) free(fragment->tasks);
    if (fragment->outputs) free(fragment->outputs);
    free(fragment);
}

size_t pto_fragment_size(RecordedFragment* fragment) {
    if (!fragment) return 0;
    size_t size = sizeof(RecordedFragment);
    size += fragment->task_count * sizeof(RecordedTask);
    size += fragment->output_count * sizeof(RecordedOutput);
    return size;
}

// =============================================================================
// Loop Replay Implementation (using compact_task for 120x cache improvement)
// =============================================================================

// =============================================================================
// Global flag for record/replay optimization
// =============================================================================

int pto_record_replay_enabled = 1;  // Enabled by default

void pto_set_record_replay(int enabled) {
    pto_record_replay_enabled = enabled;
    DEBUG_PRINT("[Loop Replay] Record/replay %s\n", enabled ? "ENABLED" : "DISABLED");
}

void pto_loop_init(LoopReplayCtx* ctx, const char* name, int32_t stride, OffsetMode mode) {
    if (!ctx) return;
    ctx->fragment = NULL;
    ctx->record_start = -1;
    ctx->base_offset = 0;
    ctx->stride = stride;
    ctx->offset_mode = mode;
    ctx->loop_name = name;
}

bool pto_loop_should_record(PTORuntime* rt, LoopReplayCtx* ctx, int32_t loop_idx) {
    if (!rt || !ctx) return true;
    
    // If replay is disabled, always return true (direct task creation)
    if (!pto_record_replay_enabled) {
        return true;
    }
    
    if (ctx->fragment == NULL) {
        // First iteration: need to record
        ctx->record_start = rt->next_task_id;
        ctx->base_offset = loop_idx * ctx->stride;
        DEBUG_PRINT("[Loop Replay] Recording '%s' at iteration %d (base_offset=%d)\n",
               ctx->loop_name ? ctx->loop_name : "?", loop_idx, ctx->base_offset);
        return true;
    }
    
    return false;  // Have fragment, can replay
}

void pto_loop_finish_record(PTORuntime* rt, LoopReplayCtx* ctx) {
    if (!rt || !ctx || ctx->record_start < 0) return;
    
    // If replay is disabled, don't record (just reset state)
    if (!pto_record_replay_enabled) {
        ctx->record_start = -1;
        return;
    }
    
    int32_t end_id = rt->next_task_id;
    ctx->fragment = pto_fragment_record(rt, ctx->record_start, end_id, ctx->loop_name);
    
    DEBUG_PRINT("[Loop Replay] Recorded '%s': %d tasks (IDs %d-%d)\n",
           ctx->loop_name ? ctx->loop_name : "?",
           ctx->fragment ? ctx->fragment->task_count : 0,
           ctx->record_start, end_id - 1);
    
    ctx->record_start = -1;
}

void pto_loop_replay(PTORuntime* rt, LoopReplayCtx* ctx, int32_t loop_idx) {
    if (!rt || !ctx || !ctx->fragment) return;
    
    RecordedFragment* frag = ctx->fragment;
    int32_t offset_delta = (loop_idx * ctx->stride) - ctx->base_offset;
    
    if (rt->next_task_id + frag->task_count > PTO_MAX_TASKS) {
        fprintf(stderr, "[Loop Replay] ERROR: Task table overflow\n");
        return;
    }
    
    int32_t base_id = rt->next_task_id;
    
    // =========================================================================
    // CACHE-OPTIMIZED REPLAY: Use compact_task array (24 bytes per entry)
    // This is 120x more cache-efficient than using pend_task (2.8KB per entry)
    // =========================================================================
    for (int32_t i = 0; i < frag->task_count; i++) {
        RecordedTask* src = &frag->tasks[i];
        int32_t task_id = base_id + i;
        
        // Write only to compact_task array
        CompactTask* ct = &rt->compact_task[task_id];
        ct->template_ref = src;           // Point to immutable template
        ct->resolved_fanin = 0;           // Start at 0, will be incremented by internal producers
        ct->offset_delta = offset_delta;  // Row offset for this replay
        ct->earliest_start_cycle = 0;     // Will be updated by producers
        ct->is_complete = false;
        ct->is_active = true;
        
        rt->active_task_count++;
        rt->total_tasks_scheduled++;
    }
    
    // Re-register outputs in TensorMap with adjusted offsets
    for (int32_t i = 0; i < frag->output_count; i++) {
        RecordedOutput* out = &frag->outputs[i];
        TensorRegion adjusted_region = out->region;
        
        switch (ctx->offset_mode) {
            case OFFSET_ROW:
                adjusted_region.row_offset += offset_delta;
                break;
            case OFFSET_COL:
                adjusted_region.col_offset += offset_delta;
                break;
            case OFFSET_ROW_COL:
                adjusted_region.row_offset += offset_delta;
                adjusted_region.col_offset += offset_delta;
                break;
            default:
                break;
        }
        
        int32_t abs_producer = base_id + out->relative_producer;
        pto_tensormap_insert(rt, &adjusted_region, abs_producer);
    }
    
    rt->next_task_id = base_id + frag->task_count;
    
    // Submit ready tasks (internal_fanin == 0)
    // Note: We use internal_fanin (deps within fragment) instead of fanin (total deps)
    // because external deps from the original iteration don't apply to replay iterations
    for (int32_t i = 0; i < frag->task_count; i++) {
        RecordedTask* tmpl = &frag->tasks[i];
        if (tmpl->internal_fanin == 0) {
            if (rt->dual_queue_mode) {
                dual_ready_queue_push_threadsafe(rt, base_id + i);
            } else {
                ready_queue_push_threadsafe(rt, base_id + i);
            }
        }
    }
    
    DEBUG_PRINT("[Loop Replay] Replayed '%s' at iteration %d (offset_delta=%d, base_id=%d)\n",
           ctx->loop_name ? ctx->loop_name : "?", loop_idx, offset_delta, base_id);
}

void pto_loop_cleanup(LoopReplayCtx* ctx) {
    if (!ctx) return;
    if (ctx->fragment) {
        pto_fragment_free(ctx->fragment);
        ctx->fragment = NULL;
    }
}

/**
 * Validate that task arguments are compatible with the recorded template.
 * This function compares what would be created at loop_idx against the template.
 * 
 * Replay is CORRECT if for each argument:
 *   1. raw_tensor pointer is identical
 *   2. col_offset is identical  
 *   3. rows, cols are identical
 *   4. row_offset differs by exactly (loop_idx * stride - base_offset)
 * 
 * Returns true if compatible, false otherwise (with error message)
 */
bool pto_loop_validate_task(LoopReplayCtx* ctx, int32_t task_idx_in_fragment,
                            int32_t loop_idx, TaskArg* args, int32_t num_args) {
    if (!ctx || !ctx->fragment || task_idx_in_fragment < 0 || 
        task_idx_in_fragment >= ctx->fragment->task_count) {
        fprintf(stderr, "[Replay Validate] ERROR: Invalid context or task index\n");
        return false;
    }
    
    RecordedTask* tmpl = &ctx->fragment->tasks[task_idx_in_fragment];
    int32_t expected_offset_delta = (loop_idx * ctx->stride) - ctx->base_offset;
    
    // Check argument count matches
    if (num_args != tmpl->num_args) {
        fprintf(stderr, "[Replay Validate] ERROR: Task %d arg count mismatch: "
                "template=%d, actual=%d\n", 
                task_idx_in_fragment, tmpl->num_args, num_args);
        return false;
    }
    
    // Check each argument
    bool valid = true;
    for (int i = 0; i < num_args; i++) {
        TaskArg* actual = &args[i];
        TaskArg* expected = &tmpl->args[i];
        
        // 1. raw_tensor must match exactly
        if (actual->region.raw_tensor != expected->region.raw_tensor) {
            fprintf(stderr, "[Replay Validate] ERROR: Task %d arg %d tensor pointer mismatch: "
                    "template=%p, actual=%p\n",
                    task_idx_in_fragment, i, 
                    expected->region.raw_tensor, actual->region.raw_tensor);
            valid = false;
        }
        
        // 2. col_offset must match exactly
        if (actual->region.col_offset != expected->region.col_offset) {
            fprintf(stderr, "[Replay Validate] ERROR: Task %d arg %d col_offset mismatch: "
                    "template=%lld, actual=%lld\n",
                    task_idx_in_fragment, i,
                    (long long)expected->region.col_offset, 
                    (long long)actual->region.col_offset);
            valid = false;
        }
        
        // 3. Shape (rows, cols) must match exactly
        if (actual->region.rows != expected->region.rows ||
            actual->region.cols != expected->region.cols) {
            fprintf(stderr, "[Replay Validate] ERROR: Task %d arg %d shape mismatch: "
                    "template=[%lld,%lld], actual=[%lld,%lld]\n",
                    task_idx_in_fragment, i,
                    (long long)expected->region.rows, (long long)expected->region.cols,
                    (long long)actual->region.rows, (long long)actual->region.cols);
            valid = false;
        }
        
        // 4. row_offset should differ by expected_offset_delta
        int64_t actual_row_offset = actual->region.row_offset;
        int64_t expected_row_offset = expected->region.row_offset + expected_offset_delta;
        
        // Apply offset mode
        if (ctx->offset_mode == OFFSET_NONE) {
            expected_row_offset = expected->region.row_offset;  // No adjustment
        }
        
        if (actual_row_offset != expected_row_offset) {
            fprintf(stderr, "[Replay Validate] ERROR: Task %d arg %d row_offset mismatch: "
                    "expected=%lld (template=%lld + delta=%d), actual=%lld\n",
                    task_idx_in_fragment, i,
                    (long long)expected_row_offset,
                    (long long)expected->region.row_offset, expected_offset_delta,
                    (long long)actual_row_offset);
            valid = false;
        }
        
        // 5. is_output flag must match
        if (actual->is_output != expected->is_output) {
            fprintf(stderr, "[Replay Validate] ERROR: Task %d arg %d is_output mismatch: "
                    "template=%d, actual=%d\n",
                    task_idx_in_fragment, i, expected->is_output, actual->is_output);
            valid = false;
        }
    }
    
    if (valid) {
        DEBUG_PRINT("[Replay Validate] Task %d at loop_idx %d: VALID (offset_delta=%d)\n",
               task_idx_in_fragment, loop_idx, expected_offset_delta);
    }
    
    return valid;
}

// =============================================================================
// Helper Macros for Code Generation
// =============================================================================

/**
 * Convenience macro to schedule an InCore function call
 * 
 * Usage:
 *   PTO_SCHEDULE_INCORE(rt, rowmax, input, 0, 0, 8, 8, output, 0, 0, 8, 1);
 */
#define PTO_SCHEDULE_INCORE_1IN_1OUT(rt, func, in_ptr, in_row, in_col, in_rows, in_cols, \
                                      out_ptr, out_row, out_col, out_rows, out_cols) \
    do { \
        int32_t _tid = pto_task_alloc(rt, #func, (void*)func); \
        if (_tid >= 0) { \
            pto_task_add_input(rt, _tid, in_ptr, in_row, in_col, in_rows, in_cols); \
            pto_task_add_output(rt, _tid, out_ptr, out_row, out_col, out_rows, out_cols); \
            pto_task_submit(rt, _tid); \
        } \
    } while(0)

#define PTO_SCHEDULE_INCORE_2IN_1OUT(rt, func, in1_ptr, in1_row, in1_col, in1_rows, in1_cols, \
                                      in2_ptr, in2_row, in2_col, in2_rows, in2_cols, \
                                      out_ptr, out_row, out_col, out_rows, out_cols) \
    do { \
        int32_t _tid = pto_task_alloc(rt, #func, (void*)func); \
        if (_tid >= 0) { \
            pto_task_add_input(rt, _tid, in1_ptr, in1_row, in1_col, in1_rows, in1_cols); \
            pto_task_add_input(rt, _tid, in2_ptr, in2_row, in2_col, in2_rows, in2_cols); \
            pto_task_add_output(rt, _tid, out_ptr, out_row, out_col, out_rows, out_cols); \
            pto_task_submit(rt, _tid); \
        } \
    } while(0)

// =============================================================================
// Multi-threaded Execution - Worker Thread and Runtime Entry
// =============================================================================

/**
 * Worker thread context
 */
typedef struct {
    PTORuntime* rt;
    int worker_id;
} WorkerContext;

/**
 * Execute a single task by calling its InCore function
 */
static void execute_task_internal(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    // Check if this is a compact task (replay) or a regular task
    CompactTask* ct = &rt->compact_task[task_id];
    bool is_compact = (ct->template_ref != NULL);
    
    // Get task info from appropriate source
    void* func_ptr;
    CycleCostFunc cycle_func;
    int32_t num_args;
    TaskArg* task_args;
    int32_t offset_delta;
    const char* func_name;
    
    if (is_compact) {
        RecordedTask* tmpl = ct->template_ref;
        func_ptr = tmpl->func_ptr;
        cycle_func = tmpl->cycle_func;
        num_args = tmpl->num_args;
        task_args = tmpl->args;
        offset_delta = ct->offset_delta;  // Apply replay offset
        func_name = tmpl->func_name;
    } else {
        PendingTask* task = &rt->pend_task[task_id];
        func_ptr = task->func_ptr;
        cycle_func = task->cycle_func;
        num_args = task->num_args;
        task_args = task->args;
        offset_delta = 0;
        func_name = task->func_name;
    }
    
    DEBUG_PRINT("[Worker] Executing task %d: %s%s\n", task_id, func_name,
           is_compact ? " (replay)" : "");
    
    // Build argument array from task arguments
    void* args[PTO_MAX_ARGS * 2];
    int arg_idx = 0;
    
    for (int i = 0; i < num_args; i++) {
        TaskArg* arg = &task_args[i];
        float* base_ptr = (float*)arg->region.raw_tensor;
        
        // Only apply offset_delta to tile-varying arguments
        // Heuristic: if rows > 1 AND matches typical tile size, it's tile-varying
        // Weights/constants typically have rows=1 or very small rows
        // Also: if the original row_offset was non-zero, always apply delta
        bool apply_offset = (offset_delta != 0) && 
                           (arg->region.row_offset > 0 || arg->region.rows > 1);
        
        int64_t row_offset = arg->region.row_offset;
        if (apply_offset) {
            row_offset += offset_delta;
        }
        
        int64_t offset = row_offset * arg->region.cols + arg->region.col_offset;
        args[arg_idx++] = (void*)(base_ptr + offset);
    }
    
    // Simulation mode: call cycle function and record trace
    if (rt->simulation_mode && cycle_func) {
        int64_t cycle_cost = cycle_func(args, num_args);
        
        // Get earliest_start_cycle for this task (set by producers in pto_task_complete)
        int64_t task_earliest_start = is_compact ? 
            ct->earliest_start_cycle : rt->pend_task[task_id].earliest_start_cycle;
        
        // Get worker's current cycle
        int64_t worker_current = pto_trace_get_cycle(worker_id);
        
        // Actual start time = max(worker ready, dependencies satisfied)
        int64_t actual_start = (worker_current > task_earliest_start) ? 
            worker_current : task_earliest_start;
        int64_t actual_end = actual_start + cycle_cost;
        
        // Update task's end_cycle (for propagating to dependents)
        if (is_compact) {
            ct->earliest_start_cycle = actual_end;  // Reuse field for end_cycle
        } else {
            rt->pend_task[task_id].end_cycle = actual_end;
        }
        
        // Record with dependency-aware timing
        pto_trace_record_with_time(worker_id, func_name, actual_start, actual_end);
        DEBUG_PRINT("[Worker] Task %d: %s (simulated, %lld cycles, start=%lld)\n", 
               task_id, func_name, (long long)cycle_cost, (long long)actual_start);
    }
    // Normal mode: call the InCore function
    else if (func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)func_ptr;
        func(args, num_args);
    } else {
        DEBUG_PRINT("[Worker] Task %d: %s (no execution - no func_ptr)\n", 
               task_id, func_name);
    }
}

static void execute_task(PTORuntime* rt, int32_t task_id) {
    execute_task_internal(rt, task_id, 0);
}

void pto_execute_task_with_worker(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    execute_task_internal(rt, task_id, worker_id);
}

/**
 * Worker thread function
 * Continuously fetches and executes tasks until shutdown
 */
static void* worker_thread_func(void* arg) {
    WorkerContext* ctx = (WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[Worker %d] Started\n", worker_id);
    fflush(stdout);
    
    while (!rt->shutdown_requested) {
        // Get next ready task (blocking)
        int32_t task_id = pto_get_ready_task_blocking(rt);
        
        if (task_id < 0) {
            // No task available - check if we should exit
            if (rt->shutdown_requested) {
                break;
            }
            // Check if all tasks are done
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) {
                break;
            }
            continue;
        }
        
        // Execute the task (pass worker_id for trace recording)
        execute_task_internal(rt, task_id, worker_id);
        
        // Mark task as complete (updates dependencies, may wake other workers)
        pto_task_complete_threadsafe(rt, task_id);
    }
    
    DEBUG_PRINT("[Worker %d] Exiting\n", worker_id);
    free(ctx);
    return NULL;
}

/**
 * ARM64 Runtime Entry Point
 */
int runtime_entry_arm64(PTOOrchFunc orch_func, void* user_data, int num_workers,
                        int execution_task_threshold) {
    if (!orch_func) {
        fprintf(stderr, "[PTO Runtime] ERROR: No orchestration function provided\n");
        return -1;
    }
    
    if (num_workers < 1) num_workers = 1;
    if (num_workers > PTO_MAX_WORKERS) num_workers = PTO_MAX_WORKERS;
    if (execution_task_threshold < 0) execution_task_threshold = 0;
    
    printf("[PTO Runtime] ========================================\n");
    printf("[PTO Runtime] ARM64 Multi-threaded Execution\n");
    printf("[PTO Runtime] Workers: %d\n", num_workers);
    if (execution_task_threshold > 0) {
        printf("[PTO Runtime] Execution threshold: %d tasks (pipelined)\n", execution_task_threshold);
    } else {
        printf("[PTO Runtime] Execution mode: wait for orchestration\n");
    }
    printf("[PTO Runtime] ========================================\n");
    
    // Allocate runtime on heap (PTORuntime can be large)
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "[PTO Runtime] ERROR: Failed to allocate runtime\n");
        return -1;
    }
    
    // Initialize runtime
    pto_runtime_init(rt);
    rt->num_workers = num_workers;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    rt->execution_task_threshold = execution_task_threshold;
    
    // Spawn worker threads
    printf("[PTO Runtime] Spawning %d worker threads...\n", num_workers);
    for (int i = 0; i < num_workers; i++) {
        WorkerContext* ctx = (WorkerContext*)malloc(sizeof(WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[PTO Runtime] ERROR: Failed to allocate worker context\n");
            // Cleanup already spawned threads
            rt->shutdown_requested = true;
            pthread_cond_broadcast(&rt->queue_not_empty);
            for (int j = 0; j < i; j++) {
                pthread_join(rt->workers[j], NULL);
            }
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
        ctx->rt = rt;
        ctx->worker_id = i;
        
        if (pthread_create(&rt->workers[i], NULL, worker_thread_func, ctx) != 0) {
            fprintf(stderr, "[PTO Runtime] ERROR: Failed to create worker thread %d\n", i);
            free(ctx);
            rt->shutdown_requested = true;
            pthread_cond_broadcast(&rt->queue_not_empty);
            for (int j = 0; j < i; j++) {
                pthread_join(rt->workers[j], NULL);
            }
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
        printf("[PTO Runtime] Created worker thread %d\n", i);
        fflush(stdout);
    }
    
    // Give workers a moment to start
    struct timespec start_delay = {0, 10000000};  // 10ms
    nanosleep(&start_delay, NULL);
    printf("[PTO Runtime] Workers started, now building task graph...\n");
    fflush(stdout);
    
    // Build task graph by calling orchestration function
    printf("[PTO Runtime] Building task graph...\n");
    fflush(stdout);
    orch_func(rt, user_data);
    
    // Mark that orchestration is complete - all tasks are now submitted
    pthread_mutex_lock(&rt->task_mutex);
    rt->execution_started = true;
    int64_t total_tasks = rt->total_tasks_scheduled;
    pthread_mutex_unlock(&rt->task_mutex);
    
    printf("[PTO Runtime] Task graph built: %lld tasks\n", (long long)total_tasks);
    printf("[PTO Runtime] Executing tasks...\n");
    
    // Wake up any workers that might be waiting
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    // Wait for all tasks to complete
    // We poll active_task_count periodically
    struct timespec poll_interval = {0, 1000000};  // 1ms (was 50ms - too slow!)
    while (1) {
        pthread_mutex_lock(&rt->task_mutex);
        bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
        int64_t completed = rt->total_tasks_completed;
        pthread_mutex_unlock(&rt->task_mutex);
        
        if (all_done) {
            printf("[PTO Runtime] All %lld tasks completed!\n", (long long)completed);
            break;
        }
        
        // Progress report every 100ms or so
        static int64_t last_reported = 0;
        if (completed > last_reported + 1000 || completed == total_tasks) {
            printf("[PTO Runtime] Progress: %lld / %lld tasks (%.1f%%)\n",
                   (long long)completed, (long long)total_tasks,
                   100.0 * completed / total_tasks);
            last_reported = completed;
        }
        
        nanosleep(&poll_interval, NULL);
    }
    
    // Signal workers to shutdown
    printf("[PTO Runtime] Shutting down workers...\n");
    rt->shutdown_requested = true;
    
    // Wake up all workers
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    // Wait for all workers to exit
    for (int i = 0; i < num_workers; i++) {
        pthread_join(rt->workers[i], NULL);
    }
    
    // Print statistics
    printf("[PTO Runtime] ========================================\n");
    printf("[PTO Runtime] Execution Statistics\n");
    printf("[PTO Runtime]   Total tasks: %lld\n", (long long)rt->total_tasks_scheduled);
    printf("[PTO Runtime]   Completed:   %lld\n", (long long)rt->total_tasks_completed);
    printf("[PTO Runtime]   Workers:     %d\n", num_workers);
    printf("[PTO Runtime] ========================================\n");
    
    // Cleanup
    pto_runtime_shutdown(rt);
    free(rt);
    
    return 0;
}

/**
 * Register an InCore function (for lookup by name)
 */
void pto_register_incore_func(PTORuntime* rt, const char* func_name, PTOInCoreFunc func_ptr) {
    // For now, we store the function pointer directly in task->func_ptr when allocating
    // This function is a placeholder for a more sophisticated registry
    DEBUG_PRINT("[PTO Runtime] Registered InCore function: %s\n", func_name);
    (void)rt;
    (void)func_name;
    (void)func_ptr;
}

// =============================================================================
// Example: Generated Orchestration Function
// =============================================================================

#ifdef PTO_RUNTIME_EXAMPLE

// Forward declarations for InCore functions
void rowmax(float* input, float* output);
void rowexpandsub(float* input_x, float* input_row, float* output);
void elem_exp(float* input, float* output);
void rowsum(float* input, float* output);
void rowexpanddiv(float* input_x, float* input_row, float* output);

/**
 * Example: Generated Orchestration function for fused_softmax
 * 
 * This shows how an Orchestration function would be generated
 * to use the PTO runtime for scheduling InCore calls.
 */
void fused_softmax_orchestration(PTORuntime* rt,
                                  float* input, float* output,
                                  float* temp_rowmax, float* temp_shifted,
                                  float* temp_exp, float* temp_rowsum) {
    // Task 0: rowmax(input) -> temp_rowmax
    int32_t t0 = pto_task_alloc(rt, "rowmax", (void*)rowmax);
    pto_task_add_input(rt, t0, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t0, temp_rowmax, 0, 0, 8, 1);
    pto_task_submit(rt, t0);
    
    // Task 1: rowexpandsub(input, temp_rowmax) -> temp_shifted
    int32_t t1 = pto_task_alloc(rt, "rowexpandsub", (void*)rowexpandsub);
    pto_task_add_input(rt, t1, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t1, temp_rowmax, 0, 0, 8, 1);  // Creates dependency on t0
    pto_task_add_output(rt, t1, temp_shifted, 0, 0, 8, 8);
    pto_task_submit(rt, t1);
    
    // Task 2: elem_exp(temp_shifted) -> temp_exp
    int32_t t2 = pto_task_alloc(rt, "elem_exp", (void*)elem_exp);
    pto_task_add_input(rt, t2, temp_shifted, 0, 0, 8, 8);  // Creates dependency on t1
    pto_task_add_output(rt, t2, temp_exp, 0, 0, 8, 8);
    pto_task_submit(rt, t2);
    
    // Task 3: rowsum(temp_exp) -> temp_rowsum
    int32_t t3 = pto_task_alloc(rt, "rowsum", (void*)rowsum);
    pto_task_add_input(rt, t3, temp_exp, 0, 0, 8, 8);  // Creates dependency on t2
    pto_task_add_output(rt, t3, temp_rowsum, 0, 0, 8, 1);
    pto_task_submit(rt, t3);
    
    // Task 4: rowexpanddiv(temp_exp, temp_rowsum) -> output
    int32_t t4 = pto_task_alloc(rt, "rowexpanddiv", (void*)rowexpanddiv);
    pto_task_add_input(rt, t4, temp_exp, 0, 0, 8, 8);     // Creates dependency on t2
    pto_task_add_input(rt, t4, temp_rowsum, 0, 0, 8, 1);  // Creates dependency on t3
    pto_task_add_output(rt, t4, output, 0, 0, 8, 8);
    pto_task_submit(rt, t4);
    
    // Execute all scheduled tasks
    pto_execute_all(rt);
}

// Example main function
int main() {
    // Initialize runtime
    PTORuntime rt;
    pto_runtime_init(&rt);
    
    // Allocate buffers
    float input[64];
    float output[64];
    float temp_rowmax[8];
    float temp_shifted[64];
    float temp_exp[64];
    float temp_rowsum[8];
    
    // Initialize input (example)
    for (int i = 0; i < 64; i++) {
        input[i] = (float)i / 64.0f;
    }
    
    // Execute orchestration function
    fused_softmax_orchestration(&rt, input, output,
                                 temp_rowmax, temp_shifted,
                                 temp_exp, temp_rowsum);
    
    // Print statistics
    pto_runtime_stats(&rt);
    
    // Shutdown runtime
    pto_runtime_shutdown(&rt);
    
    return 0;
}

#endif // PTO_RUNTIME_EXAMPLE

// =============================================================================
// Cycle Trace Recording Implementation
// =============================================================================

CycleTrace* pto_global_trace = NULL;

void pto_trace_init(int32_t num_workers) {
    if (pto_global_trace) {
        free(pto_global_trace);
    }
    pto_global_trace = (CycleTrace*)calloc(1, sizeof(CycleTrace));
    if (!pto_global_trace) return;
    
    pto_global_trace->count = 0;
    pto_global_trace->num_workers = num_workers > 0 ? num_workers : 1;
    pto_global_trace->enabled = true;
    
    // Initialize per-worker cycle counters
    for (int i = 0; i < PTO_MAX_WORKERS; i++) {
        pto_global_trace->per_worker_cycle[i] = 0;
    }
}

void pto_trace_record(int32_t worker_id, const char* func_name, int64_t cycle_cost) {
    if (!pto_global_trace || !pto_global_trace->enabled) return;
    if (pto_global_trace->count >= PTO_MAX_TRACE_ENTRIES) return;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return;
    
    int idx = pto_global_trace->count++;
    CycleTraceEntry* entry = &pto_global_trace->entries[idx];
    
    // Copy function name
    strncpy(entry->func_name, func_name ? func_name : "unknown", PTO_MAX_FUNC_NAME_LEN - 1);
    entry->func_name[PTO_MAX_FUNC_NAME_LEN - 1] = '\0';
    
    entry->worker_id = worker_id;
    entry->start_cycle = pto_global_trace->per_worker_cycle[worker_id];
    entry->end_cycle = entry->start_cycle + cycle_cost;
    
    // Update worker cycle counter
    pto_global_trace->per_worker_cycle[worker_id] = entry->end_cycle;
}

void pto_trace_record_with_time(int32_t worker_id, const char* func_name, 
                                 int64_t start_cycle, int64_t end_cycle) {
    if (!pto_global_trace || !pto_global_trace->enabled) return;
    if (pto_global_trace->count >= PTO_MAX_TRACE_ENTRIES) return;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return;
    
    int idx = pto_global_trace->count++;
    CycleTraceEntry* entry = &pto_global_trace->entries[idx];
    
    // Copy function name
    strncpy(entry->func_name, func_name ? func_name : "unknown", PTO_MAX_FUNC_NAME_LEN - 1);
    entry->func_name[PTO_MAX_FUNC_NAME_LEN - 1] = '\0';
    
    entry->worker_id = worker_id;
    entry->start_cycle = start_cycle;
    entry->end_cycle = end_cycle;
    
    // Update worker cycle counter to the end of this task
    pto_global_trace->per_worker_cycle[worker_id] = end_cycle;
}

int64_t pto_trace_get_cycle(int32_t worker_id) {
    if (!pto_global_trace) return 0;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return 0;
    return pto_global_trace->per_worker_cycle[worker_id];
}

void pto_trace_cleanup(void) {
    if (pto_global_trace) {
        free(pto_global_trace);
        pto_global_trace = NULL;
    }
}

char* pto_trace_to_chrome_json(void) {
    if (!pto_global_trace) return NULL;
    
    // Estimate output size (generous allocation)
    size_t buf_size = 1024 + pto_global_trace->count * 256;
    char* buf = (char*)malloc(buf_size);
    if (!buf) return NULL;
    
    char* ptr = buf;
    ptr += sprintf(ptr, "{\n  \"traceEvents\": [\n");
    
    for (int i = 0; i < pto_global_trace->count; i++) {
        CycleTraceEntry* e = &pto_global_trace->entries[i];
        int64_t duration = e->end_cycle - e->start_cycle;
        
        // Chrome Tracing format (duration event)
        ptr += sprintf(ptr, "    {\"name\": \"%s\", \"cat\": \"task\", \"ph\": \"X\", "
                       "\"ts\": %lld, \"dur\": %lld, \"pid\": 0, \"tid\": %d}%s\n",
                       e->func_name,
                       (long long)(e->start_cycle),     // timestamp in microseconds (we use cycles)
                       (long long)duration,              // duration
                       e->worker_id,                     // thread ID = worker ID
                       (i < pto_global_trace->count - 1) ? "," : "");
    }
    
    ptr += sprintf(ptr, "  ],\n");
    ptr += sprintf(ptr, "  \"displayTimeUnit\": \"ns\",\n");
    ptr += sprintf(ptr, "  \"metadata\": {\n");
    ptr += sprintf(ptr, "    \"num_workers\": %d,\n", pto_global_trace->num_workers);
    ptr += sprintf(ptr, "    \"total_entries\": %d\n", pto_global_trace->count);
    ptr += sprintf(ptr, "  }\n");
    ptr += sprintf(ptr, "}\n");
    
    return buf;
}

void pto_trace_write_json(const char* filename) {
    char* json = pto_trace_to_chrome_json();
    if (!json) {
        fprintf(stderr, "Error: Failed to generate trace JSON\n");
        return;
    }
    
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Failed to open %s for writing\n", filename);
        free(json);
        return;
    }
    
    fputs(json, f);
    fclose(f);
    free(json);
    
    printf("Trace written to: %s\n", filename);
    printf("  Open in Chrome: chrome://tracing and load the file\n");
}

void pto_trace_print_summary(void) {
    if (!pto_global_trace) {
        printf("Trace: not initialized\n");
        return;
    }
    
    printf("\n=== Cycle Trace Summary ===\n");
    printf("Total entries: %d\n", pto_global_trace->count);
    printf("Num workers: %d\n", pto_global_trace->num_workers);
    
    // Per-worker statistics
    int64_t max_cycle = 0;
    for (int w = 0; w < pto_global_trace->num_workers; w++) {
        int64_t cycle = pto_global_trace->per_worker_cycle[w];
        printf("  Worker %d: %lld cycles\n", w, (long long)cycle);
        if (cycle > max_cycle) max_cycle = cycle;
    }
    printf("Max cycle (makespan): %lld\n", (long long)max_cycle);
    
    // Function breakdown
    printf("\nFunction breakdown:\n");
    
    // Simple aggregation (could be made more efficient with hash table)
    typedef struct { char name[PTO_MAX_FUNC_NAME_LEN]; int64_t total_cycles; int count; } FuncStats;
    FuncStats stats[100];
    int num_stats = 0;
    
    for (int i = 0; i < pto_global_trace->count; i++) {
        CycleTraceEntry* e = &pto_global_trace->entries[i];
        int64_t dur = e->end_cycle - e->start_cycle;
        
        // Find or create entry
        int found = -1;
        for (int s = 0; s < num_stats; s++) {
            if (strcmp(stats[s].name, e->func_name) == 0) {
                found = s;
                break;
            }
        }
        
        if (found >= 0) {
            stats[found].total_cycles += dur;
            stats[found].count++;
        } else if (num_stats < 100) {
            strncpy(stats[num_stats].name, e->func_name, PTO_MAX_FUNC_NAME_LEN - 1);
            stats[num_stats].total_cycles = dur;
            stats[num_stats].count = 1;
            num_stats++;
        }
    }
    
    for (int s = 0; s < num_stats; s++) {
        printf("  %s: %lld cycles (%d calls)\n", 
               stats[s].name, (long long)stats[s].total_cycles, stats[s].count);
    }
    printf("===========================\n\n");
}
