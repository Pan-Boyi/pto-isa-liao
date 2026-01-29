/**
 * PTO Runtime System - A2A3 (Ascend) Platform Implementation
 * 
 * This file includes the modular A2A3 implementation from the
 * host/, orchestration/, and core/ subdirectories.
 * 
 * The implementation is split into:
 * - host/a2a3_host.c: Host CPU interface, memory management
 * - orchestration/a2a3_orchestration.c: Task queues, dependency management
 * - core/a2a3_core_worker.c: Worker threads and task execution
 * - core/: InCore intrinsics (header-only, inline implementations)
 */

// POSIX definitions must come FIRST, before ANY includes
// This enables clock_gettime, nanosleep, etc.
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <time.h>  // For clock_gettime, nanosleep

#include "pto_runtime_a2a3.h"

// Include layer implementations
#include "orchestration/a2a3_orchestration.c"
#include "host/a2a3_host.c"

// Include core worker implementation (platform-specific)
// Priority: A2A3_TARGET_SIMULATOR > A2A3_TARGET_HARDWARE
// If neither is defined, default to simulator
#if defined(A2A3_TARGET_SIMULATOR) || (!defined(A2A3_TARGET_HARDWARE) && !defined(CANN_SDK_AVAILABLE))
// Simulator: use cycle-accurate simulation (default when no hardware/SDK)
#include "../runtime_a2a3_sim/core/a2a3_core_worker.c"
#else
// Hardware: use hardware implementation (requires CANN SDK)
#include "core/a2a3_core_worker.c"
#endif
