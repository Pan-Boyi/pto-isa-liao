// PTO Program: rowexpanddiv
// Function Type: InCore (single-core tile computation)
// Execution Mode: Single-Core (SPSD) - NOT SPMD kernel
// This function is scheduled as a task by PTO Runtime
// ======================================================================
// TILE BUFFER ANALYSIS: rowexpanddiv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 544 bytes (0.5 KB)
//   Total capacity (w/ reuse): 544 bytes (0.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               8x8        f32       256   [  2,   3]           -
//   row_vals             8x1        f32        32   [  1,   2]           -
//   x                    8x8        f32       256   [  0,   2]           -
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

/**
 * InCore Function: rowexpanddiv
 * Single-core tile computation function.
 * Called by PTO Runtime as a scheduled task.
 * NOT a kernel entry - use rowexpanddiv_kernel_wrapper() to launch as kernel.
 */
class rowexpanddivInCore {
public:
    // Single-core constructor - no block coordination
    __aicore__ inline rowexpanddivInCore() {}

    // Initialize with global memory pointers
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 544);
        pipe.InitBuffer(outQueueY, 1, 544);
    }

    // Main processing - single tile, single core
    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 136);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 1 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (2 ops): TROWEXPANDDIV; TSTORE
        BroadcastDiv(result, x, row_vals, 64, 8);  // row-wise broadcast divide
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 136);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outputGm;
};

/**
 * Callable InCore function for PTO Runtime task scheduling.
 * This function is invoked by the runtime when this task is dispatched.
 * Execution: Single AI Core, single tile at specified offset.
 */
__aicore__ inline void rowexpanddiv(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    // Calculate byte offsets for this tile
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);
    
    rowexpanddivInCore op;
    op.Init((GM_ADDR)((uint8_t*)input + in_offset), 
            (GM_ADDR)((uint8_t*)output + out_offset));
    op.Process();
}

#ifdef PTO_GENERATE_SPMD_KERNEL
/**
 * SPMD Kernel Wrapper (for standalone testing only)
 * This launches the InCore function as a multi-core kernel.
 * In production, use PTO Runtime to schedule tasks instead.
 */
extern "C" __global__ __aicore__ void rowexpanddiv_kernel(GM_ADDR input, GM_ADDR output) {
    rowexpanddivInCore op;
    op.Init(input, output);
    op.Process();
}
#endif  // PTO_GENERATE_SPMD_KERNEL