// PTO Program: indexer_prolog
// Function Type: InCore (single-core tile computation)
// Execution Mode: Single-Core (SPSD) - NOT SPMD kernel
// This function is scheduled as a task by PTO Runtime
// ======================================================================
// TILE BUFFER ANALYSIS: indexer_prolog
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     54
//   Total capacity (no reuse): 8,256 bytes (8.1 KB)
//   Total capacity (w/ reuse): 5,056 bytes (4.9 KB)
//   Reuse savings:            3,200 bytes (38.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   abs_x                8x8        f32       256   [ 27,  70]           -
//   beta                 8x8        f32       256   [ 43,  54]           -
//   beta_1n              1x8        f32        32   [ 41,  43]           -
//   cache_index          8x1        i64        32   [ 77,  -1]           -
//   cos                  8x4        f32       128   [ 15,  63]           -
//   diff                 8x8        f32       256   [ 46,  52]           -
//   gamma                8x8        f32       256   [ 42,  53]           -
//   gamma_1n             1x8        f32        32   [ 40,  42]           <- w_scale_1n
//   hadamard_q           8x8        f32       256   [ 25,  -1]           <- w_scale_exp
//   k_lin                8x8        f32       256   [ 39,  46]           -
//   k_ln                 8x8        f32       256   [ 52,  74]           <- sq
//   k_ln_scaled          8x8        f32       256   [ 53,  54]           <- diff
//   k_nope               8x4        f32       128   [ 56,  -1]           -
//   k_rope               8x4        f32       128   [ 55,  63]           -
//   left                 8x2        f32        64   [ 17,  -1]           -
//   max_safe             8x1        f32        32   [ 29,  76]           -
//   mean                 8x1        f32        32   [ 45,  46]           -
//   part1                8x4        f32       128   [ 21,  65]           -
//   part2                8x4        f32       128   [ 22,  65]           <- q_rope
//   q                    8x8        f32       256   [ 12,  14]           <- w_f32
//   q_nope               8x4        f32       128   [ 14,  -1]           -
//   q_norm               8x8        i8         64   [  3,   7]           -
//   q_norm_f32           8x8        f32       256   [  7,   8]           -
//   q_norm_fp32          8x8        f32       256   [  8,  -1]           -
//   q_norm_scale         8x1        f32        32   [  4,   8]           -
//   q_pre                8x8        f32       256   [ 26,  32]           <- q
//   q_rope               8x4        f32       128   [ 13,  21]           -
//   q_rope_rotated       8x4        f32       128   [ 23,  -1]           -
//   right                8x2        f32        64   [ 18,  61]           -
//   right_neg            8x2        f32        64   [ 19,  -1]           -
//   rotated              8x4        f32       128   [ 20,  64]           -
//   rowmax               8x1        f32        32   [ 28,  71]           <- q_norm_scale
//   scale_127_tile       8x8        f32       256   [ 30,  73]           -
//   scale_dequant        8x1        f32        32   [ 34,  79]           -
//   scale_quant          8x8        f32       256   [ 31,  74]           -
//   sin                  8x4        f32       128   [ 16,  64]           -
//   sq                   8x8        f32       256   [ 47,  48]           <- k_lin
//   std                  8x1        f32        32   [ 51,  52]           <- var
//   sum                  8x1        f32        32   [ 44,  45]           -
//   var                  8x1        f32        32   [ 49,  50]           <- mean
//   var_eps              8x1        f32        32   [ 50,  51]           <- var_sum
//   var_sum              8x1        f32        32   [ 48,  49]           <- sum
//   w                    8x8        i8         64   [  5,   9]           -
//   w_f32                8x8        f32       256   [  9,  11]           <- q_norm_f32
//   w_fp32               8x8        f32       256   [ 11,  -1]           -
//   w_k                  8x8        f32       256   [ 38,  -1]           -
//   w_proj               8x8        f32       256   [ 80,  -1]           <- abs_x
//   w_scale_1n           1x8        f32        32   [  6,  10]           -
//   w_scale_exp          8x8        f32       256   [ 10,  11]           -
//   weights              8x8        f32       256   [ 81,  82]           <- scale_127_tile
//   weights_scaled       8x8        f32       256   [ 82,  83]           <- scale_quant
//   x                    8x8        f32       256   [ 37,  -1]           <- q_pre
//   y                    8x8        f32       256   [ 32,  75]           -
//   y_int8               8x8        i8         64   [ 33,  78]           <- q_norm
//
// BUFFER REUSE MAP:
//   w_f32 reuses buffer of q_norm_f32
//   q reuses buffer of w_f32
//   part2 reuses buffer of q_rope
//   hadamard_q reuses buffer of w_scale_exp
//   q_pre reuses buffer of q
//   rowmax reuses buffer of q_norm_scale
//   y_int8 reuses buffer of q_norm
//   x reuses buffer of q_pre
//   sq reuses buffer of k_lin
//   var_sum reuses buffer of sum
//   var reuses buffer of mean
//   var_eps reuses buffer of var_sum
//   std reuses buffer of var
//   k_ln reuses buffer of sq
//   k_ln_scaled reuses buffer of diff
//   gamma_1n reuses buffer of w_scale_1n
//   w_proj reuses buffer of abs_x
//   weights reuses buffer of scale_127_tile
//   weights_scaled reuses buffer of scale_quant
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend A2/A3 (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

/**
 * InCore Function: indexer_prolog
 * Single-core tile computation function.
 * Called by PTO Runtime as a scheduled task.
 * NOT a kernel entry - use indexer_prolog_kernel_wrapper() to launch as kernel.
 */
class indexer_prologInCore {
public:
    // Single-core constructor - no block coordination
    __aicore__ inline indexer_prologInCore() {}

    // Initialize with global memory pointers
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 8832);
        pipe.InitBuffer(outQueueY, 1, 8832);
    }

    // Main processing - single tile, single core
    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 2208);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 18 loop overheads saved

        int inv_cols = 8;

        int eps = 0;

        int scale = 8;

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // TCVT: q_norm_f32 = (float)(q_norm) rmode=CAST_RINT
        for (int _i = 0; _i < 8; _i++) {
            for (int _j = 0; _j < 8; _j++) {
                q_norm_f32[_i][_j] = (float)(q_norm[_i][_j]);
            }}

        // FUSED (1 ops): TROWEXPANDMUL
        BroadcastMul(q_norm_fp32, q_norm_f32, q_norm_scale, 64, 8);  // row-wise broadcast multiply

        // TCVT: w_f32 = (float)(w) rmode=CAST_RINT
        for (int _i = 0; _i < 8; _i++) {
            for (int _j = 0; _j < 8; _j++) {
                w_f32[_i][_j] = (float)(w[_i][_j]);
            }}

        // TCOLEXPAND: Not implemented

        // FUSED (1 ops): TMUL
        Mul(w_fp32, w_f32, w_scale_exp, 64);

        // TMATMUL: q = q_norm_fp32 @ w_fp32
        Matmul(q, q_norm_fp32, w_fp32, 8, 8);

        // TEXTRACT: q_rope = extract(q, [0, 0])
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 4; _c++) {
                q_rope[_r][_c] = q[(0) + _r][(0) + _c];
            }}

        // TEXTRACT: q_nope = extract(q, [0, 4])
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 4; _c++) {
                q_nope[_r][_c] = q[(0) + _r][(4) + _c];
            }}

        // FUSED (2 ops): TLOAD; TLOAD
        // TLOAD: Operation
        // TLOAD: Operation

        // TEXTRACT: left = extract(q_rope, [0, 0])
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 2; _c++) {
                left[_r][_c] = q_rope[(0) + _r][(0) + _c];
            }}

        // TEXTRACT: right = extract(q_rope, [0, 2])
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 2; _c++) {
                right[_r][_c] = q_rope[(0) + _r][(2) + _c];
            }}

        // FUSED (1 ops): TNEG
        Neg(right_neg, right, 64);

        // TCONCAT(axis=1): rotated = [ right_neg | left ]
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 2; _c++) {
                rotated[_r][_c] = right_neg[_r][_c];
            }
            for (int _c = 0; _c < 2; _c++) {
                rotated[_r][2 + _c] = left[_r][_c];
            }}

        // FUSED (3 ops): TMUL; TMUL; TADD
        Mul(part1, q_rope, cos, 64);
        Mul(part2, rotated, sin, 64);
        Add(q_rope_rotated, part1, part2, 64);

        // TCONCAT(axis=1): q = [ q_rope_rotated | q_nope ]
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 4; _c++) {
                q[_r][_c] = q_rope_rotated[_r][_c];
            }
            for (int _c = 0; _c < 4; _c++) {
                q[_r][4 + _c] = q_nope[_r][_c];
            }}

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // TMATMUL: q_pre = q @ hadamard_q
        Matmul(q_pre, q, hadamard_q, 8, 8);

        // FUSED (1 ops): TABS
        Abs(abs_x, q_pre, 64);

        // TROWMAX: reduction max operation
        ReduceMax(rowmax, abs_x, 8);

        // FUSED (3 ops): TEXPANDS; TROWEXPANDDIV; TMUL
        Duplicate(scale_127_tile, 127.0f, 64);
        BroadcastDiv(scale_quant, scale_127_tile, max_safe, 64, 8);  // row-wise broadcast divide
        Mul(y, q_pre, scale_quant, 64);

        // TCVT: y_int8 = (int8_t)lrintf(y) F32->I8 rmode=CAST_RINT
        for (int _i = 0; _i < 8; _i++) {
            for (int _j = 0; _j < 8; _j++) {
                y_int8[_i][_j] = (int8_t)lrintf(y[_i][_j]);
            }}

        // FUSED (1 ops): TDIVS
        Divs(scale_dequant, max_safe, 127.0f, 64);

        // FUSED (1 ops): TSTORE
        // TSTORE: Operation

        // FUSED (1 ops): TSTORE
        // TSTORE: Operation

        // FUSED (2 ops): TLOAD; TLOAD
        // TLOAD: Operation
        // TLOAD: Operation

        // TMATMUL: k_lin = x @ w_k
        Matmul(k_lin, x, w_k, 8, 8);

        // FUSED (2 ops): TLOAD; TLOAD
        // TLOAD: Operation
        // TLOAD: Operation

        // TCOLEXPAND: Not implemented

        // TCOLEXPAND: Not implemented

        // TROWSUM: reduction operation
        ReduceSum(sum, k_lin, 8);

        // FUSED (1 ops): TDIVS
        Divs(mean, sum, inv_colsf, 64);

        // FUSED (2 ops): TROWEXPANDSUB; TMUL
        BroadcastSub(diff, k_lin, mean, 64, 8);  // row-wise broadcast subtract
        Mul(sq, diff, diff, 64);

        // TROWSUM: reduction operation
        ReduceSum(var_sum, sq, 8);

        // FUSED (3 ops): TDIVS; TADDS; TSQRT
        Divs(var, var_sum, inv_colsf, 64);
        Adds(var_eps, var, epsf, 64);
        Sqrt(std, var_eps, 64);

        // FUSED (3 ops): TROWEXPANDDIV; TMUL; TADD
        BroadcastDiv(k_ln, diff, std, 64, 8);  // row-wise broadcast divide
        Mul(k_ln_scaled, k_ln, gamma, 64);
        Add(k_ln, k_ln_scaled, beta, 64);

        // TEXTRACT: k_rope = extract(k_ln, [0, 0])
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 4; _c++) {
                k_rope[_r][_c] = k_ln[(0) + _r][(0) + _c];
            }}

        // TEXTRACT: k_nope = extract(k_ln, [0, 4])
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 4; _c++) {
                k_nope[_r][_c] = k_ln[(0) + _r][(4) + _c];
            }}

        // FUSED (2 ops): TLOAD; TLOAD
        // TLOAD: Operation
        // TLOAD: Operation

        // TEXTRACT: left = extract(k_rope, [0, 0])
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 2; _c++) {
                left[_r][_c] = k_rope[(0) + _r][(0) + _c];
            }}

        // TEXTRACT: right = extract(k_rope, [0, 2])
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 2; _c++) {
                right[_r][_c] = k_rope[(0) + _r][(2) + _c];
            }}

        // FUSED (1 ops): TNEG
        Neg(right_neg, right, 64);

        // TCONCAT(axis=1): rotated = [ right_neg | left ]
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 2; _c++) {
                rotated[_r][_c] = right_neg[_r][_c];
            }
            for (int _c = 0; _c < 2; _c++) {
                rotated[_r][2 + _c] = left[_r][_c];
            }}

        // FUSED (3 ops): TMUL; TMUL; TADD
        Mul(part1, k_rope, cos, 64);
        Mul(part2, rotated, sin, 64);
        Add(q_rope_rotated, part1, part2, 64);

        // TCONCAT(axis=1): q = [ q_rope_rotated | k_nope ]
        for (int _r = 0; _r < 8; _r++) {
            for (int _c = 0; _c < 4; _c++) {
                q[_r][_c] = q_rope_rotated[_r][_c];
            }
            for (int _c = 0; _c < 4; _c++) {
                q[_r][4 + _c] = k_nope[_r][_c];
            }}

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // TMATMUL: k_ln = q @ hadamard_q
        Matmul(k_ln, q, hadamard_q, 8, 8);

        // FUSED (1 ops): TABS
        Abs(abs_x, k_ln, 64);

        // TROWMAX: reduction max operation
        ReduceMax(rowmax, abs_x, 8);

        // FUSED (3 ops): TEXPANDS; TROWEXPANDDIV; TMUL
        Duplicate(scale_127_tile, 127.0f, 64);
        BroadcastDiv(scale_quant, scale_127_tile, max_safe, 64, 8);  // row-wise broadcast divide
        Mul(y, k_ln, scale_quant, 64);

        // TCVT: y_int8 = (int8_t)lrintf(y) F32->I8 rmode=CAST_RINT
        for (int _i = 0; _i < 8; _i++) {
            for (int _j = 0; _j < 8; _j++) {
                y_int8[_i][_j] = (int8_t)lrintf(y[_i][_j]);
            }}

        // FUSED (1 ops): TDIVS
        Divs(scale_dequant, max_safe, 127.0f, 64);

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // SCATTER_UPDATE: k_cache[cache_index[i,0]*8+j] = y_int8[i,j]
        for (int _i = 0; _i < 8; _i++) {
            int _row = (int)cache_index[_i][0];
            for (int _j = 0; _j < 8; _j++) {
                k_cache[_row * 8 + _j] = y_int8[_i][_j];
            }}

        // SCATTER_UPDATE: k_scale_cache[cache_index[i,0]*1+j] = scale_dequant[i,j]
        for (int _i = 0; _i < 8; _i++) {
            int _row = (int)cache_index[_i][0];
            for (int _j = 0; _j < 1; _j++) {
                k_scale_cache[_row * 1 + _j] = scale_dequant[_i][_j];
            }}

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // TMATMUL: weights = x @ w_proj
        Matmul(weights, x, w_proj, 8, 8);

        // FUSED (2 ops): TDIVS; TSTORE
        Divs(weights_scaled, weights, scalef, 64);
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 2208);
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
__aicore__ inline void indexer_prolog(
    GM_ADDR input, int32_t in_row_off, int32_t in_col_off,
    GM_ADDR output, int32_t out_row_off, int32_t out_col_off,
    int32_t tile_rows, int32_t tile_cols)
{
    // Calculate byte offsets for this tile
    int32_t in_offset = (in_row_off * tile_cols + in_col_off) * sizeof(float);
    int32_t out_offset = (out_row_off * tile_cols + out_col_off) * sizeof(float);
    
    indexer_prologInCore op;
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
extern "C" __global__ __aicore__ void indexer_prolog_kernel(GM_ADDR input, GM_ADDR output) {
    indexer_prologInCore op;
    op.Init(input, output);
    op.Process();
}
#endif  // PTO_GENERATE_SPMD_KERNEL