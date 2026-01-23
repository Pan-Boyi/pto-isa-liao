// PTO Program: mla_indexer_prolog_quant_demo
// Backend: Ascend A5 via PTO header implementations
#ifndef REGISTER_BASE
#define REGISTER_BASE
#endif
#include <stdint.h>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

#ifndef TPUSH
#define TPUSH(src_pipe, dst_pipe) set_flag((src_pipe), (dst_pipe), EVENT_ID0)
#endif
#ifndef TPOP
#define TPOP(src_pipe, dst_pipe) wait_flag((src_pipe), (dst_pipe), EVENT_ID0)
#endif

extern "C" __global__ AICORE void mla_indexer_prolog_quant_demo_kernel(__gm__ float __in__ *x_mem, __gm__ float __in__ *w_dq_mem, __gm__ float __in__ *w_qb_mem, __gm__ float __in__ *w_proj_mem, __gm__ float __out__ *q_out_mem, __gm__ float __out__ *weights_mem) {
    using Shape_x_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_x_mem = Stride<1, 1, 1, 4, 1>;
    using Global_x_mem = GlobalTensor<float, Shape_x_mem, Stride_x_mem>;
    using Shape_w_dq_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_w_dq_mem = Stride<1, 1, 1, 4, 1>;
    using Global_w_dq_mem = GlobalTensor<float, Shape_w_dq_mem, Stride_w_dq_mem>;
    using Shape_w_qb_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_w_qb_mem = Stride<1, 1, 1, 4, 1>;
    using Global_w_qb_mem = GlobalTensor<float, Shape_w_qb_mem, Stride_w_qb_mem>;
    using Shape_w_proj_mem = Shape<1, 1, 1, 4, 2>;
    using Stride_w_proj_mem = Stride<1, 1, 1, 2, 1>;
    using Global_w_proj_mem = GlobalTensor<float, Shape_w_proj_mem, Stride_w_proj_mem>;
    using Shape_q_out_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_q_out_mem = Stride<1, 1, 1, 4, 1>;
    using Global_q_out_mem = GlobalTensor<float, Shape_q_out_mem, Stride_q_out_mem>;
    using Shape_weights_mem = Shape<1, 1, 1, 4, 2>;
    using Stride_weights_mem = Stride<1, 1, 1, 2, 1>;
    using Global_weights_mem = GlobalTensor<float, Shape_weights_mem, Stride_weights_mem>;

    using Tile_x = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_dq = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_qb = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_proj = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_q_norm = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_q_out = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_weights = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;

    Tile_x x(4, 4);
    Tile_w_dq w_dq(4, 4);
    Tile_w_qb w_qb(4, 4);
    Tile_w_proj w_proj(4, 2);
    Tile_q_norm q_norm(4, 4);
    Tile_q_out q_out(4, 4);
    Tile_weights weights(4, 2);

    TASSIGN(x, 0x0);
    TASSIGN(w_dq, 0x80);
    TASSIGN(w_qb, 0x100);
    TASSIGN(w_proj, 0x180);
    TASSIGN(q_norm, 0x200);
    TASSIGN(q_out, 0x280);
    TASSIGN(weights, 0x300);

    Global_x_mem g_x_mem_0(x_mem + (0) * 4 + (0));
    TLOAD(x, g_x_mem_0);
    Global_w_dq_mem g_w_dq_mem_1(w_dq_mem + (0) * 4 + (0));
    TLOAD(w_dq, g_w_dq_mem_1);
    Global_w_qb_mem g_w_qb_mem_2(w_qb_mem + (0) * 4 + (0));
    TLOAD(w_qb, g_w_qb_mem_2);
    Global_w_proj_mem g_w_proj_mem_3(w_proj_mem + (0) * 2 + (0));
    TLOAD(w_proj, g_w_proj_mem_3);
    set_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
    // TODO(a5): TMATMUL requires Left/Right/Acc tile locations not encoded in current DSL
    // TODO(a5): TMATMUL requires Left/Right/Acc tile locations not encoded in current DSL
    // TODO(a5): TMATMUL requires Left/Right/Acc tile locations not encoded in current DSL
    set_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
    Global_q_out_mem g_q_out_mem_7(q_out_mem + (0) * 4 + (0));
    TSTORE(g_q_out_mem_7, q_out);
    Global_weights_mem g_weights_mem_8(weights_mem + (0) * 2 + (0));
    TSTORE(g_weights_mem_8, weights);
}

#ifdef PTO_NPU_SMOKE_RUNNER
#include <stddef.h>
typedef void* aclrtStream;

extern "C" const char* pto_program_name() { return "mla_indexer_prolog_quant_demo"; }
static const int kPtoNumMemrefs = 6;
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "x_mem",
    "w_dq_mem",
    "w_qb_mem",
    "w_proj_mem",
    "q_out_mem",
    "weights_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(64),
    (size_t)(64),
    (size_t)(64),
    (size_t)(32),
    (size_t)(64),
    (size_t)(32),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
    0,
    0,
    1,
    1,
};

extern "C" int pto_num_memrefs() { return kPtoNumMemrefs; }
extern "C" const char* pto_memref_name(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefNames[idx];
}
extern "C" size_t pto_memref_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefBytes[idx];
}
extern "C" const char* pto_memref_dtype(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return "";
    return kPtoMemrefDtypes[idx];
}
extern "C" size_t pto_memref_elem_bytes(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefElemBytes[idx];
}
extern "C" int pto_memref_is_output(int idx) {
    if (idx < 0 || idx >= kPtoNumMemrefs) return 0;
    return kPtoMemrefIsOutput[idx];
}

extern "C" void pto_launch(void **args, aclrtStream stream) {
    mla_indexer_prolog_quant_demo_kernel<<<1, nullptr, stream>>>((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4], (float*)args[5]);
}
#endif  // PTO_NPU_SMOKE_RUNNER