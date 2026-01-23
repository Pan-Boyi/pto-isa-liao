// PTO Program: lightning_indexer_prolog_quant_demo
// Backend: Ascend A2A3 via PTO header implementations
#ifndef MEMORY_BASE
#define MEMORY_BASE
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

extern "C" __global__ AICORE void lightning_indexer_prolog_quant_demo_kernel(__gm__ float __in__ *q_norm_mem, __gm__ float __in__ *q_norm_scale_mem, __gm__ float __in__ *w_qb_mem, __gm__ float __in__ *w_qb_scale_mem, __gm__ float __in__ *x_mem, __gm__ float __in__ *w_proj_mem, __gm__ float __out__ *q_out_mem, __gm__ float __out__ *weights_out_mem) {
    using Shape_q_norm_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_q_norm_mem = Stride<1, 1, 1, 4, 1>;
    using Global_q_norm_mem = GlobalTensor<float, Shape_q_norm_mem, Stride_q_norm_mem>;
    using Shape_q_norm_scale_mem = Shape<1, 1, 1, 4, 1>;
    using Stride_q_norm_scale_mem = Stride<1, 1, 1, 1, 1>;
    using Global_q_norm_scale_mem = GlobalTensor<float, Shape_q_norm_scale_mem, Stride_q_norm_scale_mem>;
    using Shape_w_qb_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_w_qb_mem = Stride<1, 1, 1, 4, 1>;
    using Global_w_qb_mem = GlobalTensor<float, Shape_w_qb_mem, Stride_w_qb_mem>;
    using Shape_w_qb_scale_mem = Shape<1, 1, 1, 1, 4>;
    using Stride_w_qb_scale_mem = Stride<1, 1, 1, 4, 1>;
    using Global_w_qb_scale_mem = GlobalTensor<float, Shape_w_qb_scale_mem, Stride_w_qb_scale_mem>;
    using Shape_x_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_x_mem = Stride<1, 1, 1, 4, 1>;
    using Global_x_mem = GlobalTensor<float, Shape_x_mem, Stride_x_mem>;
    using Shape_w_proj_mem = Shape<1, 1, 1, 4, 2>;
    using Stride_w_proj_mem = Stride<1, 1, 1, 2, 1>;
    using Global_w_proj_mem = GlobalTensor<float, Shape_w_proj_mem, Stride_w_proj_mem>;
    using Shape_q_out_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_q_out_mem = Stride<1, 1, 1, 4, 1>;
    using Global_q_out_mem = GlobalTensor<float, Shape_q_out_mem, Stride_q_out_mem>;
    using Shape_weights_out_mem = Shape<1, 1, 1, 4, 2>;
    using Stride_weights_out_mem = Stride<1, 1, 1, 2, 1>;
    using Global_weights_out_mem = GlobalTensor<float, Shape_weights_out_mem, Stride_weights_out_mem>;

    using Tile_q_norm = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_q_norm_scale = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_qb = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_qb_scale = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_qb_scale_expand = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_q_matmul = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_q_scaled_row = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_q_out = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_x = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_proj = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_weights_raw = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_weights_out = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;

    Tile_q_norm q_norm(4, 4);
    Tile_q_norm_scale q_norm_scale(4, 1);
    Tile_w_qb w_qb(4, 4);
    Tile_w_qb_scale w_qb_scale(1, 4);
    Tile_w_qb_scale_expand w_qb_scale_expand(4, 4);
    Tile_q_matmul q_matmul(4, 4);
    Tile_q_scaled_row q_scaled_row(4, 4);
    Tile_q_out q_out(4, 4);
    Tile_x x(4, 4);
    Tile_w_proj w_proj(4, 2);
    Tile_weights_raw weights_raw(4, 2);
    Tile_weights_out weights_out(4, 2);

    TASSIGN(q_norm, 0x0);
    TASSIGN(q_norm_scale, 0x80);
    TASSIGN(w_qb, 0x100);
    TASSIGN(w_qb_scale, 0x180);
    TASSIGN(w_qb_scale_expand, 0x200);
    TASSIGN(q_matmul, 0x280);
    TASSIGN(q_scaled_row, 0x300);
    TASSIGN(q_out, 0x380);
    TASSIGN(x, 0x400);
    TASSIGN(w_proj, 0x480);
    TASSIGN(weights_raw, 0x500);
    TASSIGN(weights_out, 0x580);

    Global_q_norm_mem g_q_norm_mem_0(q_norm_mem + (0) * 4 + (0));
    TLOAD(q_norm, g_q_norm_mem_0);
    Global_q_norm_scale_mem g_q_norm_scale_mem_1(q_norm_scale_mem + (0) * 1 + (0));
    TLOAD(q_norm_scale, g_q_norm_scale_mem_1);
    Global_w_qb_mem g_w_qb_mem_2(w_qb_mem + (0) * 4 + (0));
    TLOAD(w_qb, g_w_qb_mem_2);
    Global_w_qb_scale_mem g_w_qb_scale_mem_3(w_qb_scale_mem + (0) * 4 + (0));
    TLOAD(w_qb_scale, g_w_qb_scale_mem_3);
    Global_x_mem g_x_mem_4(x_mem + (0) * 4 + (0));
    TLOAD(x, g_x_mem_4);
    Global_w_proj_mem g_w_proj_mem_5(w_proj_mem + (0) * 2 + (0));
    TLOAD(w_proj, g_w_proj_mem_5);
    set_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
    // TODO(a2a3): TMATMUL requires Left/Right/Acc tile locations not encoded in current DSL
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    TROWEXPANDMUL(q_scaled_row, q_matmul, q_norm_scale);
    TCOLEXPAND(w_qb_scale_expand, w_qb_scale);
    TMUL(q_out, q_scaled_row, w_qb_scale_expand);
    set_flag(PIPE_V, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
    // TODO(a2a3): TMATMUL requires Left/Right/Acc tile locations not encoded in current DSL
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    TMULS(weights_out, weights_raw, 0.5);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Global_q_out_mem g_q_out_mem_12(q_out_mem + (0) * 4 + (0));
    TSTORE(g_q_out_mem_12, q_out);
    Global_weights_out_mem g_weights_out_mem_13(weights_out_mem + (0) * 2 + (0));
    TSTORE(g_weights_out_mem_13, weights_out);
}

#ifdef PTO_NPU_SMOKE_RUNNER
#include <stddef.h>
typedef void* aclrtStream;

extern "C" const char* pto_program_name() { return "lightning_indexer_prolog_quant_demo"; }
static const int kPtoNumMemrefs = 8;
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "q_norm_mem",
    "q_norm_scale_mem",
    "w_qb_mem",
    "w_qb_scale_mem",
    "x_mem",
    "w_proj_mem",
    "q_out_mem",
    "weights_out_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(64),
    (size_t)(16),
    (size_t)(64),
    (size_t)(16),
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
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
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
    lightning_indexer_prolog_quant_demo_kernel<<<1, nullptr, stream>>>((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3], (float*)args[4], (float*)args[5], (float*)args[6], (float*)args[7]);
}
#endif  // PTO_NPU_SMOKE_RUNNER