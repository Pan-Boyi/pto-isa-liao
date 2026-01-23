// PTO Program: mla_prolog_quant_demo
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

extern "C" __global__ AICORE void mla_prolog_quant_demo_kernel(__gm__ float __in__ *x_mem, __gm__ float __in__ *w_dq_mem, __gm__ float __in__ *w_uq_qr_mem, __gm__ float __out__ *q_out_mem) {
    using Shape_x_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_x_mem = Stride<1, 1, 1, 4, 1>;
    using Global_x_mem = GlobalTensor<float, Shape_x_mem, Stride_x_mem>;
    using Shape_w_dq_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_w_dq_mem = Stride<1, 1, 1, 4, 1>;
    using Global_w_dq_mem = GlobalTensor<float, Shape_w_dq_mem, Stride_w_dq_mem>;
    using Shape_w_uq_qr_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_w_uq_qr_mem = Stride<1, 1, 1, 4, 1>;
    using Global_w_uq_qr_mem = GlobalTensor<float, Shape_w_uq_qr_mem, Stride_w_uq_qr_mem>;
    using Shape_q_out_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_q_out_mem = Stride<1, 1, 1, 4, 1>;
    using Global_q_out_mem = GlobalTensor<float, Shape_q_out_mem, Stride_q_out_mem>;

    using Tile_x = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_dq = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_w_uq_qr = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_x_proj = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_x_sq = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_sum_sq = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_mean_sq = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_mean_eps = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_rms = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_x_norm = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_q_out = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_pto_tmp_0 = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;

    Tile_x x(4, 4);
    Tile_w_dq w_dq(4, 4);
    Tile_w_uq_qr w_uq_qr(4, 4);
    Tile_x_proj x_proj(4, 4);
    Tile_x_sq x_sq(4, 4);
    Tile_sum_sq sum_sq(4, 1);
    Tile_mean_sq mean_sq(4, 1);
    Tile_mean_eps mean_eps(4, 1);
    Tile_rms rms(4, 1);
    Tile_x_norm x_norm(4, 4);
    Tile_q_out q_out(4, 4);
    Tile_pto_tmp_0 pto_tmp_0(4, 4);

    TASSIGN(x, 0x0);
    TASSIGN(w_dq, 0x80);
    TASSIGN(w_uq_qr, 0x100);
    TASSIGN(x_proj, 0x180);
    TASSIGN(x_sq, 0x200);
    TASSIGN(sum_sq, 0x280);
    TASSIGN(mean_sq, 0x300);
    TASSIGN(mean_eps, 0x380);
    TASSIGN(rms, 0x400);
    TASSIGN(x_norm, 0x480);
    TASSIGN(q_out, 0x500);
    TASSIGN(pto_tmp_0, 0x580);

    Global_x_mem g_x_mem_0(x_mem + (0) * 4 + (0));
    TLOAD(x, g_x_mem_0);
    Global_w_dq_mem g_w_dq_mem_1(w_dq_mem + (0) * 4 + (0));
    TLOAD(w_dq, g_w_dq_mem_1);
    Global_w_uq_qr_mem g_w_uq_qr_mem_2(w_uq_qr_mem + (0) * 4 + (0));
    TLOAD(w_uq_qr, g_w_uq_qr_mem_2);
    set_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID0);
    // TODO(a2a3): TMATMUL requires Left/Right/Acc tile locations not encoded in current DSL
    set_flag(PIPE_M, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_V, EVENT_ID0);
    TMUL(x_sq, x_proj, x_proj);
    TROWSUM(sum_sq, x_sq, pto_tmp_0);
    TDIVS(mean_sq, sum_sq, 4.0);
    TADDS(mean_eps, mean_sq, 1e-06);
    TSQRT(rms, mean_eps);
    TROWEXPANDDIV(x_norm, x_proj, rms);
    set_flag(PIPE_V, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
    // TODO(a2a3): TMATMUL requires Left/Right/Acc tile locations not encoded in current DSL
    set_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
    Global_q_out_mem g_q_out_mem_11(q_out_mem + (0) * 4 + (0));
    TSTORE(g_q_out_mem_11, q_out);
}

#ifdef PTO_NPU_SMOKE_RUNNER
#include <stddef.h>
typedef void* aclrtStream;

extern "C" const char* pto_program_name() { return "mla_prolog_quant_demo"; }
static const int kPtoNumMemrefs = 4;
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "x_mem",
    "w_dq_mem",
    "w_uq_qr_mem",
    "q_out_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(64),
    (size_t)(64),
    (size_t)(64),
    (size_t)(64),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
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
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
    0,
    0,
    0,
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
    mla_prolog_quant_demo_kernel<<<1, nullptr, stream>>>((float*)args[0], (float*)args[1], (float*)args[2], (float*)args[3]);
}
#endif  // PTO_NPU_SMOKE_RUNNER