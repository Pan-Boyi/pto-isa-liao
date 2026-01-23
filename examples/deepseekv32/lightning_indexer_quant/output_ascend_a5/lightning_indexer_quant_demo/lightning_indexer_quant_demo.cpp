// PTO Program: lightning_indexer_quant_demo
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

extern "C" __global__ AICORE void lightning_indexer_quant_demo_kernel(__gm__ float __in__ *q_mem, __gm__ float __in__ *k_mem, __gm__ float __out__ *scores_mem) {
    using Shape_q_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_q_mem = Stride<1, 1, 1, 4, 1>;
    using Global_q_mem = GlobalTensor<float, Shape_q_mem, Stride_q_mem>;
    using Shape_k_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_k_mem = Stride<1, 1, 1, 4, 1>;
    using Global_k_mem = GlobalTensor<float, Shape_k_mem, Stride_k_mem>;
    using Shape_scores_mem = Shape<1, 1, 1, 4, 4>;
    using Stride_scores_mem = Stride<1, 1, 1, 4, 1>;
    using Global_scores_mem = GlobalTensor<float, Shape_scores_mem, Stride_scores_mem>;

    using Tile_q = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_k = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_k_t = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_scores = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;
    using Tile_pto_tmp_0 = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, -1, -1>;

    Tile_q q(4, 4);
    Tile_k k(4, 4);
    Tile_k_t k_t(4, 4);
    Tile_scores scores(4, 4);
    Tile_pto_tmp_0 pto_tmp_0(4, 4);

    TASSIGN(q, 0x0);
    TASSIGN(k, 0x80);
    TASSIGN(k_t, 0x100);
    TASSIGN(scores, 0x180);
    TASSIGN(pto_tmp_0, 0x200);

    Global_q_mem g_q_mem_0(q_mem + (0) * 4 + (0));
    TLOAD(q, g_q_mem_0);
    Global_k_mem g_k_mem_1(k_mem + (0) * 4 + (0));
    TLOAD(k, g_k_mem_1);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TTRANS(k_t, k, pto_tmp_0);
    set_flag(PIPE_V, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
    // TODO(a5): TMATMUL requires Left/Right/Acc tile locations not encoded in current DSL
    set_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
    Global_scores_mem g_scores_mem_4(scores_mem + (0) * 4 + (0));
    TSTORE(g_scores_mem_4, scores);
}

#ifdef PTO_NPU_SMOKE_RUNNER
#include <stddef.h>
typedef void* aclrtStream;

extern "C" const char* pto_program_name() { return "lightning_indexer_quant_demo"; }
static const int kPtoNumMemrefs = 3;
static const char* const kPtoMemrefNames[kPtoNumMemrefs] = {
    "q_mem",
    "k_mem",
    "scores_mem",
};
static const size_t kPtoMemrefBytes[kPtoNumMemrefs] = {
    (size_t)(64),
    (size_t)(64),
    (size_t)(64),
};
static const char* const kPtoMemrefDtypes[kPtoNumMemrefs] = {
    "f32",
    "f32",
    "f32",
};
static const size_t kPtoMemrefElemBytes[kPtoNumMemrefs] = {
    (size_t)(4),
    (size_t)(4),
    (size_t)(4),
};
static const int kPtoMemrefIsOutput[kPtoNumMemrefs] = {
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
    lightning_indexer_quant_demo_kernel<<<1, nullptr, stream>>>((float*)args[0], (float*)args[1], (float*)args[2]);
}
#endif  // PTO_NPU_SMOKE_RUNNER