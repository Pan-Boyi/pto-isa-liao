"""
PTO 实现的 indexer_prolog 合并版（单函数）

在 pto-isa-liao 框架上实现 deepseekv32 indexer_prolog，**所有逻辑写进同一 InCore**：
  - Q: q_linear_fp32（dequant+I8->F32 matmul）→ rope → hadamard → quant_int8 → query, query_scale
  - K: x@w_k → layer_norm → rope → hadamard → quant_int8 → scatter(k_cache, k_scale_cache)
  - Weights: x@w_proj * (n*d)^(-0.5)

生成一份 CPU (ARM64) .c 与一份 Ascend .cpp。Q 路径在 CPU 上为 fp32 回退；Ascend 上 int8×int8->int32 未实现。
详见同目录 IMPLEMENTATION_NOTES.md。
"""

import os
import sys

# 本脚本在 examples/deepseekv32/，需将项目根目录加入 path 以导入 pto_compile
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _ROOT)

from pto_compile import (
    PTOFunctionBuilder, PTOModule, PTOModuleCompiler,
    MultiBackendCodeGenerator,
)
from pto_isa_definition import ElementType, MemorySpace

TILE_ROWS = 8
TILE_COLS = 8


# =============================================================================
# InCore: indexer_prolog（合并版）— 所有逻辑写进同一函数，一份 CPU + 一份 Ascend
# =============================================================================
# 数据流: Q: q_linear_fp32 -> rope -> hadamard -> quant_int8 -> query, query_scale
#         K: x@w_k -> layer_norm -> rope -> hadamard -> quant_int8 -> scatter(k_cache, k_scale_cache)
#         Weights: x@w_proj * (n*d)^-0.5
# 单 8x8 tile，tile 复用以控制规模；与 golden 的 5 输出对齐：query(I8), query_scale(F32), idx_k_cache(I8), idx_k_scale_cache(F32), weights(F32)。
# =============================================================================

def create_indexer_prolog_combined_func(rows=TILE_ROWS, k=8, n=TILE_COLS, rope_head_dim=4, nope_dim=4):
    """
    rope_head_dim: RoPE 只对前 rope_head_dim 列做（默认 4，即 8 列的一半）。
    nope_dim: 不做 RoPE 的列数（默认 4，即 d - rope_head_dim）。
    """
    b = (
        PTOFunctionBuilder("indexer_prolog")
        .in_core()
        # ----- tiles: Q linear -----
        .tile("q_norm", rows, k, ElementType.I8)
        .tile("q_norm_scale", rows, 1, ElementType.F32)
        .tile("q_norm_f32", rows, k, ElementType.F32)
        .tile("q_norm_fp32", rows, k, ElementType.F32)
        .tile("w", k, n, ElementType.I8)
        .tile("w_scale_1n", 1, n, ElementType.F32)
        .tile("w_f32", k, n, ElementType.F32)
        .tile("w_scale_exp", k, n, ElementType.F32)
        .tile("w_fp32", k, n, ElementType.F32)
        .tile("q", rows, n, ElementType.F32)
        # ----- tiles: rope (shared Q/K) - 只对 rope_head_dim 做 -----
        .tile("q_rope", rows, rope_head_dim, ElementType.F32)
        .tile("q_nope", rows, nope_dim, ElementType.F32)
        .tile("cos", rows, rope_head_dim, ElementType.F32)
        .tile("sin", rows, rope_head_dim, ElementType.F32)
        .tile("left", rows, rope_head_dim // 2, ElementType.F32)
        .tile("right", rows, rope_head_dim // 2, ElementType.F32)
        .tile("right_neg", rows, rope_head_dim // 2, ElementType.F32)
        .tile("rotated", rows, rope_head_dim, ElementType.F32)
        .tile("part1", rows, rope_head_dim, ElementType.F32)
        .tile("part2", rows, rope_head_dim, ElementType.F32)
        .tile("q_rope_rotated", rows, rope_head_dim, ElementType.F32)
        # ----- tiles: hadamard -----
        .tile("hadamard_q", rows, n, ElementType.F32)
        .tile("q_pre", rows, n, ElementType.F32)
        # ----- tiles: quant (shared Q/K) -----
        .tile("abs_x", rows, n, ElementType.F32)
        .tile("rowmax", rows, 1, ElementType.F32)
        .tile("max_safe", rows, 1, ElementType.F32)
        .tile("scale_127_tile", rows, n, ElementType.F32)
        .tile("scale_quant", rows, n, ElementType.F32)
        .tile("y", rows, n, ElementType.F32)
        .tile("y_int8", rows, n, ElementType.I8)
        .tile("scale_dequant", rows, 1, ElementType.F32)
        # ----- tiles: K -----
        .tile("x", rows, k, ElementType.F32)
        .tile("w_k", k, n, ElementType.F32)
        .tile("k_lin", rows, n, ElementType.F32)
        .tile("sum", rows, 1, ElementType.F32)
        .tile("mean", rows, 1, ElementType.F32)
        .tile("diff", rows, n, ElementType.F32)
        .tile("sq", rows, n, ElementType.F32)
        .tile("var_sum", rows, 1, ElementType.F32)
        .tile("var", rows, 1, ElementType.F32)
        .tile("var_eps", rows, 1, ElementType.F32)
        .tile("std", rows, 1, ElementType.F32)
        .tile("k_ln", rows, n, ElementType.F32)
        .tile("k_ln_scaled", rows, n, ElementType.F32)
        .tile("gamma_1n", 1, n, ElementType.F32)
        .tile("beta_1n", 1, n, ElementType.F32)
        .tile("gamma", rows, n, ElementType.F32)
        .tile("beta", rows, n, ElementType.F32)
        .tile("k_rope", rows, rope_head_dim, ElementType.F32)
        .tile("k_nope", rows, nope_dim, ElementType.F32)
        .tile("cache_index", rows, 1, ElementType.I64)
        # ----- tiles: weights -----
        .tile("w_proj", k, n, ElementType.F32)
        .tile("weights", rows, n, ElementType.F32)
        .tile("weights_scaled", rows, n, ElementType.F32)
        # ----- memrefs -----
        .memref("input_x", MemorySpace.GM, ElementType.F32)
        .memref("input_q_norm", MemorySpace.GM, ElementType.I8)
        .memref("input_q_norm_scale", MemorySpace.GM, ElementType.F32)
        .memref("input_w", MemorySpace.GM, ElementType.I8)
        .memref("input_w_qb_scale", MemorySpace.GM, ElementType.F32)
        .memref("input_w_k", MemorySpace.GM, ElementType.F32)
        .memref("input_w_proj", MemorySpace.GM, ElementType.F32)
        .memref("input_cos", MemorySpace.GM, ElementType.F32)
        .memref("input_sin", MemorySpace.GM, ElementType.F32)
        .memref("input_hadamard_q", MemorySpace.GM, ElementType.F32)
        .memref("input_hadamard_k", MemorySpace.GM, ElementType.F32)
        .memref("input_layer_norm_gamma", MemorySpace.GM, ElementType.F32)
        .memref("input_layer_norm_beta", MemorySpace.GM, ElementType.F32)
        .memref("cache_index_src", MemorySpace.GM, ElementType.I64)
        .memref("k_cache", MemorySpace.GM, ElementType.I8)
        .memref("k_scale_cache", MemorySpace.GM, ElementType.F32)
        .memref("output_query", MemorySpace.GM, ElementType.I8)
        .memref("output_query_scale", MemorySpace.GM, ElementType.F32)
        .memref("output_weights", MemorySpace.GM, ElementType.F32)
        # ----- scalars -----
        .scalar("inv_cols", ElementType.F32)
        .scalar("eps", ElementType.F32)
        .scalar("scale", ElementType.F32)
        .scalar_li("inv_cols", 8)
        .scalar_li("eps", 0)  # 1e-6 无法用 SLI 设置，先用 0 近似（与 torch 的 1e-6 有微小差异）
        .scalar_li("scale", 8)  # (n*d)^-0.5，n=8, d=8 时 1/sqrt(64)=1/8
    )
    # ----- 1. Q linear -----
    b = (b
        .load("q_norm", "input_q_norm", 0, 0)
        .load("q_norm_scale", "input_q_norm_scale", 0, 0)
        .load("w", "input_w", 0, 0)
        .load("w_scale_1n", "input_w_qb_scale", 0, 0)
        .tcvt("q_norm_f32", "q_norm")
        .rowexpandmul("q_norm_fp32", "q_norm_f32", "q_norm_scale")
        .tcvt("w_f32", "w")
        .colexpand("w_scale_exp", "w_scale_1n")
        .mul("w_fp32", "w_f32", "w_scale_exp")
        .matmul("q", "q_norm_fp32", "w_fp32")
    )
    # ----- 2. Q rope: 只对前 rope_head_dim 列做 RoPE，其余列不做 -----
    b = (b
        .extract("q_rope", "q", 0, 0)  # 前 rope_head_dim 列
        .extract("q_nope", "q", 0, rope_head_dim)  # 后 nope_dim 列
        .load("cos", "input_cos", 0, 0)  # 只加载前 rope_head_dim 列
        .load("sin", "input_sin", 0, 0)
        .extract("left", "q_rope", 0, 0)
        .extract("right", "q_rope", 0, rope_head_dim // 2)
        .neg("right_neg", "right")
        .concat_col("rotated", "right_neg", "left")
        .mul("part1", "q_rope", "cos")
        .mul("part2", "rotated", "sin")
        .add("q_rope_rotated", "part1", "part2")
        .concat_col("q", "q_rope_rotated", "q_nope")  # concat: [q_rope_rotated, q_nope]
    )
    # ----- 3. Q hadamard -----
    b = (b
        .load("hadamard_q", "input_hadamard_q", 0, 0)
        .matmul("q_pre", "q", "hadamard_q")
    )
    # ----- 4. Q quant_int8 -----
    b = (b
        .abs("abs_x", "q_pre")
        .rowmax("rowmax", "abs_x")
        .maxs("max_safe", "rowmax", 1e-6)
        .expands("scale_127_tile", 127.0)
        .rowexpanddiv("scale_quant", "scale_127_tile", "max_safe")
        .mul("y", "q_pre", "scale_quant")
        .tcvt("y_int8", "y")
        .divs("scale_dequant", "max_safe", 127.0)
        .store("y_int8", "output_query", 0, 0)
        .store("scale_dequant", "output_query_scale", 0, 0)
    )
    # ----- 5. K linear -----
    b = (b
        .load("x", "input_x", 0, 0)
        .load("w_k", "input_w_k", 0, 0)
        .matmul("k_lin", "x", "w_k")
    )
    # ----- 6. K layer_norm: (x - mean) / sqrt(var + eps) * gamma + beta -----
    b = (b
        .load("gamma_1n", "input_layer_norm_gamma", 0, 0)  # (n,) -> (1, n)
        .load("beta_1n", "input_layer_norm_beta", 0, 0)
        .colexpand("gamma", "gamma_1n")  # (1, n) -> (rows, n)
        .colexpand("beta", "beta_1n")
        .rowsum("sum", "k_lin")
        .divs("mean", "sum", "inv_cols")
        .rowexpandsub("diff", "k_lin", "mean")
        .mul("sq", "diff", "diff")
        .rowsum("var_sum", "sq")
        .divs("var", "var_sum", "inv_cols")
        .adds("var_eps", "var", "eps")  # var + eps (eps=0，近似 torch 的 1e-6)
        .sqrt("std", "var_eps")
        .rowexpanddiv("k_ln", "diff", "std")  # (x - mean) / std
        .mul("k_ln_scaled", "k_ln", "gamma")  # * gamma
        .add("k_ln", "k_ln_scaled", "beta")  # + beta
    )
    # ----- 7. K rope: 只对前 rope_head_dim 列做 RoPE，其余列不做 -----
    b = (b
        .extract("k_rope", "k_ln", 0, 0)  # 前 rope_head_dim 列
        .extract("k_nope", "k_ln", 0, rope_head_dim)  # 后 nope_dim 列
        .load("cos", "input_cos", 0, 0)  # 重新加载 cos（Q rope 后可能被覆盖）
        .load("sin", "input_sin", 0, 0)
        .extract("left", "k_rope", 0, 0)
        .extract("right", "k_rope", 0, rope_head_dim // 2)
        .neg("right_neg", "right")
        .concat_col("rotated", "right_neg", "left")
        .mul("part1", "k_rope", "cos")
        .mul("part2", "rotated", "sin")
        .add("q_rope_rotated", "part1", "part2")
        .concat_col("q", "q_rope_rotated", "k_nope")  # concat: [k_rope_rotated, k_nope]
    )
    # ----- 8. K hadamard (load hadamard_k into hadamard_q, result in k_ln) -----
    b = (b
        .load("hadamard_q", "input_hadamard_k", 0, 0)
        .matmul("k_ln", "q", "hadamard_q")
    )
    # ----- 9. K quant_int8 + load cache_index + scatter -----
    b = (b
        .abs("abs_x", "k_ln")
        .rowmax("rowmax", "abs_x")
        .maxs("max_safe", "rowmax", 1e-6)
        .expands("scale_127_tile", 127.0)
        .rowexpanddiv("scale_quant", "scale_127_tile", "max_safe")
        .mul("y", "k_ln", "scale_quant")
        .tcvt("y_int8", "y")
        .divs("scale_dequant", "max_safe", 127.0)
        .load("cache_index", "cache_index_src", 0, 0)
        .scatter_update("k_cache", "y_int8", "cache_index")
        .scatter_update("k_scale_cache", "scale_dequant", "cache_index")
    )
    # ----- 10. Weights (x already in tile from step 5) -----
    b = (b
        .load("w_proj", "input_w_proj", 0, 0)
        .matmul("weights", "x", "w_proj")
        .divs("weights_scaled", "weights", "scale")
        .store("weights_scaled", "output_weights", 0, 0)
    )
    return b.build()


# =============================================================================
# 构建 PTO 模块并生成代码
# =============================================================================

def create_indexer_prolog_module():
    module = PTOModule("indexer_prolog_pto")
    module.add_function(create_indexer_prolog_combined_func())
    module.set_entry("indexer_prolog")
    return module


def main():
    output_base = _SCRIPT_DIR
    print("=" * 60)
    print("PTO indexer_prolog 合并版代码生成（单函数 → CPU + Ascend）")
    print("=" * 60)

    module = create_indexer_prolog_module()
    compiler = PTOModuleCompiler(inline_in_core=False, eliminate_redundant_mem=False)
    _ = compiler.compile(module)

    arm64_dir = os.path.join(output_base, "output_arm64", "indexer_prolog")
    ascend_dir = os.path.join(output_base, "output_ascend", "indexer_prolog")
    os.makedirs(arm64_dir, exist_ok=True)
    os.makedirs(ascend_dir, exist_ok=True)

    gen = MultiBackendCodeGenerator(enable_fusion=True, analyze_buffers=True, module=module)
    prog = module.get_function("indexer_prolog")

    # CPU (ARM64)
    code_arm64 = gen.generate_arm64(prog)
    path_arm64 = os.path.join(arm64_dir, "indexer_prolog.c")
    with open(path_arm64, "w") as f:
        f.write(code_arm64)
    print(f"  [ARM64] indexer_prolog -> {path_arm64}")

    # Ascend
    code_ascend = gen.generate_ascend(prog)
    path_ascend = os.path.join(ascend_dir, "indexer_prolog.cpp")
    with open(path_ascend, "w") as f:
        f.write(code_ascend)
    print(f"  [Ascend] indexer_prolog -> {path_ascend}")

    print("=" * 60)
    print("生成完成。运行 test_indexer_prolog_pto_cpu.py 做 CPU 端到端精度检验。")
    print("=" * 60)
    return module


if __name__ == "__main__":
    main()
