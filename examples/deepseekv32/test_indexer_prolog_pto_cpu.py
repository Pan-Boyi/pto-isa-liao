#!/usr/bin/env python3
"""
检测 pto-isa-liao 生成的**合并版** indexer_prolog（单函数）在 CPU 下的执行与端到端精度。

用法:
  1. 生成 C：python pto_indexer_prolog.py（在 pto-isa-liao 根目录）
  2. 本脚本：python test_indexer_prolog_pto_cpu.py（同上）

本脚本会：
  - 若 indexer_prolog.c 尚未生成则先调用 pto_indexer_prolog 生成
  - 对生成 C 做兼容性修补（scalef/inv_colsf/epsf、TMAXS）
  - 编译为 .so（需 ARM64 及 arm_neon.h，否则跳过）
  - 用 ctypes 调用 pto_launch，跑合并版 indexer_prolog
  - 以 8x8 组合公式（q_linear->rope->hadamard->quant; k: x@w_k->ln->rope->hadamard->quant->scatter; weights）作为 golden，对比 5 路输出的端到端精度
"""

import os
import sys
import subprocess
import platform
import ctypes
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = _SCRIPT_DIR
_PTO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PTO_ROOT)

EXAMPLES = _EXAMPLES
PTO_ROOT = _PTO_ROOT
OUTPUT_DIR = os.path.join(EXAMPLES, "output_arm64", "indexer_prolog")
TILE_R, TILE_C = 8, 8

# 合并版 indexer_prolog 的 memref 顺序（与 pto_launch 一致）
MEMREF_ORDER = [
    "input_x", "input_q_norm", "input_q_norm_scale", "input_w", "input_w_qb_scale",
    "input_w_k", "input_w_proj", "input_cos", "input_sin",
    "input_hadamard_q", "input_hadamard_k", "cache_index_src",
    "k_cache", "k_scale_cache", "output_query", "output_query_scale", "output_weights",
]


def ensure_codegen():
    """若尚无 indexer_prolog.c 则运行 pto_indexer_prolog 生成。"""
    c_path = os.path.join(OUTPUT_DIR, "indexer_prolog.c")
    if not os.path.isfile(c_path):
        subprocess.check_call(
            [sys.executable, os.path.join(EXAMPLES, "pto_indexer_prolog.py")],
            cwd=PTO_ROOT,
        )
    if not os.path.isfile(c_path):
        raise FileNotFoundError(f"未找到 {c_path}，请先运行 pto_indexer_prolog.py")


def patch_scalar_float(c_path: str, replacements: list):
    with open(c_path, "r", encoding="utf-8") as f:
        c = f.read()
    for old, new in replacements:
        c = c.replace(old, new)
    with open(c_path, "w", encoding="utf-8") as f:
        f.write(c)


def patch_indexer_prolog(c_path: str) -> bool:
    """对合并版 indexer_prolog.c 做修补：scalef/inv_colsf/epsf、两处 TMAXS。"""
    with open(c_path, "r", encoding="utf-8") as f:
        c = f.read()
    changed = False
    reps = []
    if "scalef" in c:
        reps.append(("scalef", "(float)scale"))
    if "inv_colsf" in c:
        reps.append(("inv_colsf", "(float)inv_cols"))
    if "epsf" in c:
        reps.append(("epsf", "(float)eps"))
    if reps:
        for old, new in reps:
            c = c.replace(old, new)
        changed = True

    tmaxs_block = (
        "\n    // TMAXS: max_safe = max(rowmax, 1e-6f)  [PATCH: codegen omits TMAXS for (rows,1)]\n"
        "    for (int _row = 0; _row < 8; _row++) {\n"
        "        max_safe[_row][0] = fmaxf(rowmax[_row][0], 1e-6f);\n"
        "    }\n\n"
    )
    for tail in ("y=TMUL(q_pre,scale_quant)", "y=TMUL(k_ln,scale_quant)"):
        old = "        rowmax[_row][0] = _max;}\n\n    // FUSED LOOP (3 ops): scale_127_tile=TEXPANDS(127.0f); scale_quant=TROWEXPANDDIV(scale_127_tile,max_safe); " + tail
        if old in c and "TMAXS: max_safe" not in c.split(old)[0][-200:]:
            new = "        rowmax[_row][0] = _max;}" + tmaxs_block + "    // FUSED LOOP (3 ops): scale_127_tile=TEXPANDS(127.0f); scale_quant=TROWEXPANDDIV(scale_127_tile,max_safe); " + tail
            c = c.replace(old, new, 1)
            changed = True

    if changed:
        with open(c_path, "w", encoding="utf-8") as f:
            f.write(c)
    return changed


def compile_so(c_path: str, so_path: str) -> bool:
    cmd = [
        "clang" if platform.system() != "Windows" else "clang",
        "-shared", "-fPIC", "-O2", "-std=c11",
        "-DPTO_CPU_SMOKE_RUNNER",
        "-o", so_path, c_path, "-lm",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=PTO_ROOT)
        return os.path.isfile(so_path)
    except Exception:
        return False


# -------- Golden：合并版 8x8 全流程（与 PTO indexer_prolog 一致）--------

def _golden_q_linear_fp32(q_norm, q_norm_scale, w, w_qb_scale):
    q = np.asarray(q_norm, dtype=np.float32) * np.asarray(q_norm_scale, dtype=np.float32)
    wf = np.asarray(w, dtype=np.float32) * np.asarray(w_qb_scale, dtype=np.float32)
    return (q @ wf).tolist()


def _golden_rope(x, cos, sin, rope_head_dim=4):
    """
    RoPE: 只对前 rope_head_dim 列做 RoPE，其余列不做。
    与 torch 的 single_rope + split + concat 逻辑一致。
    """
    a = np.asarray(x, dtype=np.float32)
    c = np.asarray(cos, dtype=np.float32)
    s = np.asarray(sin, dtype=np.float32)
    # 只对前 rope_head_dim 列做 RoPE
    x_rope = a[..., :rope_head_dim]
    x_nope = a[..., rope_head_dim:]
    cos_rope = c[..., :rope_head_dim]
    sin_rope = s[..., :rope_head_dim]
    # rotate_half: x1, x2 = chunk(2), rotated = concat(-x2, x1)
    half = rope_head_dim // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    x_rope_rotated = x_rope * cos_rope + rotated * sin_rope
    # concat: [x_rope_rotated, x_nope]
    return np.concatenate([x_rope_rotated, x_nope], axis=-1).tolist()


def _golden_layer_norm(x, gamma=None, beta=None, eps=1e-6):
    """
    Layer Norm: (x - mean) / sqrt(var + eps) * gamma + beta
    如果 gamma/beta 为 None，则 gamma=1, beta=0（等价于无 gamma/beta，但公式更正确）。
    """
    a = np.asarray(x, dtype=np.float32)
    mean = a.mean(axis=1, keepdims=True)
    var = ((a - mean) ** 2).mean(axis=1, keepdims=True)
    std = np.sqrt(var + eps)
    normalized = np.where(std == 0, 1.0, (a - mean) / std)
    if gamma is not None:
        gamma_arr = np.asarray(gamma, dtype=np.float32)
        if gamma_arr.ndim == 1:
            gamma_arr = gamma_arr.reshape(1, -1)
        normalized = normalized * gamma_arr
    if beta is not None:
        beta_arr = np.asarray(beta, dtype=np.float32)
        if beta_arr.ndim == 1:
            beta_arr = beta_arr.reshape(1, -1)
        normalized = normalized + beta_arr
    return normalized.tolist()


def _golden_quant_int8(x, eps=1e-6):
    """
    quant_int8: 与 torch 一致，使用 round + trunc（通过 clip 模拟 trunc 的裁剪效果）。
    torch: round -> int32 -> trunc -> int8
    这里: round -> int32 -> clip -> int8（clip 在 [-128,127] 范围内等价于 trunc）。
    """
    a = np.asarray(x, dtype=np.float32)
    max_val = np.maximum(np.amax(np.abs(a), axis=1, keepdims=True), eps)
    scale_quant = 127.0 / max_val
    y_fp32 = a * scale_quant
    y_int32 = np.round(y_fp32).astype(np.int32)
    # torch.trunc(y_int32.to(x_dtype)).to(int8) 等价于 clip 到 [-128, 127] 然后 cast
    y_int8 = np.clip(y_int32, -128, 127).astype(np.int8)
    scale_dequant = max_val / 127.0
    return y_int8.tolist(), scale_dequant.tolist()


def _matmul_8x8(a, b):
    return (np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)).tolist()


def _golden_scatter_k_cache(k_cache_init, k_int8, cache_index, rows=8, cols=8):
    out = list(k_cache_init) if hasattr(k_cache_init, "__len__") else [0] * (rows * cols)
    for i in range(rows):
        r = int(cache_index[i][0])
        for j in range(cols):
            idx = r * cols + j
            if idx < len(out):
                out[idx] = int(k_int8[i][j])
    return out


def _golden_scatter_k_scale_cache(k_scale_init, scale_tile, cache_index, rows=8, cols=1):
    out = list(k_scale_init) if hasattr(k_scale_init, "__len__") else [0.0] * 64
    for i in range(rows):
        r = int(cache_index[i][0])
        idx = r * cols
        if idx < len(out):
            out[idx] = float(scale_tile[i][0])
    return out


def golden_indexer_prolog_combined(inp, rope_head_dim=4, n=8, d=8):
    """
    inp: dict with 8x8 (or 8x1, 1x8) arrays: x, q_norm, q_norm_scale, w, w_qb_scale, w_k, w_proj,
         cos, sin, hadamard_q, hadamard_k, cache_index (8x1 int).
        可选: layer_norm_gamma (8,), layer_norm_beta (8,)，默认 gamma=1, beta=0。
    rope_head_dim: RoPE 只对前 rope_head_dim 列做（默认 4，即 8 列的一半）。
    n, d: 用于 weights scale = (n*d)^-0.5（默认 n=8, d=8，即 1/sqrt(64)=1/8）。
    返回: query (8x8 int8), query_scale (8x1), k_cache (64 int8, 按 scatter 后), k_scale_cache (8 float), weights (8x8).
    """
    x = inp["x"]
    q_norm = inp["q_norm"]
    q_norm_scale = inp["q_norm_scale"]
    w = inp["w"]
    w_qb_scale = inp["w_qb_scale"]
    w_k = inp["w_k"]
    w_proj = inp["w_proj"]
    cos = inp["cos"]
    sin = inp["sin"]
    hadamard_q = inp["hadamard_q"]
    hadamard_k = inp["hadamard_k"]
    cache_index = inp["cache_index"]
    layer_norm_gamma = inp.get("layer_norm_gamma", [1.0] * d)
    layer_norm_beta = inp.get("layer_norm_beta", [0.0] * d)

    # Q: q_linear -> rope -> hadamard -> quant
    q = _golden_q_linear_fp32(q_norm, q_norm_scale, w, w_qb_scale)
    q = _golden_rope(q, cos, sin, rope_head_dim=rope_head_dim)
    q = _matmul_8x8(q, hadamard_q)
    query, query_scale = _golden_quant_int8(q, eps=1e-6)

    # K: x@w_k -> ln -> rope -> hadamard -> quant -> scatter
    k_lin = _matmul_8x8(x, w_k)
    k_ln = _golden_layer_norm(k_lin, gamma=layer_norm_gamma, beta=layer_norm_beta, eps=1e-6)
    k = _golden_rope(k_ln, cos, sin, rope_head_dim=rope_head_dim)
    k = _matmul_8x8(k, hadamard_k)
    k_int8, k_scale = _golden_quant_int8(k, eps=1e-6)
    k_cache = _golden_scatter_k_cache([0] * 64, k_int8, cache_index, 8, 8)
    k_scale_cache = _golden_scatter_k_scale_cache([0.0] * 8, k_scale, cache_index, 8, 1)

    # Weights: (n*d)^-0.5
    weights = np.asarray(_matmul_8x8(x, w_proj), dtype=np.float32)
    weights = weights * (n ** -0.5) * (d ** -0.5)
    weights = weights.tolist()

    return {
        "query": query,
        "query_scale": query_scale,
        "k_cache": k_cache,
        "k_scale_cache": k_scale_cache,
        "weights": weights,
    }


# -------- 运行合并版 indexer_prolog --------

def _fill_f32(src, buf, size=64):
    arr = np.ascontiguousarray(np.asarray(src, dtype=np.float32))
    n = min(size, arr.size)
    for i in range(n):
        buf[i] = float(arr.flat[i])


def _fill_i8(src, buf, size=64):
    flat = [int(src[i][j]) for i in range(len(src)) for j in range(len(src[0]) if src else 0)]
    for i in range(min(size, len(flat))):
        buf[i] = max(-128, min(127, int(flat[i])))


def _fill_f32_8x1(src, buf):
    for i in range(8):
        buf[i] = float(src[i][0] if isinstance(src[i], (list, tuple)) else src[i])


def _fill_i64_8x1(src, buf):
    for i in range(8):
        buf[i] = int(src[i][0] if isinstance(src[i], (list, tuple)) else src[i])


def run_indexer_prolog(so_path: str, inp: dict) -> dict:
    """跑合并版 indexer_prolog，返回 query, query_scale, k_cache(64), k_scale_cache(8), weights。"""
    lib = ctypes.CDLL(so_path)
    pto_launch = lib.pto_launch
    pto_launch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    pto_launch.restype = None

    # 分配 19 个 buffer（新增 gamma, beta）
    input_x = (ctypes.c_float * 64)()
    input_q_norm = (ctypes.c_int8 * 64)()
    input_q_norm_scale = (ctypes.c_float * 8)()
    input_w = (ctypes.c_int8 * 64)()
    input_w_qb_scale = (ctypes.c_float * 8)()  # 1x8
    input_w_k = (ctypes.c_float * 64)()
    input_w_proj = (ctypes.c_float * 64)()
    input_cos = (ctypes.c_float * 64)()
    input_sin = (ctypes.c_float * 64)()
    input_hadamard_q = (ctypes.c_float * 64)()
    input_hadamard_k = (ctypes.c_float * 64)()
    input_layer_norm_gamma = (ctypes.c_float * 8)()
    input_layer_norm_beta = (ctypes.c_float * 8)()
    cache_index_src = (ctypes.c_int64 * 8)()
    k_cache = (ctypes.c_int8 * 64)()  # 零初
    k_scale_cache = (ctypes.c_float * 8)()  # 零初
    output_query = (ctypes.c_int8 * 64)()
    output_query_scale = (ctypes.c_float * 8)()
    output_weights = (ctypes.c_float * 64)()

    _fill_f32(inp["x"], input_x)
    _fill_i8(inp["q_norm"], input_q_norm)
    _fill_f32_8x1(inp["q_norm_scale"], input_q_norm_scale)
    _fill_i8(inp["w"], input_w)
    for j in range(8):
        input_w_qb_scale[j] = float(inp["w_qb_scale"][0][j])
    _fill_f32(inp["w_k"], input_w_k)
    _fill_f32(inp["w_proj"], input_w_proj)
    _fill_f32(inp["cos"], input_cos)
    _fill_f32(inp["sin"], input_sin)
    _fill_f32(inp["hadamard_q"], input_hadamard_q)
    _fill_f32(inp["hadamard_k"], input_hadamard_k)
    for i in range(8):
        input_layer_norm_gamma[i] = float(inp["layer_norm_gamma"][i])
        input_layer_norm_beta[i] = float(inp["layer_norm_beta"][i])
    _fill_i64_8x1(inp["cache_index"], cache_index_src)

    arr = (ctypes.c_void_p * 19)(
        ctypes.c_void_p(ctypes.addressof(input_x)),
        ctypes.c_void_p(ctypes.addressof(input_q_norm)),
        ctypes.c_void_p(ctypes.addressof(input_q_norm_scale)),
        ctypes.c_void_p(ctypes.addressof(input_w)),
        ctypes.c_void_p(ctypes.addressof(input_w_qb_scale)),
        ctypes.c_void_p(ctypes.addressof(input_w_k)),
        ctypes.c_void_p(ctypes.addressof(input_w_proj)),
        ctypes.c_void_p(ctypes.addressof(input_cos)),
        ctypes.c_void_p(ctypes.addressof(input_sin)),
        ctypes.c_void_p(ctypes.addressof(input_hadamard_q)),
        ctypes.c_void_p(ctypes.addressof(input_hadamard_k)),
        ctypes.c_void_p(ctypes.addressof(input_layer_norm_gamma)),
        ctypes.c_void_p(ctypes.addressof(input_layer_norm_beta)),
        ctypes.c_void_p(ctypes.addressof(cache_index_src)),
        ctypes.c_void_p(ctypes.addressof(k_cache)),
        ctypes.c_void_p(ctypes.addressof(k_scale_cache)),
        ctypes.c_void_p(ctypes.addressof(output_query)),
        ctypes.c_void_p(ctypes.addressof(output_query_scale)),
        ctypes.c_void_p(ctypes.addressof(output_weights)),
    )
    pto_launch(ctypes.cast(arr, ctypes.POINTER(ctypes.c_void_p)), None)

    query = [[int(output_query[i * 8 + j]) for j in range(8)] for i in range(8)]
    query_scale = [[float(output_query_scale[i])] for i in range(8)]
    k_cache_list = [int(k_cache[i]) for i in range(64)]
    k_scale_cache_list = [float(k_scale_cache[i]) for i in range(8)]
    weights = [[float(output_weights[i * 8 + j]) for j in range(8)] for i in range(8)]

    return {
        "query": query,
        "query_scale": query_scale,
        "k_cache": k_cache_list,
        "k_scale_cache": k_scale_cache_list,
        "weights": weights,
    }


def compare_e2e(out: dict, gold: dict, atol: float = 1e-3) -> bool:
    ok = True
    # query: int8 完全一致
    ok_q = all(out["query"][i][j] == gold["query"][i][j] for i in range(8) for j in range(8))
    print(f"  [query] int8 match={ok_q}  pass={ok_q}")
    ok = ok and ok_q

    # query_scale: L_inf
    linf_qs = max(abs(float(out["query_scale"][i][0]) - float(gold["query_scale"][i][0])) for i in range(8))
    ok_qs = linf_qs <= atol
    print(f"  [query_scale] L_inf={linf_qs:.2e}  pass={ok_qs}")
    ok = ok and ok_qs

    # k_cache: 64 个 int8 一致（scatter 写回的 8 行）
    ok_k = all(out["k_cache"][i] == gold["k_cache"][i] for i in range(64))
    print(f"  [k_cache] int8 match={ok_k}  pass={ok_k}")
    ok = ok and ok_k

    # k_scale_cache: 8 个 L_inf
    linf_ks = max(abs(out["k_scale_cache"][i] - gold["k_scale_cache"][i]) for i in range(8))
    ok_ks = linf_ks <= atol
    print(f"  [k_scale_cache] L_inf={linf_ks:.2e}  pass={ok_ks}")
    ok = ok and ok_ks

    # weights: L_inf
    linf_w = max(abs(float(out["weights"][i][j]) - float(gold["weights"][i][j])) for i in range(8) for j in range(8))
    ok_w = linf_w <= atol
    print(f"  [weights] L_inf={linf_w:.2e}  pass={ok_w}")
    ok = ok and ok_w

    return ok


def main():
    print("=" * 60)
    print("indexer_prolog 合并版 — CPU 端到端精度检验")
    print("=" * 60)

    ensure_codegen()
    c_path = os.path.join(OUTPUT_DIR, "indexer_prolog.c")
    so_path = os.path.join(OUTPUT_DIR, "indexer_prolog_cpu.so")

    # 修补
    reps = []
    with open(c_path, "r", encoding="utf-8") as f:
        raw = f.read()
    if "scalef" in raw:
        reps.append(("scalef", "(float)scale"))
    if "inv_colsf" in raw:
        reps.append(("inv_colsf", "(float)inv_cols"))
    if "epsf" in raw:
        reps.append(("epsf", "(float)eps"))
    if reps:
        patch_scalar_float(c_path, reps)
        print("  [patch] indexer_prolog.c: " + ", ".join(f"{a}->{b}" for a, b in reps))
    if patch_indexer_prolog(c_path):
        print("  [patch] indexer_prolog.c: TMAXS(max_safe=max(rowmax,1e-6f)) x2")

    # 编译
    if not compile_so(c_path, so_path):
        print("  [skip] indexer_prolog 编译失败（需 ARM64/arm_neon），跳过")
        return 0
    print("  [ok] indexer_prolog 编译成功")

    # 测试数据（8x8 单 tile）
    rng = np.random.default_rng(42)
    rope_head_dim = 4  # 8 列的一半
    n, d = 8, 8
    inp = {
        "x": (rng.standard_normal((8, 8)).astype(np.float32) * 0.1).tolist(),
        "q_norm": (rng.integers(-127, 128, (8, 8), dtype=np.int8)).tolist(),
        "q_norm_scale": [[float(rng.random() * 0.01 + 0.001)] for _ in range(8)],
        "w": (rng.integers(-127, 128, (8, 8), dtype=np.int8)).tolist(),
        "w_qb_scale": [[float(rng.random() * 0.01 + 0.001) for _ in range(8)]],
        "w_k": (rng.standard_normal((8, 8)).astype(np.float32) * 0.1).tolist(),
        "w_proj": (rng.standard_normal((8, 8)).astype(np.float32) * 0.1).tolist(),
        "cos": (rng.standard_normal((8, 8)).astype(np.float32) * 0.3).clip(-1, 1).tolist(),  # 8x8，只使用前 rope_head_dim 列
        "sin": (rng.standard_normal((8, 8)).astype(np.float32) * 0.3).clip(-1, 1).tolist(),
        "hadamard_q": (rng.standard_normal((8, 8)).astype(np.float32) * 0.1).tolist(),
        "hadamard_k": (rng.standard_normal((8, 8)).astype(np.float32) * 0.1).tolist(),
        "layer_norm_gamma": [1.0] * d,  # 默认 gamma=1
        "layer_norm_beta": [0.0] * d,  # 默认 beta=0
        "cache_index": [[i] for i in range(8)],
    }

    gold = golden_indexer_prolog_combined(inp, rope_head_dim=rope_head_dim, n=n, d=d)
    out = run_indexer_prolog(so_path, inp)
    ok = compare_e2e(out, gold, atol=1e-2)

    print("=" * 60)
    if ok:
        print("结果: 通过（indexer_prolog 合并版 5 路输出与 golden 端到端一致）")
    else:
        print("结果: 存在差异，请查看上方各 pass")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
