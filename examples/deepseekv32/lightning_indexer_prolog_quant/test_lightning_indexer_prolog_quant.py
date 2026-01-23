#!/usr/bin/env python3
"""
CPU accuracy test for lightning_indexer_prolog_quant.
Uses x86 reference C on x86, ARM64 generated C on arm_neon.
"""

import os
import subprocess
import tempfile
import ctypes
import numpy as np
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
X86_SRC = os.path.join(ROOT, "lightning_indexer_prolog_quant_x86.c")
ARM_SRC = os.path.join(ROOT, "output_arm64", "lightning_indexer_prolog_quant_demo",
                       "lightning_indexer_prolog_quant_demo.c")

T = 4
Q_LORA_RANK = 4
H = 4
HEAD_NUM = 2
HEAD_DIM = 2
OUT_DIM = HEAD_NUM * HEAD_DIM


def compile_shared_lib(src_path: str, extra_cflags=None) -> str:
    fd, so_path = tempfile.mkstemp(suffix=".so")
    os.close(fd)
    cmd = ["gcc", "-O2", "-shared", "-fPIC", "-o", so_path, src_path, "-lm"]
    if extra_cflags:
        cmd[1:1] = extra_cflags
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")
    return so_path


def reference(q_norm, q_norm_scale, w_qb, w_qb_scale, x, w_proj):
    q_matmul = q_norm @ w_qb
    q_scaled = q_matmul * q_norm_scale.reshape(T, 1)
    q_out = q_scaled * w_qb_scale.reshape(1, OUT_DIM)

    weights_raw = x @ w_proj
    weight_scale = 1.0 / np.sqrt(HEAD_NUM * HEAD_DIM)
    weights_out = weights_raw * weight_scale
    return q_out, weights_out


def main():
    np.random.seed(0)
    q_norm = np.random.uniform(-1, 1, (T, Q_LORA_RANK)).astype(np.float32)
    q_norm_scale = np.random.uniform(-1, 1, (T, 1)).astype(np.float32)
    w_qb = np.random.uniform(-1, 1, (Q_LORA_RANK, OUT_DIM)).astype(np.float32)
    w_qb_scale = np.random.uniform(-1, 1, (1, OUT_DIM)).astype(np.float32)
    x = np.random.uniform(-1, 1, (T, H)).astype(np.float32)
    w_proj = np.random.uniform(-1, 1, (H, HEAD_NUM)).astype(np.float32)

    is_arm = platform.machine() in ("arm64", "aarch64")
    if is_arm:
        so_path = compile_shared_lib(ARM_SRC, extra_cflags=["-march=armv8-a+simd"])
        func_name = "lightning_indexer_prolog_quant_demo"
    else:
        so_path = compile_shared_lib(X86_SRC)
        func_name = "lightning_indexer_prolog_quant_x86"
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, func_name)
    func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 8
    func.restype = None

    q_out = np.zeros((T, OUT_DIM), dtype=np.float32)
    weights_out = np.zeros((T, HEAD_NUM), dtype=np.float32)
    func(
        q_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q_norm_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_qb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_qb_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    ref_q, ref_w = reference(q_norm, q_norm_scale, w_qb, w_qb_scale, x, w_proj)

    q_ok = np.allclose(q_out, ref_q, rtol=1e-4, atol=1e-5)
    w_ok = np.allclose(weights_out, ref_w, rtol=1e-4, atol=1e-5)
    print(f"q_out match: {q_ok}, max_err={np.max(np.abs(q_out - ref_q)):.3e}")
    print(f"weights_out match: {w_ok}, max_err={np.max(np.abs(weights_out - ref_w)):.3e}")
    return 0 if (q_ok and w_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
