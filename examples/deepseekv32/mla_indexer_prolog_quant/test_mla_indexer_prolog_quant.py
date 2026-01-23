#!/usr/bin/env python3
"""
CPU accuracy test for mla_indexer_prolog_quant.
Uses x86 reference C on x86, ARM64 generated C on arm_neon.
"""

import os
import subprocess
import tempfile
import ctypes
import numpy as np
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
X86_SRC = os.path.join(ROOT, "mla_indexer_prolog_quant_x86.c")
ARM_SRC = os.path.join(ROOT, "output_arm64", "mla_indexer_prolog_quant_demo", "mla_indexer_prolog_quant_demo.c")

T = 4
H = 4
Q_RANK = 4
OUT = 4
HEAD_NUM = 2


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


def reference(x, w_dq, w_qb, w_proj):
    q_norm = x @ w_dq
    q_out = q_norm @ w_qb
    weights = x @ w_proj
    return q_out, weights


def main():
    np.random.seed(0)
    x = np.random.uniform(-1, 1, (T, H)).astype(np.float32)
    w_dq = np.random.uniform(-1, 1, (H, Q_RANK)).astype(np.float32)
    w_qb = np.random.uniform(-1, 1, (Q_RANK, OUT)).astype(np.float32)
    w_proj = np.random.uniform(-1, 1, (H, HEAD_NUM)).astype(np.float32)

    is_arm = platform.machine() in ("arm64", "aarch64")
    if is_arm:
        so_path = compile_shared_lib(ARM_SRC, extra_cflags=["-march=armv8-a+simd"])
        func_name = "mla_indexer_prolog_quant_demo"
    else:
        so_path = compile_shared_lib(X86_SRC)
        func_name = "mla_indexer_prolog_quant_x86"
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, func_name)
    func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 6
    func.restype = None

    q_out = np.zeros((T, OUT), dtype=np.float32)
    weights = np.zeros((T, HEAD_NUM), dtype=np.float32)
    func(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_dq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_qb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    ref_q, ref_w = reference(x, w_dq, w_qb, w_proj)
    q_ok = np.allclose(q_out, ref_q, rtol=1e-4, atol=1e-5)
    w_ok = np.allclose(weights, ref_w, rtol=1e-4, atol=1e-5)
    print(f"q_out match: {q_ok}, max_err={np.max(np.abs(q_out - ref_q)):.3e}")
    print(f"weights match: {w_ok}, max_err={np.max(np.abs(weights - ref_w)):.3e}")
    return 0 if (q_ok and w_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
