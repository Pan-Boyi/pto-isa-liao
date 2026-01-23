#!/usr/bin/env python3
"""
CPU accuracy test for sparse_flash_attention_quant.
Uses x86 reference C on x86, ARM64 generated C on arm_neon.
"""

import os
import subprocess
import tempfile
import ctypes
import numpy as np
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
X86_SRC = os.path.join(ROOT, "sparse_flash_attention_quant_x86.c")
ARM_SRC = os.path.join(ROOT, "output_arm64", "sparse_flash_attention_quant_demo", "sparse_flash_attention_quant_demo.c")

TQ = 4
TK = 4
D = 4


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


def reference(q, k, v):
    scores = (q @ k.T) / np.sqrt(D)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    out = probs @ v
    return out


def main():
    np.random.seed(0)
    q = np.random.uniform(-1, 1, (TQ, D)).astype(np.float32)
    k = np.random.uniform(-1, 1, (TK, D)).astype(np.float32)
    v = np.random.uniform(-1, 1, (TK, D)).astype(np.float32)

    is_arm = platform.machine() in ("arm64", "aarch64")
    if is_arm:
        so_path = compile_shared_lib(ARM_SRC, extra_cflags=["-march=armv8-a+simd"])
        func_name = "sparse_flash_attention_quant_demo"
    else:
        so_path = compile_shared_lib(X86_SRC)
        func_name = "sparse_flash_attention_quant_x86"
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, func_name)
    func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4
    func.restype = None

    out = np.zeros((TQ, D), dtype=np.float32)
    func(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    ref = reference(q, k, v)
    ok = np.allclose(out, ref, rtol=1e-4, atol=1e-5)
    print(f"out match: {ok}, max_err={np.max(np.abs(out - ref)):.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
