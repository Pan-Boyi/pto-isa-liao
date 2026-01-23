#!/usr/bin/env python3
"""
CPU accuracy test for lightning_indexer_quant.
Uses x86 reference C on x86, ARM64 generated C on arm_neon.
"""

import os
import subprocess
import tempfile
import ctypes
import numpy as np
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
X86_SRC = os.path.join(ROOT, "lightning_indexer_quant_x86.c")
ARM_SRC = os.path.join(ROOT, "output_arm64", "lightning_indexer_quant_demo",
                       "lightning_indexer_quant_demo.c")

T = 4
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


def reference(q, k):
    return q @ k.T


def main():
    np.random.seed(0)
    q = np.random.uniform(-1, 1, (T, D)).astype(np.float32)
    k = np.random.uniform(-1, 1, (T, D)).astype(np.float32)

    is_arm = platform.machine() in ("arm64", "aarch64")
    if is_arm:
        so_path = compile_shared_lib(ARM_SRC, extra_cflags=["-march=armv8-a+simd"])
        func_name = "lightning_indexer_quant_demo"
    else:
        so_path = compile_shared_lib(X86_SRC)
        func_name = "lightning_indexer_quant_x86"
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, func_name)
    func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
    func.restype = None

    scores = np.zeros((T, T), dtype=np.float32)
    func(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    ref = reference(q, k)
    ok = np.allclose(scores, ref, rtol=1e-4, atol=1e-5)
    print(f"scores match: {ok}, max_err={np.max(np.abs(scores - ref)):.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
