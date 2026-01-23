#!/usr/bin/env python3
"""
CPU accuracy test for mla_prolog_quant.
Uses x86 reference C on x86, ARM64 generated C on arm_neon.
"""

import os
import subprocess
import tempfile
import ctypes
import numpy as np
import platform

ROOT = os.path.dirname(os.path.abspath(__file__))
X86_SRC = os.path.join(ROOT, "mla_prolog_quant_x86.c")
ARM_SRC = os.path.join(ROOT, "output_arm64", "mla_prolog_quant_demo", "mla_prolog_quant_demo.c")

T = 4
H = 4
OUT = 4


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


def reference(x, w_dq, w_uq_qr):
    x_proj = x @ w_dq
    mean_sq = np.mean(x_proj * x_proj, axis=1, keepdims=True)
    rms = np.sqrt(mean_sq + 1e-6)
    x_norm = x_proj / rms
    q_out = x_norm @ w_uq_qr
    return q_out


def main():
    np.random.seed(0)
    x = np.random.uniform(-1, 1, (T, H)).astype(np.float32)
    w_dq = np.random.uniform(-1, 1, (H, H)).astype(np.float32)
    w_uq_qr = np.random.uniform(-1, 1, (H, OUT)).astype(np.float32)

    is_arm = platform.machine() in ("arm64", "aarch64")
    if is_arm:
        so_path = compile_shared_lib(ARM_SRC, extra_cflags=["-march=armv8-a+simd"])
        func_name = "mla_prolog_quant_demo"
    else:
        so_path = compile_shared_lib(X86_SRC)
        func_name = "mla_prolog_quant_x86"
    lib = ctypes.CDLL(so_path)
    func = getattr(lib, func_name)
    func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4
    func.restype = None

    q_out = np.zeros((T, OUT), dtype=np.float32)
    func(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_dq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_uq_qr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    ref = reference(x, w_dq, w_uq_qr)
    ok = np.allclose(q_out, ref, rtol=1e-4, atol=1e-5)
    print(f"q_out match: {ok}, max_err={np.max(np.abs(q_out - ref)):.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
