"""
PTO frontend demo for mla_prolog_quant (simplified).

Implements:
  x_proj = x @ w_dq
  rms = sqrt(mean(x_proj^2) + eps)
  x_norm = x_proj / rms
  q_out = x_norm @ w_uq_qr

Skipped parts:
  - RoPE (rope_v2/rope_3d_v2)
  - quantization and cache scatter
  - NZ weight layouts
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pto_compile import PTOFunctionBuilder, generate_all_backends, BACKENDS
from pto_isa_definition import ElementType, MemorySpace


def mla_prolog_quant_demo(t=4, h=4, out=4, eps=1e-6):
    return (PTOFunctionBuilder("mla_prolog_quant_demo")
        .tile("x", t, h, ElementType.F32)
        .tile("w_dq", h, h, ElementType.F32)
        .tile("w_uq_qr", h, out, ElementType.F32)
        .tile("x_proj", t, h, ElementType.F32)
        .tile("x_sq", t, h, ElementType.F32)
        .tile("sum_sq", t, 1, ElementType.F32)
        .tile("mean_sq", t, 1, ElementType.F32)
        .tile("mean_eps", t, 1, ElementType.F32)
        .tile("rms", t, 1, ElementType.F32)
        .tile("x_norm", t, h, ElementType.F32)
        .tile("q_out", t, out, ElementType.F32)

        .memref("x_mem", MemorySpace.GM, ElementType.F32)
        .memref("w_dq_mem", MemorySpace.GM, ElementType.F32)
        .memref("w_uq_qr_mem", MemorySpace.GM, ElementType.F32)
        .memref("q_out_mem", MemorySpace.GM, ElementType.F32)

        .load("x", "x_mem", 0, 0)
        .load("w_dq", "w_dq_mem", 0, 0)
        .load("w_uq_qr", "w_uq_qr_mem", 0, 0)

        .matmul("x_proj", "x", "w_dq")
        .mul("x_sq", "x_proj", "x_proj")
        .rowsum("sum_sq", "x_sq")
        .divs("mean_sq", "sum_sq", float(h))
        .adds("mean_eps", "mean_sq", eps)
        .sqrt("rms", "mean_eps")
        .rowexpanddiv("x_norm", "x_proj", "rms")

        .matmul("q_out", "x_norm", "w_uq_qr")
        .store("q_out", "q_out_mem", 0, 0)

        .build())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_prefix = "mla_prolog_quant_demo"

    program = mla_prolog_quant_demo()
    results = generate_all_backends(
        program,
        output_prefix,
        output_base_dir=script_dir,
        enable_fusion=True
    )

    print("=" * 70)
    print("MLA Prolog Quant Demo Generation Complete")
    print(f"Generated files: {len(results)}")
    print("Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{output_prefix}/")
    print("  - output_pto/{output_prefix}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
