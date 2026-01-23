"""
PTO frontend demo for lightning_indexer_prolog_quant (simplified).

Implements:
  q = q_norm @ w_qb
  q = q * q_norm_scale * w_qb_scale
  weights = x @ w_proj * scale

Skipped parts:
  - RoPE / Hadamard
  - quantization (TCVT)
  - cache scatter update (TSCATTER)
  - layer norm and key path
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pto_compile import PTOFunctionBuilder, generate_all_backends, BACKENDS
from pto_isa_definition import ElementType, MemorySpace


def lightning_indexer_prolog_quant_demo(
    t=4,
    q_lora_rank=4,
    h=4,
    head_num=2,
    head_dim=2,
):
    out_dim = head_num * head_dim
    weight_scale = 1.0 / math.sqrt(head_num * head_dim)

    return (PTOFunctionBuilder("lightning_indexer_prolog_quant_demo")
        .tile("q_norm", t, q_lora_rank, ElementType.F32)
        .tile("q_norm_scale", t, 1, ElementType.F32)
        .tile("w_qb", q_lora_rank, out_dim, ElementType.F32)
        .tile("w_qb_scale", 1, out_dim, ElementType.F32)
        .tile("w_qb_scale_expand", t, out_dim, ElementType.F32)
        .tile("q_matmul", t, out_dim, ElementType.F32)
        .tile("q_scaled_row", t, out_dim, ElementType.F32)
        .tile("q_out", t, out_dim, ElementType.F32)

        .tile("x", t, h, ElementType.F32)
        .tile("w_proj", h, head_num, ElementType.F32)
        .tile("weights_raw", t, head_num, ElementType.F32)
        .tile("weights_out", t, head_num, ElementType.F32)

        .memref("q_norm_mem", MemorySpace.GM, ElementType.F32)
        .memref("q_norm_scale_mem", MemorySpace.GM, ElementType.F32)
        .memref("w_qb_mem", MemorySpace.GM, ElementType.F32)
        .memref("w_qb_scale_mem", MemorySpace.GM, ElementType.F32)
        .memref("x_mem", MemorySpace.GM, ElementType.F32)
        .memref("w_proj_mem", MemorySpace.GM, ElementType.F32)
        .memref("q_out_mem", MemorySpace.GM, ElementType.F32)
        .memref("weights_out_mem", MemorySpace.GM, ElementType.F32)

        .load("q_norm", "q_norm_mem", 0, 0)
        .load("q_norm_scale", "q_norm_scale_mem", 0, 0)
        .load("w_qb", "w_qb_mem", 0, 0)
        .load("w_qb_scale", "w_qb_scale_mem", 0, 0)
        .load("x", "x_mem", 0, 0)
        .load("w_proj", "w_proj_mem", 0, 0)

        .matmul("q_matmul", "q_norm", "w_qb")
        .rowexpandmul("q_scaled_row", "q_matmul", "q_norm_scale")
        .colexpand("w_qb_scale_expand", "w_qb_scale")
        .mul("q_out", "q_scaled_row", "w_qb_scale_expand")

        .matmul("weights_raw", "x", "w_proj")
        .muls("weights_out", "weights_raw", weight_scale)

        .store("q_out", "q_out_mem", 0, 0)
        .store("weights_out", "weights_out_mem", 0, 0)

        .build())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_prefix = "lightning_indexer_prolog_quant_demo"

    program = lightning_indexer_prolog_quant_demo()
    results = generate_all_backends(
        program,
        output_prefix,
        output_base_dir=script_dir,
        enable_fusion=True
    )

    print("=" * 70)
    print("Lightning Indexer Prolog Quant Demo Generation Complete")
    print(f"Generated files: {len(results)}")
    print("Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{output_prefix}/")
    print("  - output_pto/{output_prefix}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
