"""
PTO frontend demo for mla_indexer_prolog_quant (simplified).

Implements:
  q_norm = x @ w_dq
  q_out = q_norm @ w_qb
  weights = x @ w_proj

Skipped parts:
  - pipeline scheduling between MLA and lightning indexer
  - quantization / RoPE / cache scatter
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pto_compile import PTOFunctionBuilder, generate_all_backends, BACKENDS
from pto_isa_definition import ElementType, MemorySpace


def mla_indexer_prolog_quant_demo(t=4, h=4, q_rank=4, out=4, head_num=2):
    return (PTOFunctionBuilder("mla_indexer_prolog_quant_demo")
        .tile("x", t, h, ElementType.F32)
        .tile("w_dq", h, q_rank, ElementType.F32)
        .tile("w_qb", q_rank, out, ElementType.F32)
        .tile("w_proj", h, head_num, ElementType.F32)
        .tile("q_norm", t, q_rank, ElementType.F32)
        .tile("q_out", t, out, ElementType.F32)
        .tile("weights", t, head_num, ElementType.F32)

        .memref("x_mem", MemorySpace.GM, ElementType.F32)
        .memref("w_dq_mem", MemorySpace.GM, ElementType.F32)
        .memref("w_qb_mem", MemorySpace.GM, ElementType.F32)
        .memref("w_proj_mem", MemorySpace.GM, ElementType.F32)
        .memref("q_out_mem", MemorySpace.GM, ElementType.F32)
        .memref("weights_mem", MemorySpace.GM, ElementType.F32)

        .load("x", "x_mem", 0, 0)
        .load("w_dq", "w_dq_mem", 0, 0)
        .load("w_qb", "w_qb_mem", 0, 0)
        .load("w_proj", "w_proj_mem", 0, 0)

        .matmul("q_norm", "x", "w_dq")
        .matmul("q_out", "q_norm", "w_qb")
        .matmul("weights", "x", "w_proj")

        .store("q_out", "q_out_mem", 0, 0)
        .store("weights", "weights_mem", 0, 0)

        .build())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_prefix = "mla_indexer_prolog_quant_demo"

    program = mla_indexer_prolog_quant_demo()
    results = generate_all_backends(
        program,
        output_prefix,
        output_base_dir=script_dir,
        enable_fusion=True
    )

    print("=" * 70)
    print("MLA Indexer Prolog Quant Demo Generation Complete")
    print(f"Generated files: {len(results)}")
    print("Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{output_prefix}/")
    print("  - output_pto/{output_prefix}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
