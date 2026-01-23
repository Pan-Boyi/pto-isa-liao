"""
PTO frontend demo for sparse_flash_attention_quant (simplified).

Implements standard attention core with scaling:
  scores = (Q @ K^T) * scale
  probs = softmax(scores)
  out = probs @ V

Skipped parts:
  - flash attention online softmax updates
  - gather_in_l1/ub and dequantization from cache
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pto_compile import PTOFunctionBuilder, generate_all_backends, BACKENDS
from pto_isa_definition import ElementType, MemorySpace


def sparse_flash_attention_quant_demo(tq=4, tk=4, d=4):
    scale = 1.0 / math.sqrt(d)
    return (PTOFunctionBuilder("sparse_flash_attention_quant_demo")
        .tile("q", tq, d, ElementType.F32)
        .tile("k", tk, d, ElementType.F32)
        .tile("v", tk, d, ElementType.F32)
        .tile("k_t", d, tk, ElementType.F32)
        .tile("scores", tq, tk, ElementType.F32)
        .tile("scaled_scores", tq, tk, ElementType.F32)
        .tile("row_max", tq, 1, ElementType.F32)
        .tile("shifted", tq, tk, ElementType.F32)
        .tile("exp_scores", tq, tk, ElementType.F32)
        .tile("sum_exp", tq, 1, ElementType.F32)
        .tile("probs", tq, tk, ElementType.F32)
        .tile("out", tq, d, ElementType.F32)

        .memref("q_mem", MemorySpace.GM, ElementType.F32)
        .memref("k_mem", MemorySpace.GM, ElementType.F32)
        .memref("v_mem", MemorySpace.GM, ElementType.F32)
        .memref("out_mem", MemorySpace.GM, ElementType.F32)

        .load("q", "q_mem", 0, 0)
        .load("k", "k_mem", 0, 0)
        .load("v", "v_mem", 0, 0)

        .transpose("k_t", "k")
        .matmul("scores", "q", "k_t")
        .muls("scaled_scores", "scores", scale)

        .rowmax("row_max", "scaled_scores")
        .rowexpandsub("shifted", "scaled_scores", "row_max")
        .exp("exp_scores", "shifted")
        .rowsum("sum_exp", "exp_scores")
        .rowexpanddiv("probs", "exp_scores", "sum_exp")

        .matmul("out", "probs", "v")
        .store("out", "out_mem", 0, 0)

        .build())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_prefix = "sparse_flash_attention_quant_demo"

    program = sparse_flash_attention_quant_demo()
    results = generate_all_backends(
        program,
        output_prefix,
        output_base_dir=script_dir,
        enable_fusion=True
    )

    print("=" * 70)
    print("Sparse Flash Attention Quant Demo Generation Complete")
    print(f"Generated files: {len(results)}")
    print("Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{output_prefix}/")
    print("  - output_pto/{output_prefix}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
