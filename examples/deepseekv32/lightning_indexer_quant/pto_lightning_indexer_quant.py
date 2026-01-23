"""
PTO frontend demo for lightning_indexer_quant (simplified).

Implements:
  scores = Q @ K^T

Skipped parts:
  - TopK hierarchy (topk_sort/merge/extract)
  - dequantization with scales
  - variable-length padding/partition logic
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pto_compile import PTOFunctionBuilder, generate_all_backends, BACKENDS
from pto_isa_definition import ElementType, MemorySpace


def lightning_indexer_quant_demo(t=4, d=4):
    return (PTOFunctionBuilder("lightning_indexer_quant_demo")
        .tile("q", t, d, ElementType.F32)
        .tile("k", t, d, ElementType.F32)
        .tile("k_t", d, t, ElementType.F32)
        .tile("scores", t, t, ElementType.F32)

        .memref("q_mem", MemorySpace.GM, ElementType.F32)
        .memref("k_mem", MemorySpace.GM, ElementType.F32)
        .memref("scores_mem", MemorySpace.GM, ElementType.F32)

        .load("q", "q_mem", 0, 0)
        .load("k", "k_mem", 0, 0)

        .transpose("k_t", "k")
        .matmul("scores", "q", "k_t")
        .store("scores", "scores_mem", 0, 0)

        .build())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_prefix = "lightning_indexer_quant_demo"

    program = lightning_indexer_quant_demo()
    results = generate_all_backends(
        program,
        output_prefix,
        output_base_dir=script_dir,
        enable_fusion=True
    )

    print("=" * 70)
    print("Lightning Indexer Quant Demo Generation Complete")
    print(f"Generated files: {len(results)}")
    print("Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{output_prefix}/")
    print("  - output_pto/{output_prefix}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
