deepseekv32 examples
===================

This folder groups PTO frontends, x86 reference backends, and tests for the
DeepSeek V32 `*impl.py` demos.

Contents
--------
- `sparse_attention_antiquant/`: simplified attention core demo
- `sparse_flash_attention_quant/`: simplified flash-attention core demo
- `mla_indexer_prolog_quant/`: simplified MLA + indexer prolog demo
- `mla_prolog_quant/`: simplified MLA prolog demo
- `lightning_indexer_prolog_quant/`: simplified lightning indexer prolog demo
- `lightning_indexer_quant/`: simplified lightning indexer demo
- `unsupported.md`: summary of unsupported parts and reasons

Each subfolder contains:
- `pto_*.py`: PTO frontend (generates ARM/CUDA/Ascend/PTO code)
- `*_x86.c`: x86 reference backend (matches PTO frontend logic)
- `test_*.py`: CPU test (x86 uses x86 C, arm64 uses generated ARM C)

How to run
----------
Run a PTO frontend to generate backend code:
```bash
python /home/pan-boyi/work/pypto/pto-isa-liao/examples/deepseekv32/<impl>/pto_<impl>.py
```

Run the CPU test (auto-selects x86 vs arm64):
```bash
python /home/pan-boyi/work/pypto/pto-isa-liao/examples/deepseekv32/<impl>/test_<impl>.py
```

Examples
--------
Generate and test `sparse_attention_antiquant`:
```bash
python /home/pan-boyi/work/pypto/pto-isa-liao/examples/deepseekv32/sparse_attention_antiquant/pto_sparse_attention_antiquant.py
python /home/pan-boyi/work/pypto/pto-isa-liao/examples/deepseekv32/sparse_attention_antiquant/test_sparse_attention_antiquant.py
```

Generate and test `lightning_indexer_quant`:
```bash
python /home/pan-boyi/work/pypto/pto-isa-liao/examples/deepseekv32/lightning_indexer_quant/pto_lightning_indexer_quant.py
python /home/pan-boyi/work/pypto/pto-isa-liao/examples/deepseekv32/lightning_indexer_quant/test_lightning_indexer_quant.py
```
