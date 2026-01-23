| Module | Unsupported Part | Math Role | Reason |
|---|---|---|---|
| sparse_attention_antiquant_impl | gather_in_ub / cache dequant | gather INT8 cache and dequant to FP | No lowering for gather + packed cache layout |
| sparse_attention_antiquant_impl | complex reshape/concat | assemble q/k/v from packed cache | Non-contiguous view/reshape not lowered |
| sparse_flash_attention_quant_impl | flash attention online softmax | running max/sum update over blocks | Loop-dependent state/conditionals not lowered |
| sparse_flash_attention_quant_impl | gather_in_l1/ub | fetch cache blocks | No lowering for gather ops |
| sparse_flash_attention_quant_impl | complex dequant reshape | per-channel dequant and pack | Non-contiguous reshape/concat not lowered |
| mla_indexer_prolog_quant_impl | pipeline scheduling | device_sched_mode=2 fusion | Runtime scheduling not supported in PTO backend |
| mla_prolog_quant_impl | RoPE (rotate_half/rope_v2/rope_3d_v2) | rotary embedding | Missing rotate_half lowering and composite view/concat |
| mla_prolog_quant_impl | scatter_update | update key cache | No ARM64/CUDA lowering for TSCATTER |
| mla_prolog_quant_impl | NZ weight layout | matmul with NZ format | Layout-specific lowering not implemented |
| mla_prolog_quant_impl | quantization variants | per-channel INT8 quant | TCVT rounding modes not lowered |
| lightning_indexer_prolog_quant_impl | RoPE (rotate_half/rope_3d/rope_2d) | rotary embedding | Missing rotate_half lowering and composite view/concat |
| lightning_indexer_prolog_quant_impl | Hadamard + reshape/concat | head-wise mixing | Composite reshape/concat not lowered |
| lightning_indexer_prolog_quant_impl | LayerNorm (quant_layer_norm) | normalize key path | Reduction/broadcast path not fully lowered for this pattern |
| lightning_indexer_prolog_quant_impl | quantization (prolog_quant) | int8 quant + scale | TCVT lowering missing |
| lightning_indexer_prolog_quant_impl | scatter_update | update key cache | No ARM64/CUDA lowering for TSCATTER |
| lightning_indexer_quant_impl | topk_sort/merge/extract | multi-stage topk | No lowering for TopK ops |
| lightning_indexer_quant_impl | padding/partition logic | variable-length handling | Conditionals/loop control not lowered |
