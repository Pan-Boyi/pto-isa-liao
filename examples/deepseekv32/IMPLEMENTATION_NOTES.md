# indexer_prolog 在 pto-isa-liao 上的实现说明

## 一、已实现的子集

### 1. weights_tile (InCore)

- **对应 golden**：`weights = torch.matmul(x, w_idx_proj) * (n**-0.5) * (d**-0.5)`
- **PTO 实现**：`weights = (x @ w) / scale`，其中 `scale = sqrt(n*d)`，在 8×8 单 tile 测试中用 `scale=8` 近似 `sqrt(64)`。
- **使用的 PTO 指令**：`TLOAD`, `TMATMUL`, `TDIVS`, `TSTORE`。

### 2. layer_norm_tile (InCore)

- **对应 golden**：`k = layer_norm(k, layer_norm_gamma, layer_norm_beta)` 的**无 gamma/beta 版本**。
- **公式**：`out = (x - mean(x)) / sqrt(var(x) + eps)`。
- **使用的 PTO 指令**：`TLOAD`, `TROWSUM`, `TDIVS`, `TROWEXPANDSUB`, `TMUL`, `TROWSUM`, `TDIVS`, `TADDS`, `TSQRT`, `TROWEXPANDDIV`, `TSTORE`。
- **说明**：当前未接入 `gamma`、`beta`；`eps` 在最小测试中设为 0（因 `SLI` 对浮点常数的支持因后端而异）。

### 3. rope_tile (InCore)

- **对应 golden**：`single_rope`：`x*cos + rotate_half(x)*sin`；`rotate_half` 为 `[-x2|x1]`（沿最后一维分半，右半取负后与左半沿列拼接）。
- **PTO 实现**：通过扩展 ISA：`TEXTRACT`（子块提取）、`TCONCAT`（沿 axis 拼接），配合 `TNEG`、`TMUL`、`TADD`。
- **使用的 PTO 指令**：`TLOAD`, `TEXTRACT`, `TNEG`, `TCONCAT`, `TMUL`, `TADD`, `TSTORE`。
- **说明**：`TEXTRACT(dst, src, row, col)` 从 `src` 的 `(row, col)` 起按 `dst` 形状复制；`TCONCAT(dst, first, second, axis)`：axis=0 沿行堆叠 `[first;second]`，axis=1 沿列拼接 `[first|second]`（RoPE 用 axis=1）。

### 4. k_cache_scatter_update_tile (InCore) — **已实现**

- **对应 golden**：`scatter_update_pa_bsnd(k_cache, k_bsnd, cache_index, axis=-2)`，按 `cache_index[b,s]` 把 `k_bsnd` 分散写到 `k_cache` 的对应行。
- **PTO 实现**：`SCATTER_UPDATE` 高层接口：`k_cache[row_indices[i,0]*d + j] = k_tile[i,j]`（axis=-2，行索引在 `cache_index` 的 (rows,1) tile 中）。
- **使用的 PTO 指令**：`SCATTER_UPDATE`（在 ARM64/CUDA/Ascend 的 barrier 中生成 C 循环）；底层 ISA 亦提供 `TSCATTER`（tile→tile）、`MSCATTER`（tile→GM，线性下标）。
- **说明**：**非 int8**：仅写回 `k_cache`；**int8 量化**路径的 `k_scale_cache` 已由 `k_scale_cache_scatter_update_tile` 实现。

### 5. quant_int8_tile (InCore) — **已实现**

- **对应 golden**：`quant_int8`：`max_value=amax(|x|, dim=-1, keepdim)`，`scale_quant=127/max_value`，`y=round(x*scale_quant)→int8`，`scale_dequant=max_value/127`。
- **PTO 实现**：`TABS(abs_x,x)` → `TROWMAX(rowmax, abs_x)` → `TMAXS(max_safe, rowmax, 1e-6)`（防除零）→ `TEXPANDS(127)`、`TROWEXPANDDIV(scale_quant, scale_127_tile, max_safe)` → `TMUL(y, x, scale_quant)` → `TCVT(y_int8, y)`（F32→I8，CAST_RINT）→ `TDIVS(scale_dequant, max_safe, 127)`；双输出 `output_int8`、`output_scale`。
- **使用的 PTO 指令**：`TLOAD`, `TABS`, `TROWMAX`, `TMAXS`, `TEXPANDS`, `TROWEXPANDDIV`, `TMUL`, `TCVT`, `TDIVS`, `TSTORE`。
- **说明**：`TCVT` 已在 ARM64/CUDA/Ascend 的 `_gen_*_barrier_op` 中支持 F32→I8。codegen 对 (rows,1) 的 `TMAXS` 未生成，测试脚本通过 `patch_quant_int8_tmaxs` 在 TROWMAX 与后继 FUSED LOOP 之间插入 `max_safe[_row][0]=fmaxf(rowmax[_row][0],1e-6f)`。

### 6. k_scale_cache_scatter_update_tile (InCore) — **已实现**

- **对应 golden**：`scatter_update_pa_bsnd(k_scale_cache, k_scale_bsnd, cache_index, axis=-2)`，将 `scale_tile (rows,1)` 按 `cache_index` 写回 `k_scale_cache`。
- **PTO 实现**：与 `k_cache_scatter_update_tile` 相同 `SCATTER_UPDATE` 接口，`scale_tile` 形状 (rows,1)，`k_scale_cache` 为 F32（与 golden 的 fp16 比较时可在 Python 侧做类型转换）。

---

## 二、无法实现或仅能部分实现的部分

### 1. RoPE — **已实现**（见上一、3. rope_tile）

### 2. Q 路径（第一层线性：q_norm @ w_qb）— **CPU 已实现 fp32 回退；Ascend 未实现**

- **Golden / pypto**：
  - **deepseekv32_lightning_indexer_prolog_quant.indexer_prolog**：`q = matmul(q_norm.to(int32), w_idx_qb.to(int32))` 再 `* q_norm_scale * w_idx_qb_scale`（等价于先 dequant 再 fp32 matmul）。
  - **lightning_indexer_prolog_quant_impl**：Ascend 上 `pypto.matmul(q_norm, w_qb_in, pypto.DT_INT32)`，即 **int8×int8→int32** 原生支持。
- **PTO 实现**：
  - **CPU/ARM64**：**q_linear_fp32_tile** 已实现 **fp32 回退**：`q_norm`(I8)、`w`(I8) 经 TCVT(I8→F32) + rowexpandmul/colexpand+mul 做 dequant，再用 F32 TMATMUL。与 golden 的「先 dequant 再 fp32 matmul」等价。
  - **Ascend**：PTO 的 `TMATMUL` **无 int8 输入、int32 累加** 的定义与 codegen；Ascend 后端亦未接入 int8×int8→int32 的 matmul 原语。故 **Ascend 上 Q 的第一层线性（int8×int8→int32）当前无法在 PTO 中实现**，除非扩展 ISA 与 Ascend 后端。
- **结论**：**query**、**query_scale** 的最终输出仍需 golden/pypto 做 rope、hadamard、quant_int8；PTO 在 CPU 上已实现 Q 的**第一层线性**（q_linear_fp32_tile）；Ascend 上该段需继续使用 pypto 的 int8 matmul 或后续扩展 PTO。

### 3. scatter_update（按 cache_index 写回 K / K_scale）— **已实现**

- **k_cache**：`k_cache_scatter_update_tile`；**k_scale_cache**：`k_scale_cache_scatter_update_tile`。二者均使用 `SCATTER_UPDATE`，在 ARM64/CUDA/Ascend barrier 中生成 C 循环。

### 4. 多类型与多精度 (int8 / bf16 / fp16 / fp32) — **部分支持**

- **Golden**：`q_norm`(int8)、`w_idx_qb`(int8)、`q_norm_scale`(fp32)、`w_idx_qb_scale`(fp32)、`hadamard`(bf16)、`q`(bf16)、`q_scale`(fp16) 等混合运算与类型转换。
- **原因**：
  - PTO 的 `ElementType` 支持 I8/F16/F32/BF16 等，但 **`TMATMUL` 等对 int8 输入、累加在 int32、再 dequant 的路径**在 builder 与 codegen 中**没有现成套路**。
  - `TCVT` 存在于 pto-isa，但在 `PTOFunctionBuilder` 的便捷接口与 ARM64 融合/barrier 的 codegen 中**未完全打通**，尤其是多步 round/trunc 与 bf16/fp16 的互换。
- **结论**：单精度 fp32 的 matmul、LayerNorm、部分 scale 运算可做；int8  matmul + 多精度 dequant/quant 需在 builder 与 backend 上系统扩展。

### 5. 动态 loop 与 tiling（如 `lightning_indexer_prolog_quant_compute` 的 `loop_unroll`）— **部分支持**

- **Golden / pypto**：按 `t`、`head`、`chunk` 等做 `loop_unroll` 与 `view`/`assemble`。
- **原因**：
  - PTO 的 `for_loop`、`max_range`/`min_range`、`tile_levels` 可描述动态上界与分块，**Orchestration + CALL** 可调度多 InCore。
  - 但 **`view`/`assemble`**（按逻辑维度的切片与写回）在 PTO 中**没有对应原语**，只能通过 **TLOAD/TSTORE 的 row/col 偏移** 间接表达，且需事先知道布局与步长。
- **结论**：规则的分块和循环可做；复杂、与 pypto 一一对应的 `view`/`assemble` 无法直接实现。

---

## 三、合并版 indexer_prolog（单函数，CPU + Ascend）

**目标**：将原 7 个独立 InCore 的逻辑**写进同一函数**，生成**一份 CPU (ARM64) .c** 与**一份 Ascend .cpp**，并做端到端精度验证。

- **实现**：`create_indexer_prolog_combined_func()` 将 Q 路径（q_linear_fp32 → rope → hadamard → quant_int8）、K 路径（x@w_k → layer_norm → rope → hadamard → quant_int8 → scatter 写回 k_cache / k_scale_cache）、Weights（x@w_proj * 1/8）串在同一 InCore 内，通过 tile 复用控制规模。
- **代码生成**：`pto_indexer_prolog.py` 的 `main()` 仅对入口 `indexer_prolog` 做 `generate_arm64` 与 `generate_ascend`，输出 `output_arm64/indexer_prolog/indexer_prolog.c` 与 `output_ascend/indexer_prolog/indexer_prolog.cpp`。
- **测试**：`test_indexer_prolog_pto_cpu.py` 对合并版：修补 `scalef`/`inv_colsf`/`epsf` 与两处 TMAXS；编译 `indexer_prolog.c` 为 `indexer_prolog_cpu.so`；以 8×8 组合公式（q_linear→rope→hadamard→quant；k: x@w_k→ln→rope→hadamard→quant→scatter；weights）为 golden，对比 5 路输出：**query**(int8)、**query_scale**、**k_cache**、**k_scale_cache**、**weights**。端到端 L_inf 与 int8 一致性通过。
- **Ascend**：`generate_ascend` 会生成 `indexer_prolog.cpp`；Ascend 运行时与多 memref 的对接需在目标环境中单独验证。

---

## 四、总结

| 模块            | 实现情况 | 说明 |
|-----------------|----------|------|
| weights_tile    | 已实现   | `x@w` + scale，fp32，单 tile |
| layer_norm_tile | 已实现   | 无 gamma/beta，eps=0 的简化版 |
| rope_tile       | 已实现   | TEXTRACT+TCONCAT(axis=1)+TNEG 实现 single_rope |
| k_cache_scatter_update_tile | 已实现 | scatter_update(k_cache, k_tile, cache_index)，axis=-2 |
| k_scale_cache_scatter_update_tile | 已实现 | scatter_update(k_scale_cache, scale_tile, cache_index)，(rows,1) |
| quant_int8_tile | 已实现   | TABS,TROWMAX,TMAXS,127/max,TMUL,TCVT(F32→I8),scale_dequant；需 TMAXS 补丁 |
| **q_linear_fp32_tile (Q 第一层线性)** | **CPU 已实现** | fp32 回退：dequant(I8→F32) + F32 TMATMUL |
| **Q 路径 (Ascend int8×int8→int32)** | **无法实现** | PTO 无 I8×I8、累加 I32 的 TMATMUL；Ascend 未接入该原语 |
| **Q 路径 (query, query_scale 全链路)** | **部分** | 第一层线性 CPU 可做；rope、hadamard、quant_int8 仍靠 golden |
| 多类型/多精度   | 部分     | fp32 可行；int8 matmul 需扩展 |
| 动态 view/assemble | 部分  | 靠 TLOAD/TSTORE 偏移；无高级 view |

**合并版**（推荐）：`pto_indexer_prolog.py` 的 **`indexer_prolog`** 单函数将上述 7 段逻辑（q_linear_fp32、rope、hadamard、quant_int8；k: x@w_k、layer_norm、rope、hadamard、quant_int8、scatter；weights）全部写进同一 InCore，生成一份 CPU `.c` 与一份 Ascend `.cpp`，并由 `test_indexer_prolog_pto_cpu.py` 做 5 路输出的端到端精度验证。**Q 路径**在 CPU 上为 fp32 回退（dequant + F32 matmul）；Ascend 上 int8×int8→int32 未在 PTO 实现。

---

## 五、测试与已知问题

- **CPU 测试**：`test_indexer_prolog_pto_cpu.py` 针对**合并版** `indexer_prolog.c`：修补 `scalef`/`inv_colsf`/`epsf` 与两处 TMAXS；编译为 `indexer_prolog_cpu.so`（`-DPTO_CPU_SMOKE_RUNNER`）；用 ctypes 调 `pto_launch` 传入 17 个 memref，与 8×8 组合公式 golden 对比 5 路输出（query、query_scale、k_cache、k_scale_cache、weights）的端到端 L_inf 与 int8 一致性。需本机为 ARM64 且可编译 `arm_neon.h`，否则编译会跳过。
- **标量浮点与 TMAXS codegen**：ARM64 对 `TDIVS`/`TADDS` 等标量会生成 `scalef`、`inv_colsf`、`epsf` 等未定义符号，测试脚本做替换：`scalef`→`(float)scale`，`inv_colsf`→`(float)inv_cols`，`epsf`→`(float)eps`。codegen 对 (rows,1) 的 `TMAXS` 未生成，测试脚本在合并版两处 TROWMAX 与 FUSED LOOP 之间插入 `max_safe[_row][0]=fmaxf(rowmax[_row][0],1e-6f)`。

---

## 六、pto-isa 与 TCONCAT：底层接口分析

在 **pto-isa** 仓中，Ascend 后端与 CONCAT 相关的接口主要有：

- **TEXTRACT(dst, src, indexRow, indexCol)**  
  - 语义：从 `src` 的 `(indexRow, indexCol)` 起复制一块与 `dst` 同形状的子块到 `dst`。  
  - 方向：Mat/类似 → 子块（Left/Right/Acc 等）。  
  - **可用于**：从大块中取子块（本仓已用于 RoPE 的 left/right 切分）。

- **TINSERT(dst, src, indexRow, indexCol)**  
  - 语义：将 `src` 写入 `dst` 的 `(indexRow, indexCol)` 起始区域。  
  - 约束：`DstTileData::Loc == TileType::Mat`；`TInsertAccToMat` / `CheckTMovAccToMat` 要求 **src 为 Acc、dst 为 Mat**，即 **Acc → Mat**。  
  - **结论**：TINSERT 是 **Acc 到 Mat** 的写回，不能直接用于两个 **Mat  tile** 的拼接。若抽象层能把 `first`、`second` 放在 Acc，再两次 TINSERT 写入同一 Mat 的 `(0,0)` 与 `(0, c1)` 或 `(r1, 0)`，理论上可表达 CONCAT；当前 pto-isa-liao 的 tile 抽象未区分 Acc/Mat，且 Ascend codegen 也未做该 lowering。

- **TCONCAT 的当前实现**  
  - pto-isa 中 **无** TCONCAT 原语。  
  - 在 pto-isa-liao 的 ARM64 / CUDA / Ascend 中，TCONCAT 均通过 **C 循环** 实现：按 `axis` 将 `first`、`second` 拷贝到 `dst` 的对应区域。  
  - 若未来在 pto-isa 中增加 Mat→Mat 的 block 写或通用 TCONCAT，可再改为调用底层原语。
