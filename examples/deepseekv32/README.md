# DeepSeek V32 indexer_prolog（PTO 子集）

本目录包含在 **pto-isa-liao** 上实现的 indexer_prolog 子集（`weights_tile`、`layer_norm_tile`）及生成、测试脚本。  
路径均相对 **pto-isa-liao 根目录** 或 **本目录 `deepseekv32`**。

---

## 目录结构

```
examples/deepseekv32/
├── README.md                  # 本说明
├── IMPLEMENTATION_NOTES.md     # 实现范围与无法实现部分说明
├── pto_indexer_prolog.py       # 代码生成脚本
├── test_indexer_prolog_pto_cpu.py  # CPU 正确性与精度测试
└── output_arm64/
    └── indexer_prolog/         # 生成的 ARM64 C 与 .so（运行后产生）
        ├── weights_tile.c
        ├── layer_norm_tile.c
        ├── weights_tile_cpu.so
        └── layer_norm_tile_cpu.so
```

---

## 如何生成代码

在 **pto-isa-liao 根目录** 下执行：

```bash
python examples/deepseekv32/pto_indexer_prolog.py
```

- 生成结果写入：`examples/deepseekv32/output_arm64/indexer_prolog/`
- 产出文件：`weights_tile.c`、`layer_norm_tile.c`

---

## 如何执行测试脚本

在 **pto-isa-liao 根目录** 下执行：

```bash
python examples/deepseekv32/test_indexer_prolog_pto_cpu.py
```

测试脚本会：

1. **若尚未生成 C**：先调用 `pto_indexer_prolog.py` 生成 `weights_tile.c`、`layer_norm_tile.c`
2. **修补生成代码**：将 `scalef`、`inv_colsf`、`epsf` 等替换为 `(float)scale`、`(float)inv_cols`、`(float)eps`（因当前 codegen 会生成未定义符号）
3. **编译**：用 `clang -DPTO_CPU_SMOKE_RUNNER -lm` 生成 `weights_tile_cpu.so`、`layer_norm_tile_cpu.so`，输出到 `output_arm64/indexer_prolog/`
4. **运行并对比**：通过 ctypes 调用 `pto_launch`，与 golden 比较 L_inf

**环境要求**：

- 本机为 **ARM64** 且能编译 `arm_neon.h`，否则相关编译会跳过
- 可选：安装 `numpy`；未安装时使用 `random.Random` 与纯 Python golden

---

## 一行命令：生成 + 测试

```bash
# 在 pto-isa-liao 根目录
python examples/deepseekv32/pto_indexer_prolog.py && python examples/deepseekv32/test_indexer_prolog_pto_cpu.py
```

---

## 路径说明（相对路径）

| 用途         | 路径（相对 pto-isa-liao 根目录）        |
|--------------|----------------------------------------|
| 代码生成入口 | `examples/deepseekv32/pto_indexer_prolog.py` |
| 测试入口     | `examples/deepseekv32/test_indexer_prolog_pto_cpu.py` |
| 生成 C 输出  | `examples/deepseekv32/output_arm64/indexer_prolog/` |
| 实现与限制   | `examples/deepseekv32/IMPLEMENTATION_NOTES.md` |
