# 如何运行 LLaMA 示例

本文档说明如何在 ARM64 平台上运行 PTO ISA Compiler 生成的 LLaMA 7B 解码层示例，并验证计算精度的正确性。

## 目录结构

### 源代码文件

| 文件路径 | 说明 |
|---------|------|
| `examples/pto_llama7B_dynamic.py` | LLaMA 7B 解码层的 PTO DSL 源代码 |
| `examples/test_llama_runtime.c` | 简化版 LLaMA 层测试程序（精度验证） |
| `examples/test_llama_performance.c` | LLaMA 层性能测试程序 |
| `pto_runtime.h` | PTO 运行时头文件 |
| `pto_runtime.c` | PTO 运行时实现（任务调度、依赖跟踪） |

### 生成的代码目录

```
examples/
├── output_arm64/llama7b/          # ARM64 NEON 代码
│   ├── llama_layer_dynamic_orchestration.c   # 编排函数（任务图构建）
│   ├── llama_layer_dynamic_task_graph.txt    # 任务图文本描述
│   ├── llama_layer_dynamic_task_graph.pdf    # 任务图可视化
│   ├── rmsnorm_tile.c, rmsnorm_tile_64.c     # RMSNorm InCore 函数
│   ├── tile_matmul.c, tile_matmul_64.c       # 矩阵乘法 InCore 函数
│   ├── softmax_tile.c, softmax_tile_64.c     # Softmax InCore 函数
│   ├── swiglu_tile.c, swiglu_tile_64.c       # SwiGLU InCore 函数
│   ├── rope_tile.c, rope_tile_64.c           # RoPE InCore 函数
│   ├── flash_attn_*.c                        # Flash Attention InCore 函数
│   └── ...                                   # 其他 InCore 函数
│
├── output_cuda/llama7b/           # CUDA 代码
│   ├── llama_layer_dynamic_orchestration.c   # 编排函数（平台无关）
│   ├── *.cu                                  # CUDA InCore 内核
│   └── ...
│
├── output_ascend910b/llama7b/     # 华为 Ascend 910B 代码
│   ├── llama_layer_dynamic_orchestration.c   # 编排函数（平台无关）
│   ├── *.cpp                                 # Ascend C InCore 函数
│   └── ...
│
└── output_pto/llama7b/            # PTO 中间表示
    └── llama7b_layer.pto          # PTO 汇编代码
```

## LLaMA 7B 层架构

```
输入: [batch=1, seq_len, hidden_dim=4096]

1. Pre-Attention (并行处理所有 tile)
   ├── RMSNorm
   ├── Q = MatMul(norm, Wq)
   ├── K = MatMul(norm, Wk)
   ├── V = MatMul(norm, Wv)
   ├── Q_rope = RoPE(Q)
   └── K_rope = RoPE(K)

2. Flash Attention (跨 tile 依赖)
   └── For each Q tile:
       └── For each KV tile:
           ├── S = Q @ K^T / sqrt(d)
           ├── Online Softmax Update
           └── O += P @ V
       └── Normalize: O = O / L

3. Post-Attention (并行处理所有 tile)
   ├── O_proj = MatMul(attn_out, Wo)
   ├── Hidden = O_proj + Input  (残差连接)
   ├── RMSNorm
   ├── Gate = MatMul(norm, W_gate)
   ├── Up = MatMul(norm, W_up)
   ├── SwiGLU: out = SiLU(Gate) * Up
   ├── Down = MatMul(swiglu, W_down)
   └── Output = Down + Hidden  (残差连接)

输出: [batch=1, seq_len, hidden_dim=4096]
```

## 在 ARM64 上运行

### 1. 生成代码

```bash
cd /Users/mac/Documents/PTO_ISA_Compiler/examples
python pto_llama7B_dynamic.py
```

**预期输出：**
```
======================================================================
PTO LLaMA 7B Layer - Flash Attention (seq_len=1024)
======================================================================

Configuration:
  Batch Size: 1 (fixed)
  Hidden Dim: 4096
  Num Heads: 32
  Head Dim: 128
  Intermediate Dim: 11008
  Target ISA: ascend910b
  Standard Tile: 32x128 = 4096 elements (16.0 KB)

Flash Attention Configuration:
  SRAM Size: 256 KB
  Flash Block Size (Br=Bc): 64

Flash Attention Memory Analysis:
  Q block: 64x128 = 32.0 KB
  K block: 64x128 = 32.0 KB
  V block: 64x128 = 32.0 KB
  S block: 64x64 = 16.0 KB
  O block: 64x128 = 32.0 KB
  Total (with reuse): 160.0 KB / 256 KB
  Fits in SRAM: ✓ YES

Creating LLaMA 7B Module
...

======================================================================
Code Generation Complete!
======================================================================
```

### 2. 编译测试程序

```bash
cd /Users/mac/Documents/PTO_ISA_Compiler/examples

# 编译简化版 LLaMA 层测试（精度验证）
gcc -O2 -I.. -o test_llama_runtime test_llama_runtime.c -lpthread -lm

# 编译性能测试
gcc -O2 -I.. -o test_llama_performance test_llama_performance.c -lpthread -lm
```

### 3. 运行精度验证测试

```bash
# 用法: ./test_llama_runtime [seq_len] [num_workers] [threshold]
# 默认: seq_len=128, num_workers=4, threshold=0

# 测试序列长度 128，4 个工作线程
./test_llama_runtime 128 4
```

**预期输出：**
```
================================================================================
PTO Runtime - Simplified LLaMA Layer Test
================================================================================
Configuration:
  Sequence Length: 128
  Hidden Dim:      128
  Tile Size:       32 x 128
  Num Tiles:       4
  Workers:         4
  Threshold:       0 (safe)
  Total Elements:  16384
  Tasks:           16 (4 per tile)
================================================================================

Computing reference...
Executing with runtime_entry_arm64...
[PTO Runtime] ========================================
[PTO Runtime] ARM64 Multi-threaded Execution
[PTO Runtime] Workers: 4
[PTO Runtime] Execution mode: wait for orchestration
[PTO Runtime] ========================================
[PTO Runtime] Spawning 4 worker threads...
[PTO Runtime] Created worker thread 0
[PTO Runtime] Created worker thread 1
[PTO Runtime] Created worker thread 2
[PTO Runtime] Created worker thread 3
[PTO Runtime] Workers started, now building task graph...
[PTO Runtime] Building task graph...
[Orchestration] Building LLaMA task graph for 4 tiles...
[Orchestration] Task graph complete: 4 tiles x 4 ops = 16 tasks
[PTO Runtime] Task graph built: 16 tasks
[PTO Runtime] Executing tasks...
[PTO Runtime] All 16 tasks completed!
[PTO Runtime] Shutting down workers...
[PTO Runtime] ========================================
[PTO Runtime] Execution Statistics
[PTO Runtime]   Total tasks: 16
[PTO Runtime]   Completed:   16
[PTO Runtime]   Workers:     4
[PTO Runtime] ========================================

Verifying results...

================================================================================
Results
================================================================================
  Execution time:  103.26 ms
  Tasks/second:    155
  Verification:    PASSED ✓
================================================================================

Sample output (first tile, rows 0-2):
  Row 0: [-1.3161, 0.0602, 0.6109, -0.6175, -0.3100, -0.7701, 1.2947, 0.0333, ...]
  Row 1: [0.2591, -0.4024, -0.6439, -0.2984, -0.4686, 0.2974, -0.8113, -0.5865, ...]
  Row 2: [0.4386, 0.2065, 0.3172, -0.9764, 0.6942, 0.7825, 1.2968, -0.1830, ...]

Reference (first tile, rows 0-2):
  Row 0: [-1.3161, 0.0602, 0.6109, -0.6175, -0.3100, -0.7701, 1.2947, 0.0333, ...]
  Row 1: [0.2591, -0.4024, -0.6439, -0.2984, -0.4686, 0.2974, -0.8113, -0.5865, ...]
  Row 2: [0.4386, 0.2065, 0.3172, -0.9764, 0.6942, 0.7825, 1.2968, -0.1830, ...]
```

**验证结果：** 输出显示 `Verification: PASSED ✓`，表示 PTO 运行时计算结果与参考实现完全一致。

### 4. 测试更长序列

```bash
# 序列长度 1024，8 个工作线程
./test_llama_runtime 1024 8
```

**预期输出：**
```
================================================================================
PTO Runtime - Simplified LLaMA Layer Test
================================================================================
Configuration:
  Sequence Length: 1024
  Hidden Dim:      128
  Tile Size:       32 x 128
  Num Tiles:       32
  Workers:         8
  Threshold:       0 (safe)
  Total Elements:  131072
  Tasks:           128 (4 per tile)
================================================================================

Computing reference...
Executing with runtime_entry_arm64...
...
[PTO Runtime] All 128 tasks completed!
...

================================================================================
Results
================================================================================
  Execution time:  69.63 ms
  Tasks/second:    1838
  Verification:    PASSED ✓
================================================================================
```

### 5. 运行性能测试

```bash
./test_llama_performance
```

**预期输出：**
```
====================================================================
LLaMA Layer Orchestration Performance Test
WITH ADAPTIVE TILE OPTIMIZATION (64-row tiles, scale=2x)
====================================================================

Configuration:
  Base Tile Size: 32 x 128 (32 rows)
  Adaptive Tile Size: 64 x 128 (64 rows, scale=2x)
  Sequence Lengths: 1K to 16K

====================================================================
SeqLen   Tiles    ActIter  Tasks      NoAdapt    Build(ms)    Tasks/ms   Memory       Saved   
--------------------------------------------------------------------
1024     32       16       1024       3584       0.171        5988.3     2.81 MB      71.4%
2048     64       32       3584       13312      0.650        5513.8     9.81 MB      73.1%
4096     128      64       13312      51200      2.472        5385.1     36.41 MB     74.0%
8192     256      128      51200      200704     8.858        5780.1     139.95 MB    74.5%
12288    384      192      113664     448512     19.652       5783.8     310.61 MB    74.7%
16384    512      256      200704     794624     35.577       5641.4     548.39 MB    74.7%
====================================================================

CONCLUSION: Adaptive tile optimization provides ~75% reduction in task count
            for Flash Attention's N^2 dependency pattern.
```

## 精度验证说明

测试程序通过以下方式验证计算精度：

1. **参考实现**：使用标准 C 代码实现 LLaMA 层的每个操作
2. **PTO 运行时执行**：使用 PTO 运行时的任务调度系统执行相同计算
3. **元素级比较**：比较两者的输出，使用相对容差 `1e-4`

```c
int verify_llama(float* output, float* reference, int total_elements) {
    int errors = 0;
    float tolerance = 1e-4f;  // 相对容差
    
    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(output[i] - reference[i]);
        float rel_diff = diff / (fabsf(reference[i]) + 1e-8f);
        
        if (rel_diff > tolerance && diff > tolerance) {
            errors++;
        }
    }
    return errors;
}
```

## 关键技术特性

### 1. 动态 Tiling
- 支持动态序列长度（最高 128K tokens）
- 自适应 tile 大小（32 行 / 64 行）
- 二进制展开优化循环迭代

### 2. Flash Attention
- 在 256KB SRAM 内完成注意力计算
- 使用在线 Softmax 算法
- 支持长序列的分块计算

### 3. 多线程执行
- 基于任务图的依赖调度
- 支持配置工作线程数
- 自动依赖跟踪和并行执行

### 4. 平台无关编排
- 编排函数使用纯 C 代码
- 可在 ARM64、CUDA、Ascend 上运行
- InCore 函数根据目标平台生成

## 常见问题

### Q: 如何增加序列长度？
A: 修改 `test_llama_runtime` 的第一个参数：
```bash
./test_llama_runtime 4096 8  # 序列长度 4096
```

### Q: 如何增加工作线程数？
A: 修改第二个参数：
```bash
./test_llama_runtime 1024 16  # 16 个工作线程
```

### Q: 验证失败怎么办？
A: 检查浮点精度设置，或增加容差值。通常 `1e-4` 的相对容差对于 FP32 计算是足够的。

### Q: 如何查看任务图？
A: 查看生成的 `.txt` 或 `.pdf` 文件：
```bash
cat examples/output_arm64/llama7b/llama_layer_dynamic_task_graph.txt
open examples/output_arm64/llama7b/llama_layer_dynamic_task_graph.pdf
```
