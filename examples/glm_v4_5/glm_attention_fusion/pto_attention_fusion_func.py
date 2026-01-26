"""
PTO Attention Fusion Function Implementation

This module implements the fused attention operation from glm_attention_fusion.py
using pto-isa-liao framework. The implementation includes:

1. Input LayerNorm with residual connection
2. Input quantization
3. Quantized QKV matrix multiplication
4. Q/K LayerNorm
5. Rotary Position Embedding (RoPE)
6. Flash Attention with paged KV cache

This is a simplified version focusing on the core logic, adapted from the pypto
implementation in glm_attention_fusion.py.
"""

import os
import sys

# Add src directory to path
_example_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(_example_dir)))
_src_dir = os.path.join(_root_dir, 'src')
sys.path.insert(0, _src_dir)

from compile.pto_compile import PTOFunctionBuilder, PTOModule, MultiBackendCodeGenerator
from isa_definition.pto_isa_definition import ElementType, MemorySpace

# =============================================================================
# Configuration
# =============================================================================

# Default tile sizes
DEFAULT_BS_TILE = 8  # Batch size tile
DEFAULT_HIDDEN_SIZE = 5120  # Hidden dimension
DEFAULT_HEAD_SIZE = 128  # Head dimension
DEFAULT_Q_NUM_HEADS = 12  # Number of query heads
DEFAULT_KV_NUM_HEADS = 1  # Number of key/value heads
DEFAULT_TOTAL_HEAD_SIZE = 1792  # Total head size (q + k + v)
DEFAULT_ROTARY_DIM = 64  # Rotary embedding dimension (half_rotary_dim * 2)

# Data types
DTYPE_FP32 = ElementType.F32
DTYPE_FP16 = ElementType.F16
DTYPE_INT8 = ElementType.I8
DTYPE_INT32 = ElementType.I32

# Softmax scale (1/sqrt(head_dim))
SOFTMAX_SCALE = 1.0 / (DEFAULT_HEAD_SIZE ** 0.5)

# =============================================================================
# InCore Functions: Tile-level Operations
# =============================================================================

def create_rms_norm_bias_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
    """
    InCore: RMS Normalization with bias.
    
    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma + bias
    
    Input: [rows, cols]
    Output: [rows, cols]
    """
    return (PTOFunctionBuilder("rms_norm_bias")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("x_sq", rows, cols, dtype)
        .tile("row_sum", rows, 1, dtype)
        .tile("row_mean", rows, 1, dtype)
        .tile("row_rsqrt", rows, 1, dtype)
        .tile("x_norm", rows, cols, dtype)
        .tile("gamma", rows, cols, dtype)
        .tile("bias", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .scalar("eps", dtype)
        .scalar("inv_cols", dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("gamma_weight", MemorySpace.GM, dtype)
        .memref("bias_weight", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .load("gamma", "gamma_weight", 0, 0)
        .load("bias", "bias_weight", 0, 0)
        # x^2
        .mul("x_sq", "x", "x")
        # Row sum of x^2
        .rowsum("row_sum", "x_sq")
        # Mean = sum / cols
        .scalar_li("inv_cols", 1.0 / cols)
        .muls("row_mean", "row_sum", "inv_cols")
        # Add epsilon
        .scalar_li("eps", 1e-5)
        .adds("row_mean", "row_mean", "eps")
        # rsqrt(mean + eps)
        .rsqrt("row_rsqrt", "row_mean")
        # x * rsqrt(...)
        .rowexpandmul("x_norm", "x", "row_rsqrt")
        # Multiply by gamma
        .mul("result", "x_norm", "gamma")
        # Add bias
        .add("result", "result", "bias")
        .store("result", "output", 0, 0)
        .build())


# Note: RoPE function is complex and requires 3D tensor handling
# For now, we'll skip it in the simplified version and focus on Flash Attention
# RoPE can be added later when needed


# Reuse existing InCore functions from pto_ifa_func.py
# We'll import them or redefine them here

def create_qk_matmul_func(q_rows=DEFAULT_BS_TILE, q_cols=DEFAULT_HEAD_SIZE,
                          kv_rows=128, kv_cols=DEFAULT_HEAD_SIZE, dtype=DTYPE_FP32):
    """InCore: Compute Q @ K^T for attention scores."""
    return (PTOFunctionBuilder("qk_matmul")
        .in_core()
        .tile("q", q_rows, q_cols, dtype)
        .tile("k", kv_rows, kv_cols, dtype)
        .tile("k_t", kv_cols, kv_rows, dtype)
        .tile("s", q_rows, kv_rows, dtype)
        .memref("input_q", MemorySpace.GM, dtype)
        .memref("input_k", MemorySpace.GM, dtype)
        .memref("output_s", MemorySpace.GM, dtype)
        .load("q", "input_q", 0, 0)
        .load("k", "input_k", 0, 0)
        .transpose("k_t", "k")
        .matmul("s", "q", "k_t")
        .store("s", "output_s", 0, 0)
        .build())


def create_scale_scores_func(rows=DEFAULT_BS_TILE, cols=128, dtype=DTYPE_FP32):
    """InCore: Scale attention scores by 1/sqrt(d_k)."""
    return (PTOFunctionBuilder("scale_scores")
        .in_core()
        .tile("s", rows, cols, dtype)
        .tile("s_scaled", rows, cols, dtype)
        .scalar("scale", dtype)
        .memref("input_s", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("s", "input_s", 0, 0)
        .scalar_li("scale", SOFTMAX_SCALE)
        .muls("s_scaled", "s", "scale")
        .store("s_scaled", "output", 0, 0)
        .build())


def create_rowmax_func(rows=DEFAULT_BS_TILE, cols=128, dtype=DTYPE_FP32):
    """InCore: Compute row-wise maximum."""
    return (PTOFunctionBuilder("rowmax")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rowmax("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_rowexpandsub_func(rows=DEFAULT_BS_TILE, cols=128, dtype=DTYPE_FP32):
    """InCore: Row-wise expand and subtract."""
    return (PTOFunctionBuilder("rowexpandsub")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandsub("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


def create_exp_func(rows=DEFAULT_BS_TILE, cols=128, dtype=DTYPE_FP32):
    """InCore: Element-wise exponential."""
    return (PTOFunctionBuilder("elem_exp")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .exp("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_rowsum_func(rows=DEFAULT_BS_TILE, cols=128, dtype=DTYPE_FP32):
    """InCore: Compute row-wise sum."""
    return (PTOFunctionBuilder("rowsum")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rowsum("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_maximum_func(rows=DEFAULT_BS_TILE, dtype=DTYPE_FP32):
    """InCore: Element-wise maximum."""
    return (PTOFunctionBuilder("maximum")
        .in_core()
        .tile("a", rows, 1, dtype)
        .tile("b", rows, 1, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .max("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())


def create_sub_func(rows=DEFAULT_BS_TILE, dtype=DTYPE_FP32):
    """InCore: Element-wise subtraction."""
    return (PTOFunctionBuilder("sub")
        .in_core()
        .tile("a", rows, 1, dtype)
        .tile("b", rows, 1, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .sub("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())


def create_mul_func(rows=DEFAULT_BS_TILE, dtype=DTYPE_FP32):
    """InCore: Element-wise multiplication."""
    return (PTOFunctionBuilder("mul")
        .in_core()
        .tile("a", rows, 1, dtype)
        .tile("b", rows, 1, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .mul("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())


def create_add_func(rows=DEFAULT_BS_TILE, dtype=DTYPE_FP32):
    """InCore: Element-wise addition."""
    return (PTOFunctionBuilder("add")
        .in_core()
        .tile("a", rows, 1, dtype)
        .tile("b", rows, 1, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .add("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())


def create_pv_matmul_func(p_rows=DEFAULT_BS_TILE, p_cols=128,
                         v_rows=128, v_cols=DEFAULT_HEAD_SIZE, dtype=DTYPE_FP32):
    """InCore: Compute P @ V for attention output."""
    return (PTOFunctionBuilder("pv_matmul")
        .in_core()
        .tile("p", p_rows, p_cols, dtype)
        .tile("v", v_rows, v_cols, dtype)
        .tile("o", p_rows, v_cols, dtype)
        .memref("input_p", MemorySpace.GM, dtype)
        .memref("input_v", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("p", "input_p", 0, 0)
        .load("v", "input_v", 0, 0)
        .matmul("o", "p", "v")
        .store("o", "output", 0, 0)
        .build())


def create_rowexpandmul_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HEAD_SIZE, dtype=DTYPE_FP32):
    """InCore: Row-wise expand and multiply."""
    return (PTOFunctionBuilder("rowexpandmul")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandmul("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


def create_rowexpanddiv_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HEAD_SIZE, dtype=DTYPE_FP32):
    """InCore: Row-wise expand and divide."""
    return (PTOFunctionBuilder("rowexpanddiv")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpanddiv("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Orchestration Function: Attention Fusion
# =============================================================================

def create_attention_fusion_module(bs_tile=DEFAULT_BS_TILE, hidden_size=DEFAULT_HIDDEN_SIZE,
                                   head_size=DEFAULT_HEAD_SIZE, q_num_heads=DEFAULT_Q_NUM_HEADS,
                                   kv_num_heads=DEFAULT_KV_NUM_HEADS, dtype=DTYPE_FP32):
    """
    Create a module with Attention Fusion orchestration function.
    
    This implements the fused attention operation including:
    - Input LayerNorm + Residual (simplified - not fully implemented)
    - Input Quantization (simplified - not fully implemented)
    - QKV MatMul (simplified - using FP32 instead of quantized)
    - Q/K LayerNorm (simplified - not fully implemented)
    - RoPE (simplified - not fully implemented)
    - Flash Attention (core logic)
    
    Note: This is a simplified version focusing on the core Flash Attention logic.
    Full fusion with quantization and RoPE would require more complex implementation.
    
    Returns:
        PTOModule with all component functions
    """
    module = PTOModule("attention_fusion_module")
    
    # Add InCore building block functions
    module.add_function(create_rms_norm_bias_func(bs_tile, hidden_size, dtype))
    module.add_function(create_qk_matmul_func(bs_tile, head_size, 128, head_size, dtype))
    module.add_function(create_scale_scores_func(bs_tile, 128, dtype))
    module.add_function(create_rowmax_func(bs_tile, 128, dtype))
    module.add_function(create_rowexpandsub_func(bs_tile, 128, dtype))
    module.add_function(create_exp_func(bs_tile, 128, dtype))
    module.add_function(create_rowsum_func(bs_tile, 128, dtype))
    module.add_function(create_maximum_func(bs_tile, dtype))
    module.add_function(create_sub_func(bs_tile, dtype))
    module.add_function(create_mul_func(bs_tile, dtype))
    module.add_function(create_add_func(bs_tile, dtype))
    module.add_function(create_pv_matmul_func(bs_tile, 128, 128, head_size, dtype))
    module.add_function(create_rowexpandmul_func(bs_tile, head_size, dtype))
    module.add_function(create_rowexpanddiv_func(bs_tile, head_size, dtype))
    
    # Create orchestration function for attention fusion
    # Simplified version: focuses on core Flash Attention logic
    # Note: Full fusion (with quantization, RoPE) would be more complex
    # For simplicity, we use single head dimensions (head_size instead of q_num_heads * head_size)
    q_cols_single = head_size  # Single head dimension for simplified version
    attention_fusion = (PTOFunctionBuilder("attention_fusion_block", module=module)
        .not_in_core()
        
        # Memory references
        .memref("input_q", MemorySpace.GM, dtype, (bs_tile, q_cols_single))
        .memref("input_k", MemorySpace.GM, dtype, (128, head_size))
        .memref("input_v", MemorySpace.GM, dtype, (128, head_size))
        .memref("output_o", MemorySpace.GM, dtype, (bs_tile, q_cols_single))
        
        # State variables for Flash Attention
        .memref("state_o", MemorySpace.GM, dtype, (bs_tile, q_cols_single))
        .memref("state_l", MemorySpace.GM, dtype, (bs_tile, 1))
        # state_m: declared as (bs_tile, q_cols_single) to match actual allocation stride
        # Only the first column is used, but stride is q_cols_single for compatibility
        .memref("state_m", MemorySpace.GM, dtype, (bs_tile, q_cols_single))
        
        # Temporary buffers
        .memref("temp_s", MemorySpace.GM, dtype, (bs_tile, 128))
        .memref("temp_s_scaled", MemorySpace.GM, dtype, (bs_tile, 128))
        .memref("temp_m_new", MemorySpace.GM, dtype, (bs_tile, 1))
        .memref("temp_m_local", MemorySpace.GM, dtype, (bs_tile, 1))
        .memref("temp_s_shifted", MemorySpace.GM, dtype, (bs_tile, 128))
        .memref("temp_p", MemorySpace.GM, dtype, (bs_tile, 128))
        .memref("temp_l_local", MemorySpace.GM, dtype, (bs_tile, 1))
        .memref("temp_m_diff", MemorySpace.GM, dtype, (bs_tile, 1))
        .memref("temp_scale", MemorySpace.GM, dtype, (bs_tile, 1))
        .memref("temp_l_scaled", MemorySpace.GM, dtype, (bs_tile, 1))
        .memref("temp_o_scaled", MemorySpace.GM, dtype, (bs_tile, q_cols_single))
        .memref("temp_o_local", MemorySpace.GM, dtype, (bs_tile, q_cols_single))
        
        # Temporary tiles for computation
        .tile("tile_o_scaled", bs_tile, q_cols_single, dtype)
        .tile("tile_o_local", bs_tile, q_cols_single, dtype)
        .tile("tile_o_sum", bs_tile, q_cols_single, dtype)
        .tile("tile_m_copy", bs_tile, 1, dtype)
        
        # Flash Attention computation (similar to flash_attention_block)
        # Step 1: Compute attention scores S = Q @ K^T
        .call("qk_matmul", {
            "input_q": "input_q",
            "input_k": "input_k",
            "output_s": "temp_s"
        })
        
        # Step 2: Scale scores
        .call("scale_scores", {
            "input_s": "temp_s",
            "output": "temp_s_scaled"
        })
        
        # Step 3: Find row-wise max
        .call("rowmax", {
            "input": "temp_s_scaled",
            "output": "temp_m_local"
        })
        
        # Step 4: Compute m_new = max(m, m_local)
        .call("maximum", {
            "input_a": "state_m",
            "input_b": "temp_m_local",
            "output": "temp_m_new"
        })
        
        # Step 5: Compute S - m_new
        .call("rowexpandsub", {
            "input_x": "temp_s_scaled",
            "input_row": "temp_m_new",
            "output": "temp_s_shifted"
        })
        
        # Step 6: Compute P = exp(S - m_new)
        .call("elem_exp", {
            "input": "temp_s_shifted",
            "output": "temp_p"
        })
        
        # Step 7: Compute local sum
        .call("rowsum", {
            "input": "temp_p",
            "output": "temp_l_local"
        })
        
        # Step 8: Compute m - m_new
        .call("sub", {
            "input_a": "state_m",
            "input_b": "temp_m_new",
            "output": "temp_m_diff"
        })
        
        # Step 9: Compute scale = exp(m - m_new)
        .call("elem_exp", {
            "input": "temp_m_diff",
            "output": "temp_scale"
        })
        
        # Step 10: Compute scale * l
        .call("mul", {
            "input_a": "temp_scale",
            "input_b": "state_l",
            "output": "temp_l_scaled"
        })
        
        # Step 11: Compute l_new = scale * l + l_local
        .call("add", {
            "input_a": "temp_l_scaled",
            "input_b": "temp_l_local",
            "output": "state_l"
        })
        
        # Step 12: Compute P @ V
        .call("pv_matmul", {
            "input_p": "temp_p",
            "input_v": "input_v",
            "output": "temp_o_local"
        })
        
        # Step 13: Compute scale * O
        .call("rowexpandmul", {
            "input_x": "state_o",
            "input_row": "temp_scale",
            "output": "temp_o_scaled"
        })
        
        # Step 14: Compute O_new = scale * O + O_local
        # Load both tensors into tiles, add them, then store back to state_o
        .load("tile_o_scaled", "temp_o_scaled", 0, 0)
        .load("tile_o_local", "temp_o_local", 0, 0)
        .add("tile_o_sum", "tile_o_scaled", "tile_o_local")
        .store("tile_o_sum", "state_o", 0, 0)
        
        # Step 15: Update m = m_new
        .load("tile_m_copy", "temp_m_new", 0, 0)
        .store("tile_m_copy", "state_m", 0, 0)
        
        # Step 16: Final normalization O = state_o / state_l
        .call("rowexpanddiv", {
            "input_x": "state_o",
            "input_row": "state_l",
            "output": "output_o"
        })
        
        .build())
    
    module.add_function(attention_fusion)
    return module


def create_attention_fusion_module_default():
    """Create the complete attention fusion module with default parameters."""
    return create_attention_fusion_module(
        bs_tile=DEFAULT_BS_TILE,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        head_size=DEFAULT_HEAD_SIZE,
        q_num_heads=DEFAULT_Q_NUM_HEADS,
        kv_num_heads=DEFAULT_KV_NUM_HEADS,
        dtype=DTYPE_FP32
    )


# =============================================================================
# Main: Module Creation
# =============================================================================

def main():
    """Create and describe the Attention Fusion module."""
    print("=" * 70)
    print("PTO Attention Fusion Function")
    print("=" * 70)
    
    # Create module
    module = create_attention_fusion_module_default()
    
    print(f"\nModule: {module.name}")
    print(f"\nFunctions ({len(module.get_function_names())}):")
    
    incore_funcs = []
    orch_funcs = []
    for name in module.get_function_names():
        func = module.get_function(name)
        if func.is_in_core:
            incore_funcs.append(name)
        else:
            orch_funcs.append(name)
    
    print(f"\n  InCore ({len(incore_funcs)}):")
    for name in sorted(incore_funcs):
        print(f"    - {name}")
    
    print(f"\n  Orchestration ({len(orch_funcs)}):")
    for name in sorted(orch_funcs):
        print(f"    - {name}")
    
    print("\n" + "=" * 70)
    print("To generate code, compile, and run:")
    print("  python run_arm64.py")
    print("=" * 70)
    
    return module


if __name__ == "__main__":
    main()
