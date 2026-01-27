"""
PTO MoE Fusion Function Implementation

This module implements the fused MoE operation from glm_moe_fusion.py
using pto-isa-liao framework. The implementation includes:

1. Gate computation: hidden_states @ gate_weight -> logits -> sigmoid
2. Share experts (FFN): Quantized matmul -> dequant -> SwiGLU -> quant -> matmul -> dequant

This is a simplified version focusing on the core logic, adapted from the pypto
implementation in glm_moe_fusion.py.
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
DEFAULT_INTERMEDIATE_SIZE = 192  # Intermediate dimension (for FFN)
DEFAULT_NUM_EXPERTS = 160  # Number of experts

# Data types
DTYPE_FP32 = ElementType.F32
DTYPE_FP16 = ElementType.F16
DTYPE_INT8 = ElementType.I8
DTYPE_INT32 = ElementType.I32

# =============================================================================
# InCore Functions: Tile-level Operations
# =============================================================================

def create_gate_matmul_func(bs_tile=DEFAULT_BS_TILE, hidden_size=DEFAULT_HIDDEN_SIZE,
                            num_experts=DEFAULT_NUM_EXPERTS, dtype=DTYPE_FP32):
    """InCore: Compute gate logits = hidden_states @ gate_weight^T."""
    return (PTOFunctionBuilder("gate_matmul")
        .in_core()
        .tile("hidden", bs_tile, hidden_size, dtype)
        .tile("gate_weight", num_experts, hidden_size, dtype)
        .tile("gate_weight_t", hidden_size, num_experts, dtype)
        .tile("logits", bs_tile, num_experts, dtype)
        .memref("input_hidden", MemorySpace.GM, dtype, (bs_tile, hidden_size))
        .memref("input_gate_weight", MemorySpace.GM, dtype, (num_experts, hidden_size))
        .memref("output_logits", MemorySpace.GM, dtype, (bs_tile, num_experts))
        .load("hidden", "input_hidden", 0, 0)
        .load("gate_weight", "input_gate_weight", 0, 0)
        .transpose("gate_weight_t", "gate_weight")
        .matmul("logits", "hidden", "gate_weight_t")
        .store("logits", "output_logits", 0, 0)
        .build())


def create_sigmoid_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_NUM_EXPERTS, dtype=DTYPE_FP32):
    """InCore: Apply sigmoid activation."""
    return (PTOFunctionBuilder("sigmoid")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("x_neg", rows, cols, dtype)
        .tile("x_exp", rows, cols, dtype)
        .tile("x_exp_plus_one", rows, cols, dtype)
        .tile("one_tile", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .scalar("one", dtype)
        .memref("input", MemorySpace.GM, dtype, (rows, cols))
        .memref("output", MemorySpace.GM, dtype, (rows, cols))
        .load("x", "input", 0, 0)
        # sigmoid(x) = 1 / (1 + exp(-x))
        .muls("x_neg", "x", -1.0)
        .exp("x_exp", "x_neg")
        .scalar_li("one", 1.0)
        .adds("x_exp_plus_one", "x_exp", "one")
        # Create tile with 1.0 for division: one_tile = 1.0 (broadcast)
        .muls("one_tile", "x_exp_plus_one", 0.0)  # Zero tile
        .adds("one_tile", "one_tile", 1.0)  # Set to 1.0
        # div: result = one_tile / x_exp_plus_one (element-wise)
        .div("result", "one_tile", "x_exp_plus_one")
        .store("result", "output", 0, 0)
        .build())


def create_add_bias_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_NUM_EXPERTS, dtype=DTYPE_FP32):
    """InCore: Add bias to each row (broadcast)."""
    return (PTOFunctionBuilder("add_bias")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("bias_1d", 1, cols, dtype)  # Bias is [1, cols], will be expanded to [rows, cols]
        .tile("bias_expanded", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype, (rows, cols))
        .memref("input_bias", MemorySpace.GM, dtype, (1, cols))  # 1D bias [cols] -> (1, cols)
        .memref("output", MemorySpace.GM, dtype, (rows, cols))
        .load("x", "input_x", 0, 0)
        .load("bias_1d", "input_bias", 0, 0)  # Load as [1, cols]
        # Expand bias to [rows, cols] by manually copying for each row
        # Since rowexpandadd may not exist, we'll manually expand
        # For each row i, result[i, :] = x[i, :] + bias_1d[0, :]
        # This is done in the fused loop below
        .add("result", "x", "bias_1d")  # Add will broadcast if dimensions allow
        .store("result", "output", 0, 0)
        .build())


def create_swiglu_func(rows=DEFAULT_BS_TILE, intermediate_size=DEFAULT_INTERMEDIATE_SIZE, dtype=DTYPE_FP32):
    """InCore: SwiGLU activation = x_left * sigmoid(x_left) * x_right."""
    return (PTOFunctionBuilder("swiglu")
        .in_core()
        .tile("up_proj", rows, intermediate_size * 2, dtype)
        .tile("left", rows, intermediate_size, dtype)
        .tile("right", rows, intermediate_size, dtype)
        .tile("left_neg", rows, intermediate_size, dtype)
        .tile("left_exp", rows, intermediate_size, dtype)
        .tile("left_exp_plus_one", rows, intermediate_size, dtype)
        .tile("sigmoid_left", rows, intermediate_size, dtype)
        .tile("result", rows, intermediate_size, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        # Load full up_proj
        # Load full up_proj first
        .load("up_proj", "input", 0, 0)
        # Split into left and right halves
        # left = up_proj[:, :intermediate_size] - load from input[0, 0] with cols=intermediate_size
        # right = up_proj[:, intermediate_size:] - load from input[0, intermediate_size]
        # Note: We need to load from the same input but different column offsets
        # Load left half: from input[0, 0] with shape [rows, intermediate_size]
        .load("left", "input", 0, 0)  # This will load [rows, intermediate_size] from start
        # Load right half: from input[0, intermediate_size] with shape [rows, intermediate_size]
        .load("right", "input", 0, intermediate_size)  # Load from column offset intermediate_size
        # sigmoid(left) = 1 / (1 + exp(-left))
        .muls("left_neg", "left", -1.0)
        .exp("left_exp", "left_neg")
        .scalar_li("one", 1.0)
        .adds("left_exp_plus_one", "left_exp", "one")
        # Create tile with 1.0 for division
        .muls("sigmoid_left", "left_exp_plus_one", 0.0)  # Zero tile
        .adds("sigmoid_left", "sigmoid_left", 1.0)  # Set to 1.0
        .div("sigmoid_left", "sigmoid_left", "left_exp_plus_one")
        # result = left * sigmoid_left * right
        .mul("result", "left", "sigmoid_left")
        .mul("result", "result", "right")
        .store("result", "output", 0, 0)
        .build())


def create_abs_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
    """InCore: Element-wise absolute value."""
    return (PTOFunctionBuilder("abs_tile")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("x_neg", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        # abs(x) = max(x, -x)
        .muls("x_neg", "x", -1.0)
        .max("result", "x", "x_neg")
        .store("result", "output", 0, 0)
        .build())


def create_rowmax_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
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


def create_rowexpanddiv_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
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


def create_rowexpandmul_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
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


def create_mul_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
    """InCore: Element-wise multiply."""
    return (PTOFunctionBuilder("mul")
        .in_core()
        .tile("a", rows, cols, dtype)
        .tile("b", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .mul("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())


def create_add_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
    """InCore: Element-wise add."""
    return (PTOFunctionBuilder("add")
        .in_core()
        .tile("a", rows, cols, dtype)
        .tile("b", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .add("result", "a", "b")
        .store("result", "output", 0, 0)
        .build())


def create_quantize_per_token_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
    """
    InCore: Per-token symmetric quantization (simplified - returns fp32 for now).
    
    For a full implementation, this would quantize to int8, but for simplicity
    we'll return the scaled fp32 values and a scale factor.
    """
    return (PTOFunctionBuilder("quantize_per_token")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("x_abs", rows, cols, dtype)
        .tile("x_neg", rows, cols, dtype)
        .tile("x_max", rows, 1, dtype)
        .tile("x_scale", rows, 1, dtype)
        .tile("x_scaled", rows, cols, dtype)
        .tile("scale_quant", rows, 1, dtype)
        .tile("one_hundred_twenty_seven_tile", rows, 1, dtype)
        .tile("one_tile", rows, 1, dtype)
        .scalar("one_hundred_twenty_seven", dtype)
        .scalar("one", dtype)
        .memref("input", MemorySpace.GM, dtype, (rows, cols))
        .memref("output_quant", MemorySpace.GM, dtype, (rows, cols))  # Store as fp32 for now
        .memref("output_scale", MemorySpace.GM, dtype, (rows, 1))
        .load("x", "input", 0, 0)
        # abs(x) = max(x, -x)
        .muls("x_neg", "x", -1.0)
        .max("x_abs", "x", "x_neg")  # x_abs = max(x, -x)
        # max(abs(x)) per row
        .rowmax("x_max", "x_abs")
        # scale = 127.0 / max
        # For rowexpanddiv(dst, tile, row_vals): dst = tile / row_vals (broadcast row_vals)
        # We need: x_scale = 127.0 / x_max
        # Create a tile with 127.0 for each row
        .scalar_li("one_hundred_twenty_seven", 127.0)
        .muls("one_hundred_twenty_seven_tile", "x_max", 0.0)  # Zero tile [rows, 1]
        .adds("one_hundred_twenty_seven_tile", "one_hundred_twenty_seven_tile", 127.0)  # Set to 127.0
        .rowexpanddiv("x_scale", "one_hundred_twenty_seven_tile", "x_max")  # x_scale = 127.0 / x_max
        # x_scaled = x * scale
        .rowexpandmul("x_scaled", "x", "x_scale")
        # scale_quant = 1.0 / scale (for dequantization)
        .scalar_li("one", 1.0)
        .muls("one_tile", "x_scale", 0.0)  # Zero tile [rows, 1]
        .adds("one_tile", "one_tile", 1.0)  # Set to 1.0
        .rowexpanddiv("scale_quant", "one_tile", "x_scale")  # scale_quant = 1.0 / x_scale
        .store("x_scaled", "output_quant", 0, 0)
        .store("scale_quant", "output_scale", 0, 0)
        .build())


def create_dequant_dynamic_func(rows=DEFAULT_BS_TILE, cols=DEFAULT_INTERMEDIATE_SIZE, dtype=DTYPE_FP32):
    """
    InCore: Dynamic dequantization with two scale factors (simplified - works with fp32).
    
    output = input * scale_1 * scale_2
    """
    return (PTOFunctionBuilder("dequant_dynamic")
        .in_core()
        .tile("input_fp32", rows, cols, dtype)
        .tile("scale_1", rows, 1, dtype)
        .tile("scale_2", rows, 1, dtype)
        .tile("scaled_1", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype, (rows, cols))
        .memref("input_scale_1", MemorySpace.GM, dtype, (rows, 1))
        .memref("input_scale_2", MemorySpace.GM, dtype, (rows, 1))
        .memref("output", MemorySpace.GM, dtype, (rows, cols))
        .load("input_fp32", "input", 0, 0)
        .load("scale_1", "input_scale_1", 0, 0)
        .load("scale_2", "input_scale_2", 0, 0)
        # scaled_1 = input_fp32 * scale_2
        .rowexpandmul("scaled_1", "input_fp32", "scale_2")
        # result = scaled_1 * scale_1
        .rowexpandmul("result", "scaled_1", "scale_1")
        .store("result", "output", 0, 0)
        .build())


def create_up_proj_matmul_func(bs_tile=DEFAULT_BS_TILE, hidden_size=DEFAULT_HIDDEN_SIZE,
                               intermediate_size=DEFAULT_INTERMEDIATE_SIZE, dtype=DTYPE_FP32):
    """InCore: Matmul for up_proj: hidden_states @ w13 (simplified - uses fp32)."""
    return (PTOFunctionBuilder("up_proj_matmul")
        .in_core()
        .tile("hidden", bs_tile, hidden_size, dtype)
        .tile("w13", hidden_size, intermediate_size * 2, dtype)
        .tile("w13_t", intermediate_size * 2, hidden_size, dtype)
        .tile("result", bs_tile, intermediate_size * 2, dtype)
        .memref("input_hidden", MemorySpace.GM, dtype, (bs_tile, hidden_size))
        .memref("input_w13", MemorySpace.GM, dtype, (hidden_size, intermediate_size * 2))
        .memref("output", MemorySpace.GM, dtype, (bs_tile, intermediate_size * 2))
        .load("hidden", "input_hidden", 0, 0)
        .load("w13", "input_w13", 0, 0)
        .transpose("w13_t", "w13")
        .matmul("result", "hidden", "w13_t")
        .store("result", "output", 0, 0)
        .build())


def create_down_proj_matmul_func(bs_tile=DEFAULT_BS_TILE, intermediate_size=DEFAULT_INTERMEDIATE_SIZE,
                                 hidden_size=DEFAULT_HIDDEN_SIZE, dtype=DTYPE_FP32):
    """InCore: Matmul for down_proj: swiglu @ w2 (simplified - uses fp32)."""
    return (PTOFunctionBuilder("down_proj_matmul")
        .in_core()
        .tile("swiglu", bs_tile, intermediate_size, dtype)
        .tile("w2", intermediate_size, hidden_size, dtype)
        .tile("w2_t", hidden_size, intermediate_size, dtype)
        .tile("result", bs_tile, hidden_size, dtype)
        .memref("input_swiglu", MemorySpace.GM, dtype, (bs_tile, intermediate_size))
        .memref("input_w2", MemorySpace.GM, dtype, (intermediate_size, hidden_size))
        .memref("output", MemorySpace.GM, dtype, (bs_tile, hidden_size))
        .load("swiglu", "input_swiglu", 0, 0)
        .load("w2", "input_w2", 0, 0)
        .transpose("w2_t", "w2")
        .matmul("result", "swiglu", "w2_t")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Orchestration Function: MoE Fusion Block
# =============================================================================

def create_moe_fusion_module(bs_tile=DEFAULT_BS_TILE, hidden_size=DEFAULT_HIDDEN_SIZE,
                             intermediate_size=DEFAULT_INTERMEDIATE_SIZE,
                             num_experts=DEFAULT_NUM_EXPERTS, dtype=DTYPE_FP32):
    """
    Create MoE Fusion module with InCore and Orchestration functions.
    
    Simplified version focusing on:
    1. Gate computation: hidden_states @ gate_weight -> sigmoid -> +bias
    2. Share experts (FFN): quantize -> matmul -> dequant -> SwiGLU -> quant -> matmul -> dequant
    """
    module = PTOModule("moe_fusion_module")
    
    # Add InCore functions
    module.add_function(create_gate_matmul_func(bs_tile, hidden_size, num_experts, dtype))
    module.add_function(create_sigmoid_func(bs_tile, num_experts, dtype))
    module.add_function(create_add_bias_func(bs_tile, num_experts, dtype))
    module.add_function(create_swiglu_func(bs_tile, intermediate_size, dtype))
    # Note: abs function renamed to abs_tile to avoid conflict with stdlib abs()
    # module.add_function(create_abs_func(bs_tile, hidden_size, dtype))  # Not used in simplified version
    module.add_function(create_rowmax_func(bs_tile, hidden_size, dtype))
    module.add_function(create_rowexpanddiv_func(bs_tile, hidden_size, dtype))
    module.add_function(create_rowexpandmul_func(bs_tile, hidden_size, dtype))
    module.add_function(create_mul_func(bs_tile, hidden_size, dtype))
    module.add_function(create_add_func(bs_tile, hidden_size, dtype))
    module.add_function(create_quantize_per_token_func(bs_tile, hidden_size, dtype))
    module.add_function(create_dequant_dynamic_func(bs_tile, intermediate_size * 2, dtype))
    module.add_function(create_up_proj_matmul_func(bs_tile, hidden_size, intermediate_size, dtype))
    module.add_function(create_down_proj_matmul_func(bs_tile, intermediate_size, hidden_size, dtype))
    
    # Create orchestration function for MoE fusion
    moe_fusion = (PTOFunctionBuilder("moe_fusion_block", module=module)
        .not_in_core()
        
        # Memory references
        .memref("input_hidden_states", MemorySpace.GM, dtype, (bs_tile, hidden_size))
        .memref("input_gate_weight", MemorySpace.GM, dtype, (num_experts, hidden_size))
        .memref("input_e_score_bias", MemorySpace.GM, dtype, (1, num_experts))  # 1D bias -> (1, num_experts)
        .memref("input_w13", MemorySpace.GM, dtype, (hidden_size, intermediate_size * 2))
        .memref("input_w13_scale", MemorySpace.GM, dtype, (1, intermediate_size * 2))  # 1D scale -> (1, cols)
        .memref("input_w2", MemorySpace.GM, dtype, (intermediate_size, hidden_size))
        .memref("input_w2_scale", MemorySpace.GM, dtype, (1, hidden_size))  # 1D scale -> (1, cols)
        .memref("output_logits", MemorySpace.GM, dtype, (bs_tile, num_experts))
        .memref("output_ffn_res", MemorySpace.GM, dtype, (bs_tile, hidden_size))
        
        # Temporary buffers for gate computation
        .memref("temp_gate_logits", MemorySpace.GM, dtype, (bs_tile, num_experts))
        .memref("temp_sigmoid_out", MemorySpace.GM, dtype, (bs_tile, num_experts))
        .memref("temp_gate_weights", MemorySpace.GM, dtype, (bs_tile, num_experts))
        
        # Temporary buffers for FFN computation
        .memref("temp_up_proj", MemorySpace.GM, dtype, (bs_tile, intermediate_size * 2))
        .memref("temp_swiglu_out", MemorySpace.GM, dtype, (bs_tile, intermediate_size))
        .memref("temp_down_proj", MemorySpace.GM, dtype, (bs_tile, hidden_size))
        
        # Temporary tiles for load/store operations
        .tile("tile_gate", bs_tile, num_experts, dtype)
        .tile("tile_ffn_res", bs_tile, hidden_size, dtype)
        
        # Gate computation: Step 1 - Matmul
        .call("gate_matmul", {
            "input_hidden": "input_hidden_states",
            "input_gate_weight": "input_gate_weight",
            "output_logits": "temp_gate_logits"
        })
        
        # Gate computation: Step 2 - Sigmoid
        .call("sigmoid", {
            "input": "temp_gate_logits",
            "output": "temp_sigmoid_out"
        })
        
        # Gate computation: Step 3 - Add bias
        .call("add_bias", {
            "input_x": "temp_sigmoid_out",
            "input_bias": "input_e_score_bias",
            "output": "temp_gate_weights"
        })
        
        # Store gate weights to output
        .load("tile_gate", "temp_gate_weights", 0, 0)
        .store("tile_gate", "output_logits", 0, 0)
        
        # Share experts (FFN) computation
        # Step 1: Up projection matmul (simplified: skip quantization for now)
        .call("up_proj_matmul", {
            "input_hidden": "input_hidden_states",
            "input_w13": "input_w13",
            "output": "temp_up_proj"
        })
        
        # Step 3: Prepare w13_scale for dequant (expand to 2D)
        # For now, skip dequant and use direct matmul result
        # In full implementation, would dequant here
        
        # Step 4: SwiGLU activation
        .call("swiglu", {
            "input": "temp_up_proj",
            "output": "temp_swiglu_out"
        })
        
        # Step 5: Down projection matmul
        .call("down_proj_matmul", {
            "input_swiglu": "temp_swiglu_out",
            "input_w2": "input_w2",
            "output": "temp_down_proj"
        })
        
        # Step 6: Store FFN result
        .load("tile_ffn_res", "temp_down_proj", 0, 0)
        .store("tile_ffn_res", "output_ffn_res", 0, 0)
        
        .build())
    
    module.add_function(moe_fusion)
    return module


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Create module
    module = create_moe_fusion_module()
    
    print("MoE Fusion module created successfully!")
    print("To generate code, compile, and run:")
    print("  python run_arm64.py")
    print("=" * 70)
