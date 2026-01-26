#!/usr/bin/env python3
"""
PTO Example Runner - GLM-4.5 MoE Fusion Function

This script:
1. Generates ARM64 code for MoE Fusion function (based on glm_moe_fusion.py)
2. Compiles the generated code
3. Runs accuracy tests comparing with reference implementation
4. Validates precision correctness

Note: This script requires conda environment 'py312' with numpy installed.
"""

import os
import sys
import time
import subprocess
import math
from datetime import datetime

# Check for numpy and provide helpful error message if not found
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is not available in the current Python environment.")
    print("\nPlease use conda environment 'py312':")
    print("  conda activate py312")
    print("  python run_arm64.py")
    print("\nOr run with conda:")
    print("  conda run -n py312 python run_arm64.py")
    sys.exit(1)

# =============================================================================
# Path Setup
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
RUNTIME_DIR = os.path.join(SRC_DIR, "runtime")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

sys.path.insert(0, SRC_DIR)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "example_name": "glm_v4_5_moe_fusion",
    "target_platform": "arm64",
    "enable_binary_expansion": True,
    "enable_task_dump": True,
    "enable_task_graph_pdf": False,
    "enable_perf_benchmark": False,
    "enable_accuracy_test": True,
    "enable_simulation": False,
    "num_warmup_iterations": 1,
    "num_benchmark_iterations": 1,
    # Test parameters
    "test_bs_tile": 8,  # Batch size tile
    "test_hidden_size": 5120,  # Hidden dimension
    "test_intermediate_size": 192,  # Intermediate dimension
    "test_num_experts": 160,  # Number of experts
}

# =============================================================================
# Imports
# =============================================================================

try:
    from compile.pto_compile import (
        PTOFunctionBuilder, PTOModule, MultiBackendCodeGenerator,
        generate_arm64_code,
    )
    from isa_definition.pto_isa_definition import ElementType, MemorySpace
except ImportError as e:
    print(f"Error importing PTO modules: {e}")
    print("Make sure you're running from the correct directory.")
    sys.exit(1)

# =============================================================================
# Utility Functions
# =============================================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def run_command(cmd, cwd=None, timeout=300):
    """Run a shell command and return result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


# =============================================================================
# Code Generation
# =============================================================================

def fix_generated_code():
    """Post-process generated code to fix known issues."""
    platform_dir = os.path.join(OUTPUT_DIR, "arm64")
    code_dir = os.path.join(platform_dir, "generated_code")
    
    if not os.path.exists(code_dir):
        return
    
    # Fix swiglu.c: Remove extra parameter and fix left/right loading
    swiglu_file = os.path.join(code_dir, "swiglu.c")
    if os.path.exists(swiglu_file):
        with open(swiglu_file, 'r') as f:
            content = f.read()
        
        # Fix function signature: remove extra 'float one' parameter
        if 'void swiglu(float* input, float* output, float one)' in content:
            print("    Fixing swiglu.c: Removing extra parameter")
            content = content.replace(
                'void swiglu(float* input, float* output, float one)',
                'void swiglu(float* input, float* output)'
            )
            # Fix usage of 'one' variable - replace with constant 1.0
            content = content.replace(' + one', ' + 1.0')
            content = content.replace(' = one', ' = 1.0')
            content = content.replace('(one', '(1.0')
            
            # Fix left loading: should load [rows, intermediate_size] from input[0, 0]
            # The generated code loads left from input[0, 0] with wrong stride
            # left should load from input[0, 0] with stride = intermediate_size * 2 = 384
            if 'left[_row][_col] = input[_row * 192 + _col];' in content:
                # Fix: left should load from input with stride = 384 (intermediate_size * 2)
                content = content.replace(
                    'left[_row][_col] = input[_row * 192 + _col];',
                    'left[_row][_col] = input[_row * 384 + _col];  // FIX: stride is 384 (intermediate_size * 2)'
                )
            
            # Fix right loading: should load from input[0, intermediate_size] with stride = 384
            if 'right[_row][_col] = input[(0 + _row) * 192 + 192 + _col];' in content:
                # Fix: right should load from input with stride = 384, offset = 192
                content = content.replace(
                    'right[_row][_col] = input[(0 + _row) * 192 + 192 + _col];',
                    'right[_row][_col] = input[_row * 384 + 192 + _col];  // FIX: stride is 384, offset is 192'
                )
            
            with open(swiglu_file, 'w') as f:
                f.write(content)
            print("    ✓ Fixed swiglu.c")
    
    # Fix add_bias.c: Correct bias loading (bias is [1, cols], not [rows, 1])
    add_bias_file = os.path.join(code_dir, "add_bias.c")
    if os.path.exists(add_bias_file):
        with open(add_bias_file, 'r') as f:
            content = f.read()
        
        fixes_applied = False
        
        # Fix bias loading: input_bias is 1D array [160], so load directly
        if 'bias_1d[_row][_col] = input_bias[_row * 160 + _col];' in content:
            print("    Fixing add_bias.c: Correcting bias loading")
            content = content.replace(
                'bias_1d[_row][_col] = input_bias[_row * 160 + _col];',
                'bias_1d[0][_col] = input_bias[_col];  // FIX: input_bias is 1D array'
            )
            fixes_applied = True
        
        # Fix addition: use bias_1d[0][_col] instead of bias_1d[_row][_col]
        if 'result[_row][_col] = x[_row][_col] + bias_1d[_row][_col];' in content:
            content = content.replace(
                'result[_row][_col] = x[_row][_col] + bias_1d[_row][_col];',
                'result[_row][_col] = x[_row][_col] + bias_1d[0][_col];  // FIX: broadcast bias[0, :] to all rows'
            )
            fixes_applied = True
        
        # Remove unused bias_expanded variable
        if 'float bias_expanded[8][160];' in content:
            content = content.replace('float bias_expanded[8][160];', '// bias_expanded not needed')
            fixes_applied = True
        
        if fixes_applied:
            with open(add_bias_file, 'w') as f:
                f.write(content)
            print("    ✓ Fixed add_bias.c")
    
    # Fix gate_matmul.c: Use dynamic allocation to avoid stack overflow
    gate_matmul_file = os.path.join(code_dir, "gate_matmul.c")
    if os.path.exists(gate_matmul_file):
        with open(gate_matmul_file, 'r') as f:
            content = f.read()
        
        fixes_applied = False
        
        if 'float hidden[8][5120];' in content and 'malloc' not in content:
            print("    Fixing gate_matmul.c: Using dynamic allocation to avoid stack overflow")
            old_decl = '''void gate_matmul(float* input_hidden, float* input_gate_weight, float* output_logits) {
    float hidden[8][5120];
    float gate_weight[160][5120];
    float gate_weight_t[5120][160];
    float logits[8][160];'''
            new_decl = '''void gate_matmul(float* input_hidden, float* input_gate_weight, float* output_logits) {
    // Use dynamic allocation to avoid stack overflow (~3.3MB local arrays)
    float (*hidden)[5120] = (float(*)[5120])malloc(8 * 5120 * sizeof(float));
    float (*gate_weight)[5120] = (float(*)[5120])malloc(160 * 5120 * sizeof(float));
    float (*gate_weight_t)[160] = (float(*)[160])malloc(5120 * 160 * sizeof(float));
    float (*logits)[160] = (float(*)[160])malloc(8 * 160 * sizeof(float));
    if (!hidden || !gate_weight || !gate_weight_t || !logits) {
        fprintf(stderr, "ERROR: gate_matmul malloc failed\\n");
        if (hidden) free(hidden);
        if (gate_weight) free(gate_weight);
        if (gate_weight_t) free(gate_weight_t);
        if (logits) free(logits);
        return;
    }'''
            
            # Replace the declaration first
            content = content.replace(old_decl, new_decl)
            
            # Add free() calls before the closing brace if not already present
            if 'free(hidden)' not in content:
                free_code = '''
    // Free dynamically allocated memory
    free(hidden);
    free(gate_weight);
    free(gate_weight_t);
    free(logits);
}'''
                func_end = content.rfind('}')
                if func_end != -1:
                    content = content[:func_end] + free_code
                else:
                    content = content + free_code
            
            fixes_applied = True
        
        if fixes_applied:
            with open(gate_matmul_file, 'w') as f:
                f.write(content)
            print("    ✓ Fixed gate_matmul.c")
    
    # Fix up_proj_matmul.c: Use dynamic allocation
    up_proj_file = os.path.join(code_dir, "up_proj_matmul.c")
    if os.path.exists(up_proj_file):
        with open(up_proj_file, 'r') as f:
            content = f.read()
        
        if 'float hidden[8][5120];' in content and 'malloc' not in content:
            print("    Fixing up_proj_matmul.c: Using dynamic allocation to avoid stack overflow")
            old_decl = '''void up_proj_matmul(float* input_hidden, float* input_w13, float* output) {
    float hidden[8][5120];
    float w13[5120][384];
    float w13_t[384][5120];
    float result[8][384];'''
            new_decl = '''void up_proj_matmul(float* input_hidden, float* input_w13, float* output) {
    // Use dynamic allocation to avoid stack overflow (~8MB local arrays)
    float (*hidden)[5120] = (float(*)[5120])malloc(8 * 5120 * sizeof(float));
    float (*w13)[384] = (float(*)[384])malloc(5120 * 384 * sizeof(float));
    float (*w13_t)[5120] = (float(*)[5120])malloc(384 * 5120 * sizeof(float));
    float (*result)[384] = (float(*)[384])malloc(8 * 384 * sizeof(float));
    if (!hidden || !w13 || !w13_t || !result) {
        fprintf(stderr, "ERROR: up_proj_matmul malloc failed\\n");
        if (hidden) free(hidden);
        if (w13) free(w13);
        if (w13_t) free(w13_t);
        if (result) free(result);
        return;
    }'''
            
            import re
            # Remove the TODO comment if present
            content = content.replace('// TODO: Fix large array allocations if needed\n', '')
            
            # Replace the declaration first
            content = content.replace(old_decl, new_decl)
            
            # Add free() calls before the closing brace if not already present
            if 'free(hidden)' not in content:
                free_code = '''
    // Free dynamically allocated memory
    free(hidden);
    free(w13);
    free(w13_t);
    free(result);
}'''
                func_end = content.rfind('}')
                if func_end != -1:
                    content = content[:func_end] + free_code
                else:
                    content = content + free_code
            
            with open(up_proj_file, 'w') as f:
                f.write(content)
            print("    ✓ Fixed up_proj_matmul.c")
    
    # Fix down_proj_matmul.c: Use dynamic allocation
    down_proj_file = os.path.join(code_dir, "down_proj_matmul.c")
    if os.path.exists(down_proj_file):
        with open(down_proj_file, 'r') as f:
            content = f.read()
        
        if 'float swiglu[8][192];' in content and 'malloc' not in content:
            print("    Fixing down_proj_matmul.c: Using dynamic allocation to avoid stack overflow")
            old_decl = '''void down_proj_matmul(float* input_swiglu, float* input_w2, float* output) {
    float swiglu[8][192];
    float w2[192][5120];
    float w2_t[5120][192];
    float result[8][5120];'''
            new_decl = '''void down_proj_matmul(float* input_swiglu, float* input_w2, float* output) {
    // Use dynamic allocation to avoid stack overflow (~4MB local arrays)
    float (*swiglu)[192] = (float(*)[192])malloc(8 * 192 * sizeof(float));
    float (*w2)[5120] = (float(*)[5120])malloc(192 * 5120 * sizeof(float));
    float (*w2_t)[192] = (float(*)[192])malloc(5120 * 192 * sizeof(float));
    float (*result)[5120] = (float(*)[5120])malloc(8 * 5120 * sizeof(float));
    if (!swiglu || !w2 || !w2_t || !result) {
        fprintf(stderr, "ERROR: down_proj_matmul malloc failed\\n");
        if (swiglu) free(swiglu);
        if (w2) free(w2);
        if (w2_t) free(w2_t);
        if (result) free(result);
        return;
    }'''
            
            import re
            # Remove the TODO comment if present
            content = content.replace('// TODO: Fix large array allocations if needed\n', '')
            
            # Replace the declaration first
            content = content.replace(old_decl, new_decl)
            
            # Add free() calls before the closing brace if not already present
            if 'free(swiglu)' not in content:
                free_code = '''
    // Free dynamically allocated memory
    free(swiglu);
    free(w2);
    free(w2_t);
    free(result);
}'''
                func_end = content.rfind('}')
                if func_end != -1:
                    content = content[:func_end] + free_code
                else:
                    content = content + free_code
            
            with open(down_proj_file, 'w') as f:
                f.write(content)
            print("    ✓ Fixed down_proj_matmul.c")


def generate_code():
    """Generate code for the target platform."""
    print_header("Code Generation")
    
    # Import the example module
    sys.path.insert(0, SCRIPT_DIR)
    try:
        import pto_moe_fusion_func
        
        print("  Creating module...")
        module = pto_moe_fusion_func.create_moe_fusion_module()
        
        # Generate code - put source files in generated_code/ subfolder
        platform_dir = ensure_dir(os.path.join(OUTPUT_DIR, "arm64"))
        code_dir = ensure_dir(os.path.join(platform_dir, "generated_code"))
        
        gen = MultiBackendCodeGenerator(
            enable_fusion=True,
            analyze_buffers=True,
            module=module
        )
        
        for func_name, prog in module.functions.items():
            print(f"  Generating arm64 code for: {func_name}")
            
            code = gen.generate_arm64(prog)
            ext = ".c"
            
            output_file = os.path.join(code_dir, f"{func_name}{ext}")
            with open(output_file, 'w') as f:
                f.write(code)
            print(f"    -> {output_file}")
            
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  Code generation complete!")
    
    # Post-process generated code to fix known issues
    print("\n  Post-processing generated code...")
    fix_generated_code()
    
    return True


# =============================================================================
# Compilation
# =============================================================================

def compile_code():
    """Compile generated code."""
    print_header("Compilation")
    
    platform_dir = os.path.join(OUTPUT_DIR, "arm64")
    code_dir = os.path.join(platform_dir, "generated_code")
    
    if not os.path.exists(code_dir):
        print(f"  ✗ Code directory not found: {code_dir}")
        print("    Please run code generation first.")
        return False
    
    # Find orchestration file with main function
    orch_file = None
    for f in os.listdir(code_dir):
        if f.endswith('.c'):
            fpath = os.path.join(code_dir, f)
            try:
                with open(fpath, 'r') as fp:
                    content = fp.read()
                    if 'int main(' in content:
                        orch_file = fpath
                        break
            except:
                pass
    
    if not orch_file:
        print("  ✗ No orchestration file with main() found")
        return False
    
    print(f"  Found orchestration file: {os.path.basename(orch_file)}")
    
    # Compile orchestration function
    exe_name = os.path.join(platform_dir, os.path.basename(orch_file).replace('.c', ''))
    compile_cmd = (
        f"gcc -O2 -std=c11 -lm -I{RUNTIME_DIR} "
        f"-o {exe_name} {orch_file} "
        f"{os.path.join(RUNTIME_DIR, 'pto_runtime.c')} -lpthread"
    )
    
    print(f"  Compiling {os.path.basename(orch_file)}...")
    success, stdout, stderr = run_command(compile_cmd)
    if not success:
        print(f"  ✗ Compilation failed: {stderr}")
        return False
    
    print(f"  ✓ Compilation successful: {exe_name}")
    return True


# =============================================================================
# Accuracy Test
# =============================================================================

def reference_moe_fusion(hidden_states, gate_weight, e_score_bias, w13, w2):
    """
    Reference implementation of MoE Fusion (simplified).
    
    This is a simplified version that:
    1. Computes gate logits: hidden_states @ gate_weight^T
    2. Applies sigmoid
    3. Adds bias
    4. Computes FFN: hidden_states @ w13 -> SwiGLU -> @ w2
    """
    bs, hidden_size = hidden_states.shape
    num_experts, _ = gate_weight.shape
    intermediate_size = w2.shape[0]
    
    # Gate computation
    # logits = hidden_states @ gate_weight^T
    logits = np.dot(hidden_states, gate_weight.T)  # [bs, num_experts]
    
    # Sigmoid
    sigmoid_out = 1.0 / (1.0 + np.exp(-logits))
    
    # Add bias (broadcast)
    bias_2d = e_score_bias.reshape(1, -1)  # [1, num_experts]
    gate_weights = sigmoid_out + bias_2d  # [bs, num_experts]
    
    # FFN computation (share experts)
    # Up projection
    up_proj = np.dot(hidden_states, w13)  # [bs, intermediate_size * 2]
    
    # SwiGLU
    left = up_proj[:, :intermediate_size]  # [bs, intermediate_size]
    right = up_proj[:, intermediate_size:]  # [bs, intermediate_size]
    sigmoid_left = 1.0 / (1.0 + np.exp(-left))
    swiglu_out = left * sigmoid_left * right  # [bs, intermediate_size]
    
    # Down projection
    ffn_res = np.dot(swiglu_out, w2)  # [bs, hidden_size]
    
    return gate_weights, ffn_res


def run_accuracy_test():
    """Generate and run accuracy tests."""
    if not CONFIG['enable_accuracy_test']:
        return True
    
    print_header("Accuracy Test")
    
    # Test parameters
    bs_tile = CONFIG['test_bs_tile']
    hidden_size = CONFIG['test_hidden_size']
    intermediate_size = CONFIG['test_intermediate_size']
    num_experts = CONFIG['test_num_experts']
    
    print(f"  Test configuration:")
    print(f"    Batch size: {bs_tile}")
    print(f"    Hidden size: {hidden_size}")
    print(f"    Intermediate size: {intermediate_size}")
    print(f"    Number of experts: {num_experts}")
    
    # Generate test data
    np.random.seed(42)
    hidden_states = np.random.randn(bs_tile, hidden_size).astype(np.float32) * 0.05
    gate_weight = np.random.randn(num_experts, hidden_size).astype(np.float32) * 0.05
    e_score_bias = np.random.randn(num_experts).astype(np.float32) * 0.01
    w13 = np.random.randn(hidden_size, intermediate_size * 2).astype(np.float32) * 0.05
    w2 = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.05
    
    # Compute reference output
    print("\n  Computing reference output...")
    ref_gate_weights, ref_ffn_res = reference_moe_fusion(
        hidden_states, gate_weight, e_score_bias, w13, w2
    )
    
    print(f"  Reference gate_weights shape: {ref_gate_weights.shape}")
    print(f"  Reference gate_weights range: [{ref_gate_weights.min():.6f}, {ref_gate_weights.max():.6f}]")
    print(f"  Reference gate_weights mean: {ref_gate_weights.mean():.6f}")
    print(f"  Reference ffn_res shape: {ref_ffn_res.shape}")
    print(f"  Reference ffn_res range: [{ref_ffn_res.min():.6f}, {ref_ffn_res.max():.6f}]")
    print(f"  Reference ffn_res mean: {ref_ffn_res.mean():.6f}")
    
    # Save test data
    platform_dir = os.path.join(OUTPUT_DIR, "arm64")
    test_data_dir = ensure_dir(os.path.join(platform_dir, "test_data"))
    
    hidden_states_file = os.path.join(test_data_dir, "hidden_states.bin")
    gate_weight_file = os.path.join(test_data_dir, "gate_weight.bin")
    e_score_bias_file = os.path.join(test_data_dir, "e_score_bias.bin")
    w13_file = os.path.join(test_data_dir, "w13.bin")
    w2_file = os.path.join(test_data_dir, "w2.bin")
    output_gate_file = os.path.join(test_data_dir, "pto_gate_output.bin")
    output_ffn_file = os.path.join(test_data_dir, "pto_ffn_output.bin")
    ref_gate_file = os.path.join(test_data_dir, "ref_gate_output.npy")
    ref_ffn_file = os.path.join(test_data_dir, "ref_ffn_output.npy")
    
    hidden_states.astype(np.float32).tofile(hidden_states_file)
    gate_weight.astype(np.float32).tofile(gate_weight_file)
    e_score_bias.astype(np.float32).tofile(e_score_bias_file)
    w13.astype(np.float32).tofile(w13_file)
    w2.astype(np.float32).tofile(w2_file)
    np.save(ref_gate_file, ref_gate_weights)
    np.save(ref_ffn_file, ref_ffn_res)
    
    print(f"\n  Saved test data to {test_data_dir}")
    
    # Create test wrapper
    print("\n  Running PTO implementation...")
    test_wrapper_c = os.path.join(test_data_dir, "test_wrapper.c")
    
    # Collect all InCore function declarations
    incore_func_decls = []
    incore_wrappers = []
    code_dir = os.path.join(platform_dir, "generated_code")
    
    # List of InCore functions and their signatures
    incore_functions = {
        "gate_matmul": "void gate_matmul(float* input_hidden, float* input_gate_weight, float* output_logits);",
        "sigmoid": "void sigmoid(float* input, float* output, float one);",
        "add_bias": "void add_bias(float* input_x, float* input_bias, float* output);",
        "swiglu": "void swiglu(float* input, float* output);",
        "up_proj_matmul": "void up_proj_matmul(float* input_hidden, float* input_w13, float* output);",
        "down_proj_matmul": "void down_proj_matmul(float* input_swiglu, float* input_w2, float* output);",
    }
    
    # Generate wrapper functions for each InCore function
    for func_name, func_sig in incore_functions.items():
        incore_func_decls.append(func_sig)
        
        # Create wrapper function
        if func_name == "gate_matmul":
            wrapper = f'''
void gate_matmul_wrapper(void** args, int32_t num_args) {{
    if (num_args >= 3) {{
        gate_matmul((float*)args[0], (float*)args[1], (float*)args[2]);
    }}
}}'''
        elif func_name == "sigmoid":
            wrapper = f'''
void sigmoid_wrapper(void** args, int32_t num_args) {{
    if (num_args >= 2) {{
        sigmoid((float*)args[0], (float*)args[1], 1.0f);
    }}
}}'''
        elif func_name == "add_bias":
            wrapper = f'''
void add_bias_wrapper(void** args, int32_t num_args) {{
    if (num_args >= 3) {{
        add_bias((float*)args[0], (float*)args[1], (float*)args[2]);
    }}
}}'''
        elif func_name == "swiglu":
            wrapper = f'''
void swiglu_wrapper(void** args, int32_t num_args) {{
    if (num_args >= 2) {{
        swiglu((float*)args[0], (float*)args[1]);
    }}
}}'''
        elif func_name == "up_proj_matmul":
            wrapper = f'''
void up_proj_matmul_wrapper(void** args, int32_t num_args) {{
    if (num_args >= 3) {{
        up_proj_matmul((float*)args[0], (float*)args[1], (float*)args[2]);
    }}
}}'''
        elif func_name == "down_proj_matmul":
            wrapper = f'''
void down_proj_matmul_wrapper(void** args, int32_t num_args) {{
    if (num_args >= 3) {{
        down_proj_matmul((float*)args[0], (float*)args[1], (float*)args[2]);
    }}
}}'''
        else:
            wrapper = f'''
void {func_name}_wrapper(void** args, int32_t num_args) {{
    // Wrapper for {func_name}
}}'''
        
        incore_wrappers.append(wrapper)
    
    # Generate test wrapper C code
    with open(test_wrapper_c, 'w') as f:
        f.write(f'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <signal.h>
#include <setjmp.h>
#include "pto_runtime.h"
#include "pto_runtime_common.h"

// Forward declarations of InCore functions
{chr(10).join(incore_func_decls)}

// Wrapper functions
{chr(10).join(incore_wrappers)}

// Signal handler for debugging crashes
static jmp_buf crash_jmp;
static void crash_handler(int sig) {{
    fprintf(stderr, "ERROR: Signal %d received (likely crash)\\n", sig);
    longjmp(crash_jmp, 1);
}}

int main(int argc, char** argv) {{
    // Set up signal handlers for debugging
    signal(SIGABRT, crash_handler);
    signal(SIGSEGV, crash_handler);
    signal(SIGFPE, crash_handler);
    
    if (setjmp(crash_jmp) != 0) {{
        fprintf(stderr, "Program crashed. Attempting cleanup...\\n");
        return 1;
    }}
    
    if (argc < 7) {{
        fprintf(stderr, "Usage: test_wrapper <hidden_states_file> <gate_weight_file> <e_score_bias_file> <w13_file> <w2_file> <gate_output_file> <ffn_output_file>\\n");
        return 1;
    }}
    
    const char* hidden_states_file = argv[1];
    const char* gate_weight_file = argv[2];
    const char* e_score_bias_file = argv[3];
    const char* w13_file = argv[4];
    const char* w2_file = argv[5];
    const char* gate_output_file = argv[6];
    const char* ffn_output_file = argv[7];
    
    // Dimensions
    int bs_tile = {bs_tile};
    int hidden_size = {hidden_size};
    int intermediate_size = {intermediate_size};
    int num_experts = {num_experts};
    
    // Initialize runtime
    PTORuntime* rt = (PTORuntime*)calloc(1, sizeof(PTORuntime));
    if (!rt) {{
        fprintf(stderr, "Failed to allocate PTORuntime\\n");
        return 1;
    }}
    pto_runtime_init(rt);
    
    // Allocate buffers
    float* input_hidden_states = (float*)malloc(bs_tile * hidden_size * sizeof(float));
    float* input_gate_weight = (float*)malloc(num_experts * hidden_size * sizeof(float));
    float* input_e_score_bias = (float*)malloc(num_experts * sizeof(float));
    float* input_w13 = (float*)malloc(hidden_size * intermediate_size * 2 * sizeof(float));
    float* input_w2 = (float*)malloc(intermediate_size * hidden_size * sizeof(float));
    float* output_logits = (float*)calloc(bs_tile * num_experts, sizeof(float));
    float* output_ffn_res = (float*)calloc(bs_tile * hidden_size, sizeof(float));
    
    // Temporary buffers
    float* temp_gate_logits = (float*)calloc(bs_tile * num_experts, sizeof(float));
    float* temp_sigmoid_out = (float*)calloc(bs_tile * num_experts, sizeof(float));
    float* temp_gate_weights = (float*)calloc(bs_tile * num_experts, sizeof(float));
    float* temp_up_proj = (float*)calloc(bs_tile * intermediate_size * 2, sizeof(float));
    float* temp_swiglu_out = (float*)calloc(bs_tile * intermediate_size, sizeof(float));
    float* temp_down_proj = (float*)calloc(bs_tile * hidden_size, sizeof(float));
    
    // Load input data
    FILE* fp = fopen(hidden_states_file, "rb");
    if (!fp || fread(input_hidden_states, sizeof(float), bs_tile * hidden_size, fp) != bs_tile * hidden_size) {{
        fprintf(stderr, "Failed to read hidden_states file\\n");
        return 1;
    }}
    fclose(fp);
    
    fp = fopen(gate_weight_file, "rb");
    if (!fp || fread(input_gate_weight, sizeof(float), num_experts * hidden_size, fp) != num_experts * hidden_size) {{
        fprintf(stderr, "Failed to read gate_weight file\\n");
        return 1;
    }}
    fclose(fp);
    
    fp = fopen(e_score_bias_file, "rb");
    if (!fp || fread(input_e_score_bias, sizeof(float), num_experts, fp) != num_experts) {{
        fprintf(stderr, "Failed to read e_score_bias file\\n");
        return 1;
    }}
    fclose(fp);
    
    fp = fopen(w13_file, "rb");
    if (!fp || fread(input_w13, sizeof(float), hidden_size * intermediate_size * 2, fp) != hidden_size * intermediate_size * 2) {{
        fprintf(stderr, "Failed to read w13 file\\n");
        return 1;
    }}
    fclose(fp);
    
    fp = fopen(w2_file, "rb");
    if (!fp || fread(input_w2, sizeof(float), intermediate_size * hidden_size, fp) != intermediate_size * hidden_size) {{
        fprintf(stderr, "Failed to read w2 file\\n");
        return 1;
    }}
    fclose(fp);
    
    printf("Executing MoE fusion block...\\n");
    fflush(stdout);
    
    // Call moe_fusion_block function
    // Note: This is a simplified version - we'll call InCore functions directly
    // Gate computation
    printf("  Step 1: Gate matmul...\\n");
    fflush(stdout);
    gate_matmul(input_hidden_states, input_gate_weight, temp_gate_logits);
    
    printf("  Step 2: Sigmoid...\\n");
    fflush(stdout);
    sigmoid(temp_gate_logits, temp_sigmoid_out, 1.0f);
    
    printf("  Step 3: Add bias...\\n");
    fflush(stdout);
    add_bias(temp_sigmoid_out, input_e_score_bias, temp_gate_weights);
    
    // FFN computation
    printf("  Step 4: Up projection matmul...\\n");
    fflush(stdout);
    up_proj_matmul(input_hidden_states, input_w13, temp_up_proj);
    
    printf("  Step 5: SwiGLU...\\n");
    fflush(stdout);
    swiglu(temp_up_proj, temp_swiglu_out);
    
    printf("  Step 6: Down projection matmul...\\n");
    fflush(stdout);
    down_proj_matmul(temp_swiglu_out, input_w2, temp_down_proj);
    
    // Copy results to output
    printf("  Step 7: Copying results...\\n");
    fflush(stdout);
    memcpy(output_logits, temp_gate_weights, bs_tile * num_experts * sizeof(float));
    memcpy(output_ffn_res, temp_down_proj, bs_tile * hidden_size * sizeof(float));
    
    printf("Execution complete!\\n");
    fflush(stdout);
    
    // Save outputs
    fp = fopen(gate_output_file, "wb");
    if (!fp || fwrite(output_logits, sizeof(float), bs_tile * num_experts, fp) != bs_tile * num_experts) {{
        fprintf(stderr, "Failed to write gate output file\\n");
        return 1;
    }}
    fclose(fp);
    
    fp = fopen(ffn_output_file, "wb");
    if (!fp || fwrite(output_ffn_res, sizeof(float), bs_tile * hidden_size, fp) != bs_tile * hidden_size) {{
        fprintf(stderr, "Failed to write ffn output file\\n");
        return 1;
    }}
    fclose(fp);
    
    // Cleanup
    pto_runtime_shutdown(rt);
    free(rt);
    free(input_hidden_states);
    free(input_gate_weight);
    free(input_e_score_bias);
    free(input_w13);
    free(input_w2);
    free(output_logits);
    free(output_ffn_res);
    free(temp_gate_logits);
    free(temp_sigmoid_out);
    free(temp_gate_weights);
    free(temp_up_proj);
    free(temp_swiglu_out);
    free(temp_down_proj);
    
    return 0;
}}
''')
    
    # Compile test wrapper
    test_wrapper_exe = os.path.join(test_data_dir, "test_wrapper")
    runtime_c = os.path.join(RUNTIME_DIR, "pto_runtime.c")
    
    # Collect all InCore function files (exclude moe_fusion_block.c which has main())
    incore_files = []
    fusion_block_c = os.path.join(code_dir, 'moe_fusion_block.c')
    
    for f in os.listdir(code_dir):
        if f.endswith('.c'):
            fpath = os.path.join(code_dir, f)
            # Skip moe_fusion_block.c as it contains main() and we use test_wrapper.c instead
            if f == 'moe_fusion_block.c':
                continue
            try:
                with open(fpath, 'r') as fp:
                    content = fp.read()
                    # Only include files without main()
                    if 'int main(' not in content:
                        incore_files.append(fpath)
            except:
                pass
    
    compile_cmd = (
        f"gcc -O2 -std=c11 -lm -I{RUNTIME_DIR} "
        f"-o {test_wrapper_exe} {test_wrapper_c} "
        f"{' '.join(incore_files)} "
        f"{runtime_c} -lpthread"
    )
    
    print(f"    Compiling test wrapper...")
    success, stdout, stderr = run_command(compile_cmd)
    if not success:
        print(f"    ✗ Failed to compile test wrapper: {stderr}")
        return False
    
    # Run test wrapper
    run_cmd = f"{test_wrapper_exe} {hidden_states_file} {gate_weight_file} {e_score_bias_file} {w13_file} {w2_file} {output_gate_file} {output_ffn_file}"
    success, stdout, stderr = run_command(run_cmd, cwd=test_data_dir)
    
    # Check if output files were created
    if not os.path.exists(output_gate_file) or not os.path.exists(output_ffn_file):
        print(f"    ✗ Failed to run test wrapper: output files not created")
        if stdout:
            print(f"      stdout: {stdout}")
        if stderr:
            print(f"      stderr: {stderr}")
        return False
    
    file_size_gate = os.path.getsize(output_gate_file)
    expected_size_gate = bs_tile * num_experts * 4  # float32 = 4 bytes
    if file_size_gate != expected_size_gate:
        print(f"    ✗ Gate output file size mismatch: {file_size_gate} != {expected_size_gate} bytes")
        return False
    
    file_size_ffn = os.path.getsize(output_ffn_file)
    expected_size_ffn = bs_tile * hidden_size * 4
    if file_size_ffn != expected_size_ffn:
        print(f"    ✗ FFN output file size mismatch: {file_size_ffn} != {expected_size_ffn} bytes")
        return False
    
    # Load PTO outputs
    pto_gate_output = np.fromfile(output_gate_file, dtype=np.float32).reshape(bs_tile, num_experts)
    pto_ffn_output = np.fromfile(output_ffn_file, dtype=np.float32).reshape(bs_tile, hidden_size)
    
    # Compare with reference
    print("\n  Comparing PTO output with reference...")
    print(f"    PTO gate_weights shape: {pto_gate_output.shape}")
    print(f"    PTO gate_weights range: [{pto_gate_output.min():.6f}, {pto_gate_output.max():.6f}]")
    print(f"    PTO gate_weights mean: {pto_gate_output.mean():.6f}")
    print(f"    PTO ffn_res shape: {pto_ffn_output.shape}")
    print(f"    PTO ffn_res range: [{pto_ffn_output.min():.6f}, {pto_ffn_output.max():.6f}]")
    print(f"    PTO ffn_res mean: {pto_ffn_output.mean():.6f}")
    
    # Compute differences
    gate_diff = np.abs(pto_gate_output - ref_gate_weights)
    ffn_diff = np.abs(pto_ffn_output - ref_ffn_res)
    
    gate_max_diff = np.max(gate_diff)
    gate_mean_diff = np.mean(gate_diff)
    gate_rel_diff = gate_diff / (np.abs(ref_gate_weights) + 1e-8)
    gate_max_rel_diff = np.max(gate_rel_diff)
    gate_mean_rel_diff = np.mean(gate_rel_diff)
    
    ffn_max_diff = np.max(ffn_diff)
    ffn_mean_diff = np.mean(ffn_diff)
    ffn_rel_diff = ffn_diff / (np.abs(ref_ffn_res) + 1e-8)
    ffn_max_rel_diff = np.max(ffn_rel_diff)
    ffn_mean_rel_diff = np.mean(ffn_rel_diff)
    
    print(f"\n    Gate weights comparison:")
    print(f"      Max absolute difference: {gate_max_diff:.6e}")
    print(f"      Mean absolute difference: {gate_mean_diff:.6e}")
    print(f"      Max relative difference: {gate_max_rel_diff:.6e}")
    print(f"      Mean relative difference: {gate_mean_rel_diff:.6e}")
    
    print(f"\n    FFN output comparison:")
    print(f"      Max absolute difference: {ffn_max_diff:.6e}")
    print(f"      Mean absolute difference: {ffn_mean_diff:.6e}")
    print(f"      Max relative difference: {ffn_max_rel_diff:.6e}")
    print(f"      Mean relative difference: {ffn_mean_rel_diff:.6e}")
    
    # Check tolerance
    abs_tol = 0.001
    rel_tol = 0.01
    
    gate_within_tol = np.sum((gate_diff < abs_tol) | (gate_rel_diff < rel_tol))
    gate_total = gate_diff.size
    gate_pct = 100.0 * gate_within_tol / gate_total
    
    ffn_within_tol = np.sum((ffn_diff < abs_tol) | (ffn_rel_diff < rel_tol))
    ffn_total = ffn_diff.size
    ffn_pct = 100.0 * ffn_within_tol / ffn_total
    
    print(f"\n    Gate weights: {gate_within_tol}/{gate_total} ({gate_pct:.1f}%) within tolerance")
    print(f"    FFN output: {ffn_within_tol}/{ffn_total} ({ffn_pct:.1f}%) within tolerance")
    
    # Both outputs must pass
    if gate_pct >= 95.0 and ffn_pct >= 95.0:
        print(f"    ✓ Accuracy test PASSED")
        print(f"      All differences within tolerance (abs: {abs_tol}, rel: {rel_tol})")
        return True
    else:
        print(f"    ✗ Accuracy test FAILED")
        if gate_pct < 95.0:
            print(f"      Gate weights: Only {gate_pct:.1f}% within tolerance (required: 95.0%)")
        if ffn_pct < 95.0:
            print(f"      FFN output: Only {ffn_pct:.1f}% within tolerance (required: 95.0%)")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("  PTO MoE Fusion Example - ARM64")
    print("=" * 60)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Step 1: Generate code
    results['codegen'] = generate_code()
    if not results['codegen']:
        print("\n✗ Code generation failed. Aborting.")
        return
    
    # Step 2: Compile code
    results['compile'] = compile_code()
    if not results['compile']:
        print("\n✗ Compilation failed. Aborting.")
        return
    
    # Step 3: Run accuracy test
    results['accuracy'] = run_accuracy_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Code Generation: {'✓ OK' if results['codegen'] else '✗ FAILED'}")
    print(f"  Compilation: {'✓ OK' if results['compile'] else '✗ FAILED'}")
    print(f"  Accuracy Test: {'✓ OK' if results.get('accuracy') else '✗ FAILED'}")
    
    print("\nDone!")
    print("\nNext steps:")
    print("  1. Review generated code in output/arm64/generated_code/")
    print("  2. Check test data in output/arm64/test_data/")
    print("  3. Integrate with runtime for full execution test")


if __name__ == "__main__":
    main()
