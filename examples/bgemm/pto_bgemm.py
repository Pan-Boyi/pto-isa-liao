#!/usr/bin/env python3
"""
BGEMM example

This module provides two entry points:
1. create_bgemm_module() - For code generation using PTOFunctionBuilder API
2. main() - For direct NPU runtime execution (requires pto_as module)

For Ascend A2A3 simulation, use create_bgemm_module() which generates:
- InCore functions: gemm_tile (Cube), tile_add (Vector)
- Orchestration function: bgemm_dynamic

For NPU runtime execution, main() calls:
  `kernels/python/bgemm_performance/run_runtime.py`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# =============================================================================
# Code Generation Entry Point (PTOFunctionBuilder API)
# =============================================================================

def create_bgemm_module():
    """
    Create BGEMM module for code generation.
    
    This is the preferred entry point for:
    - Ascend A2A3 simulation (ascend_a2a3_sim)
    - Ascend A2A3 hardware (ascend_a2a3)
    
    Returns:
        PTOModule with InCore and Orchestration functions
    """
    # Import from the implementation file (same directory)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pto_bgemm_func",
        Path(__file__).parent / "pto_bgemm_func.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_bgemm_module()


# =============================================================================
# NPU Runtime Entry Point (pto_as API - requires external module)
# =============================================================================

def main() -> int:
    """
    Run BGEMM on NPU using pto_as runtime flow.
    
    Note: This requires the pto_as module to be installed.
    """
    repo_root = _repo_root()
    sys.path.insert(0, os.fspath(repo_root))

    from kernels.python.bgemm_performance.run_runtime import main as bgemm_main

    return int(bgemm_main())


if __name__ == "__main__":
    raise SystemExit(main())
