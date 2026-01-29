"""
PTO BGEMM - Blocked General Matrix Multiply with L2 Cache Optimization

Implements tiled matrix multiplication: C[M×N] = A[M×K] @ B[K×N]

=============================================================================
L2 CACHE OPTIMIZATION: SWIZZLE PATTERN
=============================================================================

Problem with naive loop order (m -> n -> k):
  - A[m,k] tiles loaded for each k, evicted before reuse across n
  - B[k,n] tiles loaded for each k, evicted before reuse across m
  - Poor L2 cache utilization

Solution: 2D Super-tile Swizzle with K-streaming
  - Group output tiles into super-tiles (SWIZZLE_M × SWIZZLE_N)
  - Process one K-slice at a time within super-tile
  - Maximizes reuse of A and B tiles in L2 cache

Loop order comparison:

  NAIVE (poor cache):              SWIZZLED (good cache):
  for m in M:                      for sm in range(0, M, SWIZZLE_M):
    for n in N:                      for sn in range(0, N, SWIZZLE_N):
      for k in K:                      for k in K:  ← K in middle!
        C[m,n] += A[m,k]@B[k,n]          for lm in SWIZZLE_M:
                                           for ln in SWIZZLE_N:
                                             m, n = sm+lm, sn+ln
                                             C[m,n] += A[m,k]@B[k,n]

Cache benefits:
  - A[m,k] loaded once, reused for SWIZZLE_N different n values
  - B[k,n] loaded once, reused for SWIZZLE_M different m values
  - Total L2 loads reduced by factor of ~(SWIZZLE_M + SWIZZLE_N) / 2

Typical swizzle sizes: 2×2, 4×4, 8×8 depending on L2 cache size

=============================================================================
"""

import os
import sys

# Add src directory to path
_example_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_example_dir))
_src_dir = os.path.join(_root_dir, 'src')
sys.path.insert(0, _src_dir)

from compile.pto_compile import PTOFunctionBuilder, PTOModule
from isa_definition.pto_isa_definition import ElementType, MemorySpace

# =============================================================================
# Configuration
# =============================================================================

DTYPE = ElementType.F32

# Tile dimensions (single tile size)
TILE_M = 64   # Output tile rows
TILE_K = 64   # Inner dimension tile size
TILE_N = 64   # Output tile cols

# Swizzle configuration for L2 cache optimization
# Super-tile = SWIZZLE_M × SWIZZLE_N output tiles processed together
# Larger swizzle = better cache reuse, but need more L2 capacity
SWIZZLE_M = 4   # Super-tile height in tiles
SWIZZLE_N = 4   # Super-tile width in tiles

# =============================================================================
# Level 1: InCore Functions
# =============================================================================

def create_gemm_tile(m=TILE_M, k=TILE_K, n=TILE_N, dtype=DTYPE):
    """
    InCore (Cube): Matrix multiply tile.
    
    Computes: output[m×n] = input_a[m×k] @ input_b[k×n]
    """
    return (PTOFunctionBuilder("gemm_tile")
        .in_core()
        .cube(True)  # Cube Core (AIC) for matmul
        
        .tile("a", m, k, dtype)
        .tile("b", k, n, dtype)
        .tile("c", m, n, dtype)
        
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .matmul("c", "a", "b")
        .store("c", "output", 0, 0)
        
        .build())


def create_tile_add(rows=TILE_M, cols=TILE_N, dtype=DTYPE):
    """
    InCore (Vector): Element-wise addition for accumulation.
    
    Computes: output = input_a + input_b
    """
    return (PTOFunctionBuilder("tile_add")
        .in_core()
        .cube(False)  # Vector Core (AIV)
        
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


# =============================================================================
# Level 2: Orchestration Function with Swizzle Pattern
# =============================================================================

def create_bgemm_dynamic(module, dtype=DTYPE):
    """
    Orchestration: Batched Blocked GEMM with L2-optimized swizzle pattern.
    
    Computes: C[b,m,n] = Σ_k A[b,m,k] @ B[b,k,n]
    
    SWIZZLE PATTERN FOR L2 CACHE OPTIMIZATION:
    ==========================================
    
    Instead of processing output tiles in row-major order, we use a
    2D super-tile swizzle that maximizes data reuse in L2 cache.
    
    Loop structure:
      for batch in [0, b):                    # Batches independent
        for sm in [0, m, SWIZZLE_M):          # Super-tile row
          for sn in [0, n, SWIZZLE_N):        # Super-tile col
            for k_idx in [0, k):              # K-streaming (middle!)
              for lm in [0, SWIZZLE_M):       # Local tile row
                for ln in [0, SWIZZLE_N):     # Local tile col
                  m_idx = sm + lm
                  n_idx = sn + ln
                  P[m,n] = A[m,k] @ B[k,n]    # Cube
                  C[m,n] += P[m,n]            # Vector
    
    Why this order maximizes L2 hit rate:
      - For a fixed k_idx within super-tile:
        - A[m_idx, k_idx] is loaded and reused for all SWIZZLE_N values of n
        - B[k_idx, n_idx] is loaded and reused for all SWIZZLE_M values of m
      - Total L2 traffic reduced by ~(SWIZZLE_M × SWIZZLE_N) / (SWIZZLE_M + SWIZZLE_N)
    
    Example with SWIZZLE_M=4, SWIZZLE_N=4:
      - Each A tile reused 4× (across n dimension)
      - Each B tile reused 4× (across m dimension)  
      - L2 traffic reduced by ~2× compared to naive order
    
    Parameters:
      A, B, C: Matrices [b × m × k], [b × k × n], [b × m × n]
      P: Temporary buffer for partial products
      b: Batch count
      m: M-dimension tiles (must be multiple of SWIZZLE_M for best perf)
      k: K-dimension tiles (reduction)
      n: N-dimension tiles (must be multiple of SWIZZLE_N for best perf)
    """
    builder = (PTOFunctionBuilder("bgemm_dynamic", module=module)
        .not_in_core()
        
        # Matrix memory references
        .memref("A", MemorySpace.GM, dtype)    # [b * m * k] tiles
        .memref("B", MemorySpace.GM, dtype)    # [b * k * n] tiles
        .memref("C", MemorySpace.GM, dtype)    # [b * m * n] tiles
        .memref("P", MemorySpace.GM, dtype)    # [b * m * n] partial products
        
        # Dimension parameters
        .scalar("b", ElementType.I32)   # batch count
        .scalar("m", ElementType.I32)   # M-dimension tiles
        .scalar("k", ElementType.I32)   # K-dimension tiles (reduction)
        .scalar("n", ElementType.I32)   # N-dimension tiles
    )
    
    # ==========================================================================
    # SWIZZLED LOOP STRUCTURE FOR L2 CACHE OPTIMIZATION
    # ==========================================================================
    
    # Level 1: Batch loop (independent batches)
    builder.for_loop("batch", 0, "b", 1)
    
    # Level 2: Super-tile loops (coarse-grained parallelism)
    # Process m dimension in chunks of SWIZZLE_M
    builder.for_loop("sm", 0, "m", SWIZZLE_M)
    # Process n dimension in chunks of SWIZZLE_N  
    builder.for_loop("sn", 0, "n", SWIZZLE_N)
    
    # Level 3: K-dimension loop (MIDDLE position for cache optimization!)
    # By placing K in the middle, we stream through K dimension while
    # keeping the super-tile's worth of A and B tiles in L2 cache
    builder.for_loop("k_idx", 0, "k", 1)
    
    # Level 4: Local tile loops within super-tile
    # These tiles share A[m,k] and B[k,n] tiles loaded in L2
    builder.for_loop("lm", 0, SWIZZLE_M, 1)
    builder.for_loop("ln", 0, SWIZZLE_N, 1)
    
    # Compute actual tile indices
    # m_idx = sm + lm, n_idx = sn + ln
    # Note: Expression uses sm + lm directly
    
    # Step 1: Compute partial product (Cube Core)
    # A index: batch * (m * k) + (sm + lm) * k + k_idx
    # B index: batch * (k * n) + k_idx * n + (sn + ln)
    # P index: batch * (m * n) + (sm + lm) * n + (sn + ln)
    builder.call("gemm_tile", {
        "input_a": ("A", "batch * (m * k) + (sm + lm) * k + k_idx", 0),
        "input_b": ("B", "batch * (k * n) + k_idx * n + (sn + ln)", 0),
        "output": ("P", "batch * (m * n) + (sm + lm) * n + (sn + ln)", 0),
    })
    
    # Step 2: Accumulate to output (Vector Core)
    # C[m,n] = C[m,n] + P[m,n]
    # The dependency through C[m,n] ensures correct k-ordering
    builder.call("tile_add", {
        "input_a": ("C", "batch * (m * n) + (sm + lm) * n + (sn + ln)", 0),
        "input_b": ("P", "batch * (m * n) + (sm + lm) * n + (sn + ln)", 0),
        "output": ("C", "batch * (m * n) + (sm + lm) * n + (sn + ln)", 0),
    })
    
    builder.end_for()  # end ln loop
    builder.end_for()  # end lm loop
    builder.end_for()  # end k_idx loop
    builder.end_for()  # end sn loop
    builder.end_for()  # end sm loop
    builder.end_for()  # end batch loop
    
    return builder.build()


# =============================================================================
# Module Creation
# =============================================================================

def create_bgemm_module():
    """
    Create the complete BGEMM module with L2-optimized swizzle pattern.
    
    Functions:
      - gemm_tile (InCore, Cube): Single tile matrix multiply
      - tile_add (InCore, Vector): Tile addition for accumulation
      - bgemm_dynamic (Orchestration): Swizzled BGEMM for L2 cache optimization
    """
    module = PTOModule("bgemm_module")
    
    module.add_function(create_gemm_tile())
    module.add_function(create_tile_add())
    module.add_function(create_bgemm_dynamic(module))
    
    return module


# =============================================================================
# Main
# =============================================================================

def main():
    print("Creating BGEMM module with L2 cache optimization...")
    module = create_bgemm_module()
    
    print(f"\nModule: {module.name}")
    print(f"Functions: {len(module.functions)}")
    
    for name, func in module.functions.items():
        func_type = "InCore" if getattr(func, 'is_in_core', True) else "Orchestration"
        if func_type == "InCore":
            is_cube = getattr(func, 'is_cube', False)
            core_type = "Cube (AIC)" if is_cube else "Vector (AIV)"
            print(f"  - {name}: {func_type} [{core_type}]")
        else:
            print(f"  - {name}: {func_type}")
    
    print("\n" + "="*70)
    print("SWIZZLE PATTERN FOR L2 CACHE OPTIMIZATION")
    print("="*70)
    print(f"""
Configuration:
  SWIZZLE_M = {SWIZZLE_M} (super-tile height in tiles)
  SWIZZLE_N = {SWIZZLE_N} (super-tile width in tiles)

Loop Order Comparison:

  NAIVE (poor L2 hit rate):        SWIZZLED (optimized L2 hit rate):
  ─────────────────────────        ──────────────────────────────────
  for batch:                       for batch:
    for m_idx:                       for sm in [0, m, {SWIZZLE_M}):     ← super-tile
      for n_idx:                       for sn in [0, n, {SWIZZLE_N}):   ← super-tile
        for k_idx:  ← innermost          for k_idx:        ← MIDDLE!
          C += A @ B                       for lm in [0, {SWIZZLE_M}):  ← local
                                             for ln in [0, {SWIZZLE_N}):
                                               C += A @ B

Why swizzle improves L2 cache hit rate:

  NAIVE order:
  ┌─────────────────────────────────────────────────────────────────┐
  │ For C[0,0]: load A[0,0..K], B[0..K,0]                           │
  │ For C[0,1]: load A[0,0..K] (HIT!), B[0..K,1] (MISS - new col)   │
  │ ...                                                              │
  │ For C[1,0]: load A[1,0..K] (MISS), B[0..K,0] (MISS - evicted!)  │
  │ Problem: B tiles evicted before reuse across m dimension        │
  └─────────────────────────────────────────────────────────────────┘

  SWIZZLED order (processing {SWIZZLE_M}×{SWIZZLE_N} super-tile, k=0):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Load A[0,0], B[0,0] → C[0,0]                                    │
  │ Reuse A[0,0], load B[0,1] → C[0,1]  (A HIT!)                    │
  │ Reuse A[0,0], load B[0,2] → C[0,2]  (A HIT!)                    │
  │ Reuse A[0,0], load B[0,3] → C[0,3]  (A HIT!)                    │
  │ Load A[1,0], reuse B[0,0] → C[1,0]  (B HIT!)                    │
  │ Reuse A[1,0], reuse B[0,1] → C[1,1] (A+B HIT!)                  │
  │ ...                                                              │
  │ Each A tile used {SWIZZLE_N}×, each B tile used {SWIZZLE_M}×    │
  └─────────────────────────────────────────────────────────────────┘

Estimated L2 traffic reduction: ~{(SWIZZLE_M * SWIZZLE_N) / (SWIZZLE_M + SWIZZLE_N):.1f}× for A+B tiles

Task submission order (b=1, m=8, k=4, n=8):
  Super-tile [0,0]: k=0: C[0..3,0..3], k=1: C[0..3,0..3], ...
  Super-tile [0,1]: k=0: C[0..3,4..7], k=1: C[0..3,4..7], ...
  Super-tile [1,0]: k=0: C[4..7,0..3], ...
  Super-tile [1,1]: k=0: C[4..7,4..7], ...
""")
    
    return module


if __name__ == "__main__":
    main()
