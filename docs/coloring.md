# PTO Instruction Graph Simplification and Coloring

This document describes the `SimplifyAndColor()` method for analyzing instruction dependencies, simplifying the dependency graph, and performing graph coloring on PTO programs.

## Overview

The `SimplifyAndColor()` method performs three main operations:

1. **Dependency Analysis**: Builds a hypergraph representing instruction dependencies
2. **Graph Simplification**: Removes transitive (redundant) edges
3. **Graph Coloring**: Assigns colors to instructions such that no two dependent instructions share the same color

This analysis is useful for:
- Understanding data flow in PTO programs
- Register allocation optimization
- Instruction scheduling
- Parallel execution planning

## Algorithm Details

### 1. Dependency Analysis

The dependency analyzer identifies three types of data dependencies:

| Dependency Type | Abbreviation | Description |
|-----------------|--------------|-------------|
| Read After Write | RAW | True dependency: instruction reads a value written by a previous instruction |
| Write After Write | WAW | Output dependency: two instructions write to the same location |
| Write After Read | WAR | Anti-dependency: instruction writes to a location read by a previous instruction |

#### Handling Indirect Memory Access

For instructions with variable (indirect) memory addressing (e.g., `tload %input[%i, 0]`), the analyzer conservatively assumes dependencies with all potential conflicting memory operations.

#### Loop-Carried Dependencies

For loops, the analyzer detects **loop-carried dependencies** where:
- A write in iteration N affects a read in iteration N+1
- These create cyclic edges in the dependency graph (stored in `fanin_succ`)

```
FOR %i = 0 to N:
    %x = tload %input[%i, 0]     // Depends on previous iteration's write
    %y = tmul %x, %acc
    %acc = tadd %acc, %y         // Writes to %acc, affects next iteration
ENDFOR
```

### 2. Graph Simplification

The simplification algorithm removes **transitive edges** to reduce graph complexity while preserving correctness.

**Rule**: If edges A→B, B→C, and A→C all exist, remove A→C (it's redundant).

```
Before:          After:
A ──→ B          A ──→ B
│     │                │
│     ▼                ▼
└───→ C          C
```

This is applied iteratively until no more transitive edges can be removed.

### 3. Graph Coloring

The coloring algorithm assigns colors (0 to TOTAL_COLOR-1) to instructions such that **no two adjacent instructions share the same color**.

#### Algorithm Steps:

1. **Check Maximum Degree**: Find the node with the highest number of neighbors
2. **Degree Reduction** (if needed): If max_degree ≥ TOTAL_COLOR-1:
   - Find two neighbors A and B that are not connected
   - Add edge A→B
   - Remove edge A→hot_node (reduces hot node's degree)
3. **Greedy Coloring**: 
   - Sort nodes by degree (highest first)
   - Assign each node the smallest color not used by its neighbors

## Data Structures

### PTOInstruction Fields

Each instruction has the following dependency-related fields:

```python
instr_id: int           # Unique instruction index
fanin_pred: List[int]   # Predecessor instruction IDs (dependencies)
fanin_succ: List[int]   # Successor instruction IDs (loop-carried deps)
color: int              # Assigned color (-1 = uncolored)
```

### Accessing Dependency Information

```python
instr = program.instructions[i]

# Get all neighbors (for graph algorithms)
neighbors = instr.get_all_neighbors()

# Get degree (number of neighbors)
degree = instr.get_degree()

# Check specific dependencies
predecessors = instr.fanin_pred   # Instructions this depends on
successors = instr.fanin_succ     # Instructions that depend on this (cyclic)
```

## Usage

### Basic Usage

```python
from pto_compile import PTOFunctionBuilder, PTOModule
from pto_isa_definition import ElementType, MemorySpace

# Build a PTO program
program = (PTOFunctionBuilder("example")
    .in_core()
    .tile("a", 8, 8, ElementType.F32)
    .tile("b", 8, 8, ElementType.F32)
    .memref("input", MemorySpace.GM, ElementType.F32)
    .memref("output", MemorySpace.GM, ElementType.F32)
    .load("a", "input", 0, 0)
    .exp("b", "a")
    .store("b", "output", 0, 0)
    .build())

# Run SimplifyAndColor
success = program.SimplifyAndColor(
    TOTAL_COLOR=8,        # Number of colors available (default: 8)
    output_dir="./output", # Directory for PDF output
    visualize=True,       # Generate PDF visualizations
    verbose=True          # Print debug information
)

print(f"Coloring succeeded: {success}")
```

### Exporting Results

#### PTO Assembly with Dependency Info

```python
# Generate PTO assembly with dependency annotations
asm = program.dump_pto_asm_with_deps("output.pto")
print(asm)
```

Output format:
```
%a = tload %input[0, 0] : ...  // id=0 color=1 succ=[1]
%b = texp %a : ...             // id=1 color=0 pred=[0] succ=[2]
tstore %b, %output[0, 0]       // id=2 color=1 pred=[1]
```

#### Visualization PDFs

```python
# Generate individual graph visualization
program.plot_program_graph(
    "deps_colored.pdf",
    title="My Program Dependencies",
    show_colors=True
)
```

### Processing Multiple Functions (Module)

```python
from pto_llama7B_dynamic import create_llama7b_module

# Create module with multiple functions
module = create_llama7b_module()

# Process each function
for func_name, program in module.functions.items():
    if len(program.instructions) > 0:
        success = program.SimplifyAndColor(
            TOTAL_COLOR=8,
            output_dir=f"./output/{func_name}",
            visualize=True
        )
        program.dump_pto_asm_with_deps(f"./output/{func_name}.pto")
```

## Output Files

When `visualize=True`, the following PDF files are generated:

| File | Description |
|------|-------------|
| `{name}_deps_original.pdf` | Original dependency graph with all edges |
| `{name}_deps_simplified.pdf` | Graph after transitive edge removal |
| `{name}_deps_colored.pdf` | Final graph with color assignments |
| `{name}_deps_comparison.pdf` | All three graphs combined for comparison |

### PDF Legend

- **Blue solid arrows**: Predecessor dependencies (RAW/WAW/WAR)
- **Red dashed arrows**: Successor dependencies (loop-carried, cyclic)
- **Node colors**: Assigned colors from the coloring algorithm
- **Labels**: `[id] OPCODE cN dM` where N=color, M=degree

## Example: LLaMA Analysis

Run the provided script to analyze all LLaMA 7B functions:

```bash
cd /path/to/PTO_ISA_Compiler
python examples/run_llama_simplify_color.py
```

This will:
1. Create the LLaMA 7B module (46 functions)
2. Run SimplifyAndColor on each function
3. Generate PDFs in `examples/output_pto/llama7b_colored/`
4. Generate colored PTO assembly files

### Sample Output

```
======================================================================
LLaMA 7B SimplifyAndColor Analysis
======================================================================

Module: llama7b_flash
Total functions: 46
  InCore functions: 45
  Orchestration functions: 1

Function                                       Type   Instrs   MaxDeg   Colors
------------------------------------------------------------------------------
flash_attn_softmax_update                    InCore       16        5        2
rmsnorm_tile_64                              InCore       12        3        3
swiglu_tile_64                               InCore        9        3        2
llama_layer_dynamic                      Orchestration       29        0        1
```

## API Reference

### PTOProgram.SimplifyAndColor()

```python
def SimplifyAndColor(
    self,
    TOTAL_COLOR: int = 8,      # Number of colors available
    output_dir: str = ".",     # Output directory for PDFs
    visualize: bool = True,    # Generate PDF visualizations
    verbose: bool = False      # Print debug information
) -> bool:
    """
    Analyze dependencies, simplify graph, and perform coloring.
    
    Returns:
        True if coloring succeeded (all nodes colored), False otherwise
    """
```

### PTOProgram.plot_program_graph()

```python
def plot_program_graph(
    self,
    output_path: str,          # Path for output PDF
    title: str = None,         # Graph title (default: program name)
    show_colors: bool = False  # Color nodes by assigned color
):
    """Generate PDF visualization of the dependency graph."""
```

### PTOProgram.dump_pto_asm_with_deps()

```python
def dump_pto_asm_with_deps(
    self,
    output_path: str = None    # Optional output file path
) -> str:
    """
    Generate PTO assembly with dependency annotations.
    
    Returns:
        Assembly string with // id=N color=C pred=[...] succ=[...] comments
    """
```

### PTOInstruction Methods

```python
instr.init_dependency_fields(instr_id)  # Initialize fields
instr.add_pred(pred_id)                  # Add predecessor dependency
instr.add_succ(succ_id)                  # Add successor dependency
instr.get_all_neighbors() -> List[int]   # Get all neighbor IDs
instr.get_degree() -> int                # Get number of neighbors
instr.to_pto_as_with_deps() -> str       # Get assembly with dep info
```

## Limitations

1. **Conservative Analysis**: Indirect memory accesses are handled conservatively, which may result in more dependencies than strictly necessary.

2. **Greedy Coloring**: The coloring algorithm is greedy and may not find the optimal coloring in all cases.

3. **Fixed Color Count**: If the graph requires more colors than TOTAL_COLOR, the algorithm will attempt degree reduction but may still fail.

## Dependencies

- **matplotlib**: Required for PDF visualization
- **PyPDF2**: Optional, for generating combined comparison PDFs
