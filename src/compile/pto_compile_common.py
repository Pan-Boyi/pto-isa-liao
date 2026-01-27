"""
PTO Compiler - Common Infrastructure (Platform Independent)

This module contains platform-independent compiler components:
- Error types and exceptions
- Symbol table for variable tracking
- Program and module representations
- DSL builder (PTOFunctionBuilder)
- Type checker
- Base code generator and optimizer
- Loop fusion infrastructure
- Tile buffer analyzer
- Record/replay optimization utilities

Platform-specific code generation is in separate modules:
- pto_codegen_arm64.py: ARM64/NEON code generation
- pto_codegen_cuda.py: CUDA code generation  
- pto_codegen_ascend.py: Ascend code generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from enum import Enum
import re
import os
import sys

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from isa_definition.pto_isa_definition import (
    # Types
    ElementType, MemorySpace, CompareMode, RoundMode, TMovMode,
    TileShape, TileType, MemRefType, EventType,
    
    # Operands
    TileOperand, ScalarOperand, MemRefOperand, IndexOperand, ImmediateOperand, Operand,
    
    # Base classes
    PTOInstruction, TileInstruction, ScalarInstruction, ControlFlowInstruction,
    
    # All instructions (for dynamic lookup)
    ALL_INSTRUCTIONS, TILE_INSTRUCTIONS, SCALAR_INSTRUCTIONS, CONTROL_FLOW_INSTRUCTIONS,
    
    # Loop constructs
    TileLoop, NestedTileLoop, FOR, ENDFOR, WHILE, DO, ENDWHILE, IF, ELSE, ENDIF,
    
    # Function call instructions
    CALL, RETURN,
    
    # Memory operations (manually handled - not auto-generated)
    TLOAD, TSTORE,
    
    # Instructions used by TypeChecker
    TMATMUL,
    
    # Scalar instructions
    SADD, SSUB, SMUL, SDIV, SMOV, SLI, SCMP,
    
    # Helper functions
    tile, scalar, index, memref, imm,
    
    # Instruction metadata for auto-generating builder methods
    INSTRUCTION_METADATA,
)

# Import for auto-generation
import isa_definition.pto_isa_definition as _pto_isa


# =============================================================================
# Compiler Error Types
# =============================================================================

class CompilerError(Exception):
    """Base class for compiler errors."""
    pass


class ParseError(CompilerError):
    """Error during parsing."""
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(f"Parse error at line {line}, col {col}: {message}")


class TypeError(CompilerError):
    """Type checking error."""
    def __init__(self, message: str, instruction: Optional[PTOInstruction] = None):
        self.instruction = instruction
        super().__init__(f"Type error: {message}")


class ValidationError(CompilerError):
    """Validation error."""
    pass


# =============================================================================
# Symbol Table
# =============================================================================

@dataclass
class Symbol:
    """A symbol in the symbol table."""
    name: str
    symbol_type: str  # "tile", "scalar", "memref", "index"
    data_type: Any    # TileType, ElementType, MemRefType, etc.
    is_const: bool = False
    value: Optional[Any] = None


class SymbolTable:
    """
    Symbol table for managing variables and their types.
    Supports nested scopes for loop constructs.
    """
    
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]
    
    def push_scope(self):
        """Enter a new scope."""
        self.scopes.append({})
    
    def pop_scope(self):
        """Exit current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()
    
    def define(self, name: str, symbol: Symbol):
        """Define a symbol in the current scope."""
        self.scopes[-1][name] = symbol
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol, searching from innermost to outermost scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def is_defined(self, name: str) -> bool:
        """Check if a symbol is defined in any scope."""
        return self.lookup(name) is not None
    
    def is_defined_in_current_scope(self, name: str) -> bool:
        """Check if a symbol is defined in the current scope."""
        return name in self.scopes[-1]


# =============================================================================
# Program Representation
# =============================================================================

@dataclass
class PTOProgram:
    """
    A complete PTO program.
    
    Attributes:
        is_in_core: If True (default), the function runs entirely within a single core.
        is_cube: If True, this InCore function requires cube unit (matmul)
        imports: List of imported function names from other source files
    """
    name: str = "main"
    tile_declarations: Dict[str, TileType] = field(default_factory=dict)
    scalar_declarations: Dict[str, ElementType] = field(default_factory=dict)
    memref_declarations: Dict[str, MemRefType] = field(default_factory=dict)
    instructions: List[PTOInstruction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_in_core: bool = True
    is_cube: bool = False
    imports: List[str] = field(default_factory=list)
    
    def add_tile(self, name: str, rows: int, cols: int, dtype: ElementType = ElementType.F32):
        """Declare a tile variable."""
        self.tile_declarations[name] = TileType.create(rows, cols, dtype)
    
    def add_scalar(self, name: str, dtype: ElementType = ElementType.F32):
        """Declare a scalar variable."""
        self.scalar_declarations[name] = dtype
    
    def add_memref(self, name: str, space: MemorySpace, dtype: ElementType,
                   shape: Optional[TileShape] = None):
        """Declare a memory reference."""
        self.memref_declarations[name] = MemRefType(space, dtype, shape)
    
    def add_instruction(self, instr: PTOInstruction):
        """Add an instruction to the program."""
        self.instructions.append(instr)
    
    def add_loop(self, loop: Union[TileLoop, NestedTileLoop]):
        """Add a loop construct by expanding it to instructions."""
        self.instructions.extend(loop.to_instructions())
    
    def add_import(self, func_name: str):
        """Add an import for a function from another source file."""
        if func_name not in self.imports:
            self.imports.append(func_name)


# =============================================================================
# Module Representation
# =============================================================================

@dataclass
class PTOModule:
    """
    A module containing multiple PTO functions.
    """
    name: str = "module"
    functions: Dict[str, 'PTOProgram'] = field(default_factory=dict)
    entry_function: Optional[str] = None
    imported_functions: Dict[str, 'PTOProgram'] = field(default_factory=dict)
    buffer_analysis: Dict[str, Dict] = field(default_factory=dict)
    
    def add_function(self, program: 'PTOProgram'):
        """Add a function to the module."""
        self.functions[program.name] = program
        if self.entry_function is None:
            self.entry_function = program.name
    
    def get_function(self, name: str) -> Optional['PTOProgram']:
        """Get a function by name."""
        if name in self.functions:
            return self.functions[name]
        return self.imported_functions.get(name)
    
    def has_function(self, name: str) -> bool:
        """Check if a function exists in the module."""
        return name in self.functions or name in self.imported_functions
    
    def set_entry(self, name: str):
        """Set the entry function."""
        if name not in self.functions:
            raise ValidationError(f"Function '{name}' not found in module")
        self.entry_function = name
    
    def import_function(self, program: 'PTOProgram'):
        """Import a function from an external source."""
        self.imported_functions[program.name] = program
    
    def get_all_functions(self) -> List['PTOProgram']:
        """Get all functions in the module."""
        return list(self.functions.values())
    
    def set_buffer_analysis(self, func_name: str, analysis: Dict):
        """Store buffer analysis results for a function."""
        self.buffer_analysis[func_name] = analysis
    
    def get_buffer_analysis(self, func_name: str) -> Optional[Dict]:
        """Get buffer analysis results for a function."""
        return self.buffer_analysis.get(func_name)
    
    def get_buffer_size(self, func_name: str) -> Tuple[float, float]:
        """Get buffer size for a function in KB."""
        analysis = self.buffer_analysis.get(func_name)
        if analysis:
            return (
                analysis.get('total_without_reuse_bytes', 0) / 1024.0,
                analysis.get('total_with_reuse_bytes', 0) / 1024.0
            )
        return (0.0, 0.0)
    
    def get_function_names(self) -> List[str]:
        """Get all function names."""
        return list(self.functions.keys())


# =============================================================================
# Loop Fusion Infrastructure (Platform Independent)
# =============================================================================

class OpCategory(Enum):
    """Operation categories for fusion optimization."""
    FUSABLE_BINARY = "fusable_binary"      # Element-wise binary ops
    FUSABLE_UNARY = "fusable_unary"        # Element-wise unary ops
    FUSABLE_SCALAR = "fusable_scalar"      # Tile-scalar ops
    REDUCTION = "reduction"                 # Row/col reductions
    BROADCAST = "broadcast"                 # Row/col broadcast ops
    MATMUL = "matmul"                       # Matrix multiplication
    MEMORY = "memory"                       # Load/store
    CONTROL_FLOW = "control_flow"          # Loops, conditionals
    SCALAR = "scalar"                       # Scalar operations
    OTHER = "other"


# Category mappings
FUSABLE_BINARY_OPS = {
    "TADD", "TSUB", "TMUL", "TDIV", "TMAX", "TMIN",
}

FUSABLE_UNARY_OPS = {
    "TABS", "TNEG", "TRECIP", "TEXP", "TLOG", "TSQRT", "TRSQRT",
    "TRELU", "TSIGMOID", "TTANH", "TGELU", "TSILU", "TFLOOR", "TCEIL",
    "TSIN", "TCOS", "TERF",
}

FUSABLE_SCALAR_OPS = {
    "TADDS", "TSUBS", "TMULS", "TDIVS", "TEXPANDS",
}

REDUCTION_OPS = {
    "TROWSUM", "TCOLSUM", "TROWMAX", "TCOLMAX", "TROWMIN", "TCOLMIN",
}

BROADCAST_OPS = {
    "TROWEXPANDSUB", "TROWEXPANDDIV", "TROWEXPANDMUL",
    "TCOLEXPANDSUB", "TCOLEXPANDDIV", "TCOLEXPANDMUL",
}

MATMUL_OPS = {"TMATMUL", "TMATMULACC"}

MEMORY_OPS = {"TLOAD", "TSTORE"}

CONTROL_FLOW_OPS = {"FOR", "ENDFOR", "WHILE", "ENDWHILE", "IF", "ELSE", "ENDIF", "IF_BIT"}

SCALAR_OPS = {"SLI", "SADD", "SSUB", "SMUL", "SDIV", "SMOV", "SCMP"}

FUNCTION_OPS = {"CALL", "RETURN"}


def get_category(opcode: str) -> OpCategory:
    """Get the category of an operation."""
    if opcode in FUSABLE_BINARY_OPS: return OpCategory.FUSABLE_BINARY
    if opcode in FUSABLE_UNARY_OPS: return OpCategory.FUSABLE_UNARY
    if opcode in FUSABLE_SCALAR_OPS: return OpCategory.FUSABLE_SCALAR
    if opcode in REDUCTION_OPS: return OpCategory.REDUCTION
    if opcode in BROADCAST_OPS: return OpCategory.BROADCAST
    if opcode in MATMUL_OPS: return OpCategory.MATMUL
    if opcode in MEMORY_OPS: return OpCategory.MEMORY
    if opcode in CONTROL_FLOW_OPS: return OpCategory.CONTROL_FLOW
    if opcode in SCALAR_OPS: return OpCategory.SCALAR
    return OpCategory.OTHER


def is_fusable(opcode: str) -> bool:
    """Check if an operation can be fused into a loop."""
    return opcode in FUSABLE_BINARY_OPS or opcode in FUSABLE_UNARY_OPS or opcode in FUSABLE_SCALAR_OPS


def is_fusion_barrier(opcode: str) -> bool:
    """Check if an operation is a fusion barrier."""
    return (opcode in REDUCTION_OPS or 
            opcode in BROADCAST_OPS or 
            opcode in MATMUL_OPS or
            opcode in MEMORY_OPS or
            opcode in CONTROL_FLOW_OPS or
            opcode in SCALAR_OPS or
            opcode in FUNCTION_OPS)


# =============================================================================
# Fusion Data Structures
# =============================================================================

@dataclass
class FusionTileShape:
    """Tile shape for fusion operations."""
    rows: int
    cols: int
    
    def __eq__(self, other):
        if isinstance(other, FusionTileShape):
            return self.rows == other.rows and self.cols == other.cols
        return False
    
    def __hash__(self):
        return hash((self.rows, self.cols))


@dataclass
class FusableOp:
    """A fusable operation."""
    opcode: str
    dst: str
    operands: List[str]
    tile_shape: FusionTileShape
    
    def get_scalar_operand_expr(self) -> str:
        """Get scalar operand expression for tile-scalar ops."""
        ops = {"TADDS": "+", "TSUBS": "-", "TMULS": "*", "TDIVS": "/"}
        return ops.get(self.opcode, "+")


@dataclass
class FusedLoop:
    """A fused loop containing multiple operations."""
    tile_shape: FusionTileShape
    operations: List[FusableOp]
    dtype: str = "f32"
    
    def add_op(self, op: FusableOp):
        """Add an operation to the fused loop."""
        self.operations.append(op)


@dataclass
class FusionBarrier:
    """A fusion barrier (non-fusable operation)."""
    opcode: str
    raw_instr: Any


# =============================================================================
# Mock Instruction for Code Generation
# =============================================================================

@dataclass
class MockTileInfo:
    """Tile info for code generation."""
    name: str
    rows: int
    cols: int
    dtype: str


@dataclass
class MockInstruction:
    """Mock instruction for fusion optimizer."""
    opcode: str
    dst: str
    operands: list
    raw_line: str = ""
    max_range: Optional[int] = None
    min_range: Optional[int] = None
    tile_levels: Optional[Dict[int, int]] = None


def _get_operand_str(operand) -> str:
    """Convert an operand to string representation."""
    if hasattr(operand, 'name'):
        return operand.name
    elif hasattr(operand, 'value'):
        return str(operand.value)
    return str(operand)


def convert_program_to_mock_instructions(program: PTOProgram) -> Tuple[Dict[str, MockTileInfo], List[MockInstruction]]:
    """Convert PTOProgram to mock instructions for fusion optimizer."""
    tile_info = {}
    for name, tile_type in program.tile_declarations.items():
        tile_info[name] = MockTileInfo(
            name=name,
            rows=tile_type.shape.rows,
            cols=tile_type.shape.cols,
            dtype=tile_type.element_type.value
        )
    
    mock_instructions = []
    for instr in program.instructions:
        opcode = getattr(instr, 'opcode', instr.__class__.__name__)
        dst = ""
        operands = []
        
        # Extract dst and operands based on instruction type
        if hasattr(instr, 'dst'):
            dst = _get_operand_str(instr.dst)
        
        if opcode == "TLOAD":
            dst = _get_operand_str(instr.dst)
            operands = [
                _get_operand_str(instr.src_mem),
                _get_operand_str(instr.row_offset),
                _get_operand_str(instr.col_offset)
            ]
        elif opcode == "TSTORE":
            dst = _get_operand_str(instr.dst_mem)
            operands = [
                _get_operand_str(instr.src),
                _get_operand_str(instr.row_offset),
                _get_operand_str(instr.col_offset)
            ]
        elif opcode == "FOR":
            dst = _get_operand_str(instr.iv)
            operands = [
                _get_operand_str(instr.lb),
                _get_operand_str(instr.ub),
                _get_operand_str(instr.step)
            ]
        elif opcode == "IF":
            dst = ""
            operands = [getattr(instr, 'condition', 'true')]
        elif opcode == "CALL":
            dst = instr.callee
            operands = instr.args if instr.args else []
        elif opcode == "RETURN":
            dst = ""
            operands = instr.values if instr.values else []
        # Handle scalar operations (TMULS, TADDS, TSUBS, TDIVS)
        elif opcode in ("TMULS", "TADDS", "TSUBS", "TDIVS"):
            # Scalar operations have 'src' and 'scalar' operands
            if hasattr(instr, 'src') and hasattr(instr, 'scalar'):
                operands = [_get_operand_str(instr.src), _get_operand_str(instr.scalar)]
            else:
                operands = []
        elif hasattr(instr, 'src0') and hasattr(instr, 'src1'):
            operands = [_get_operand_str(instr.src0), _get_operand_str(instr.src1)]
        elif hasattr(instr, 'src'):
            operands = [_get_operand_str(instr.src)]
        elif hasattr(instr, 'a') and hasattr(instr, 'b'):
            operands = [_get_operand_str(instr.a), _get_operand_str(instr.b)]
        elif hasattr(instr, 'operands'):
            operands = [_get_operand_str(op) for op in instr.operands]
        
        # Handle FOR loop with max_range
        mock = MockInstruction(opcode=opcode, dst=dst, operands=operands)
        if opcode == "FOR":
            if hasattr(instr, 'max_range'):
                mock.max_range = instr.max_range
            if hasattr(instr, 'min_range'):
                mock.min_range = instr.min_range
            if hasattr(instr, 'tile_levels'):
                mock.tile_levels = instr.tile_levels
        
        mock_instructions.append(mock)
    
    return tile_info, mock_instructions


# =============================================================================
# Tile Buffer Analyzer
# =============================================================================

@dataclass
class TileBufferInfo:
    """Information about a tile buffer."""
    name: str
    rows: int
    cols: int
    dtype: str
    element_size: int
    total_bytes: int
    first_write: int
    last_read: int
    can_reuse_from: Optional[str] = None


class TileBufferAnalyzer:
    """
    Analyzes tile buffer usage in InCore functions.
    """
    
    ELEMENT_SIZES = {
        'f32': 4, 'f16': 2, 'bf16': 2, 'i32': 4, 'i16': 2, 'i8': 1, 'u8': 1
    }
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.tile_info: Dict[str, TileBufferInfo] = {}
        self.instructions = []
        self.analysis_result = {}
    
    def analyze(self) -> Dict:
        """Run complete buffer analysis."""
        self._collect_tiles()
        self._analyze_liveness()
        self._find_reuse_opportunities()
        self._compute_totals()
        return self.analysis_result
    
    def _collect_tiles(self):
        """Collect tile information from declarations."""
        for name, tile_type in self.program.tile_declarations.items():
            dtype = tile_type.element_type.value
            elem_size = self.ELEMENT_SIZES.get(dtype, 4)
            rows = tile_type.shape.rows
            cols = tile_type.shape.cols
            
            self.tile_info[name] = TileBufferInfo(
                name=name,
                rows=rows,
                cols=cols,
                dtype=dtype,
                element_size=elem_size,
                total_bytes=rows * cols * elem_size,
                first_write=-1,
                last_read=-1
            )
    
    def _analyze_liveness(self):
        """Analyze when each tile is written and last read."""
        for idx, instr in enumerate(self.program.instructions):
            opcode = getattr(instr, 'opcode', '')
            
            # Track writes (destination)
            dst_name = None
            if hasattr(instr, 'dst') and hasattr(instr.dst, 'name'):
                dst_name = instr.dst.name
            
            if dst_name and dst_name in self.tile_info:
                if self.tile_info[dst_name].first_write < 0:
                    self.tile_info[dst_name].first_write = idx
            
            # Track reads (sources)
            read_names = []
            for attr in ['src', 'src0', 'src1', 'a', 'b', 'acc']:
                if hasattr(instr, attr):
                    src = getattr(instr, attr)
                    if hasattr(src, 'name'):
                        read_names.append(src.name)
            
            for name in read_names:
                if name in self.tile_info:
                    self.tile_info[name].last_read = idx
    
    def _find_reuse_opportunities(self):
        """Find tiles that can reuse buffers from dead tiles."""
        sorted_tiles = sorted(
            self.tile_info.values(),
            key=lambda t: t.first_write if t.first_write >= 0 else 9999
        )
        
        for tile in sorted_tiles:
            if tile.first_write < 0:
                continue
            
            for candidate in self.tile_info.values():
                if candidate.name == tile.name:
                    continue
                if candidate.can_reuse_from is not None:
                    continue
                if candidate.last_read < 0:
                    continue
                if candidate.last_read < tile.first_write:
                    if candidate.total_bytes >= tile.total_bytes:
                        tile.can_reuse_from = candidate.name
                        break
    
    def _compute_totals(self):
        """Compute total buffer sizes."""
        total_without_reuse = sum(t.total_bytes for t in self.tile_info.values())
        
        reuse_map = {}
        for tile in self.tile_info.values():
            if tile.can_reuse_from:
                reuse_map[tile.name] = tile.can_reuse_from
        
        total_with_reuse = 0
        for tile in self.tile_info.values():
            if tile.name not in reuse_map:
                total_with_reuse += tile.total_bytes
        
        self.analysis_result = {
            'total_tiles': len(self.tile_info),
            'total_without_reuse_bytes': total_without_reuse,
            'total_with_reuse_bytes': total_with_reuse,
            'reuse_savings_bytes': total_without_reuse - total_with_reuse,
            'reuse_savings_percent': (1 - total_with_reuse / total_without_reuse) * 100 if total_without_reuse > 0 else 0,
            'tiles': self.tile_info,
            'reuse_map': reuse_map,
        }
    
    def generate_report(self) -> str:
        """Generate a human-readable analysis report."""
        if not self.analysis_result:
            self.analyze()
        
        r = self.analysis_result
        lines = []
        
        lines.append("// " + "=" * 70)
        lines.append(f"// TILE BUFFER ANALYSIS: {self.program.name}")
        lines.append("// " + "=" * 70)
        lines.append("//")
        lines.append("// SUMMARY:")
        lines.append(f"//   Total tiles declared:     {r['total_tiles']}")
        lines.append(f"//   Total capacity (no reuse): {r['total_without_reuse_bytes']:,} bytes ({r['total_without_reuse_bytes']/1024:.1f} KB)")
        lines.append(f"//   Total capacity (w/ reuse): {r['total_with_reuse_bytes']:,} bytes ({r['total_with_reuse_bytes']/1024:.1f} KB)")
        lines.append(f"//   Reuse savings:            {r['reuse_savings_bytes']:,} bytes ({r['reuse_savings_percent']:.1f}%)")
        lines.append("//")
        lines.append("// " + "=" * 70)
        lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# Loop Fusion Optimizer
# =============================================================================

class LoopFusionOptimizer:
    """
    Fuses consecutive fusable operations into single loops.
    """
    
    def __init__(self, tile_info: Dict[str, MockTileInfo]):
        self.tile_info = tile_info
        self.stats = {
            'fusable_ops': 0,
            'fused_loops': 0,
            'barriers': 0,
            'fusion_savings': 0,
        }
    
    def optimize(self, instructions: List[MockInstruction]) -> List[Union[FusedLoop, FusionBarrier]]:
        """Optimize instructions by fusing consecutive fusable operations."""
        result = []
        current_fusable = []
        
        for instr in instructions:
            if is_fusable(instr.opcode):
                self.stats['fusable_ops'] += 1
                
                dst_info = self.tile_info.get(instr.dst)
                if dst_info:
                    shape = FusionTileShape(dst_info.rows, dst_info.cols)
                    dtype = dst_info.dtype
                else:
                    shape = FusionTileShape(8, 8)
                    dtype = "f32"
                
                fusable_op = FusableOp(
                    opcode=instr.opcode,
                    dst=instr.dst,
                    operands=instr.operands,
                    tile_shape=shape
                )
                
                # Check if can fuse with current group
                if current_fusable and current_fusable[0].tile_shape == shape:
                    current_fusable.append(fusable_op)
                else:
                    # Flush current group and start new one
                    if current_fusable:
                        result.append(self._create_fused_loop(current_fusable, dtype))
                    current_fusable = [fusable_op]
            else:
                # Barrier - flush current fusable group
                if current_fusable:
                    result.append(self._create_fused_loop(current_fusable, "f32"))
                    current_fusable = []
                
                self.stats['barriers'] += 1
                result.append(FusionBarrier(opcode=instr.opcode, raw_instr=instr))
        
        # Flush remaining fusable ops
        if current_fusable:
            result.append(self._create_fused_loop(current_fusable, "f32"))
        
        return result
    
    def _create_fused_loop(self, ops: List[FusableOp], dtype: str) -> FusedLoop:
        """Create a fused loop from a list of fusable operations."""
        self.stats['fused_loops'] += 1
        if len(ops) > 1:
            self.stats['fusion_savings'] += len(ops) - 1
        
        return FusedLoop(
            tile_shape=ops[0].tile_shape,
            operations=ops,
            dtype=dtype
        )


# =============================================================================
# Type Checker
# =============================================================================

class TypeChecker:
    """Type checker for PTO programs."""
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.errors: List[str] = []
    
    def check(self) -> bool:
        """Run type checking, return True if no errors."""
        self.errors = []
        
        for instr in self.program.instructions:
            self._check_instruction(instr)
        
        return len(self.errors) == 0
    
    def _check_instruction(self, instr: PTOInstruction):
        """Type check a single instruction."""
        opcode = getattr(instr, "opcode", instr.__class__.__name__)
        if opcode in ("TADD", "TSUB", "TMUL"):
            self._check_binary_tile_op(instr)
        elif opcode == "TMATMUL":
            self._check_matmul(instr)
    
    def _check_binary_tile_op(self, instr):
        """Check binary tile operation types match."""
        if hasattr(instr, 'src0') and hasattr(instr, 'src1'):
            src0_type = instr.src0.tile_type
            src1_type = instr.src1.tile_type
            
            if src0_type.shape != src1_type.shape:
                self.errors.append(
                    f"{instr.opcode}: Shape mismatch - {src0_type.shape} vs {src1_type.shape}"
                )
    
    def _check_matmul(self, instr):
        """Check matrix multiplication dimensions."""
        if hasattr(instr, 'a') and hasattr(instr, 'b'):
            a_type = instr.a.tile_type
            b_type = instr.b.tile_type
            
            if a_type.shape.cols != b_type.shape.rows:
                self.errors.append(
                    f"TMATMUL: Incompatible dimensions - "
                    f"A cols ({a_type.shape.cols}) != B rows ({b_type.shape.rows})"
                )


# =============================================================================
# Base Code Generator
# =============================================================================

class CodeGenerator:
    """Base code generator class."""
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.output_lines: List[str] = []
    
    def generate(self) -> str:
        """Generate code (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement generate()")
    
    def _emit(self, line: str):
        """Emit a line of code."""
        self.output_lines.append(line)


# =============================================================================
# Optimizer
# =============================================================================

class Optimizer:
    """
    Optimizer for PTO programs.
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
    
    def optimize(self) -> PTOProgram:
        """Run all optimization passes."""
        self._dead_code_elimination()
        return self.program
    
    def _dead_code_elimination(self):
        """Remove dead code."""
        # Simple implementation - remove unused tiles
        used_tiles = set()
        for instr in self.program.instructions:
            for attr in ['src', 'src0', 'src1', 'a', 'b', 'acc']:
                if hasattr(instr, attr):
                    src = getattr(instr, attr)
                    if hasattr(src, 'name'):
                        used_tiles.add(src.name)


# =============================================================================
# PTO Compiler (Core)
# =============================================================================

class PTOCompiler:
    """
    Main compiler class that orchestrates the compilation pipeline.
    """
    
    def __init__(self, enable_optimization: bool = True, enable_type_check: bool = True):
        self.enable_optimization = enable_optimization
        self.enable_type_check = enable_type_check
    
    def compile(self, program: PTOProgram) -> PTOProgram:
        """Compile a PTO program."""
        # Type checking
        if self.enable_type_check:
            checker = TypeChecker(program)
            if not checker.check():
                raise TypeError(f"Type errors: {checker.errors}")
        
        # Optimization
        if self.enable_optimization:
            optimizer = Optimizer(program)
            program = optimizer.optimize()
        
        return program


# =============================================================================
# Binary Expansion Utility
# =============================================================================

def apply_binary_expansion(code: str) -> str:
    """
    Transform FOR loops with @BINARY_EXPAND markers into cascaded if-statements.
    """
    lines = code.split('\n')
    result = []
    i = 0
    loop_counter = 0  # Unique ID for each expanded loop
    
    while i < len(lines):
        line = lines[i]
        
        # Check for binary expansion marker
        marker_match = re.match(r'\s*// @BINARY_EXPAND: max_range=(\d+)(?:, min_range=(\d+))?(?:, bits=\[([\d,]+)\])?(?:\s+tile_levels=\{([^}]+)\})?', line)
        if marker_match:
            max_range = int(marker_match.group(1))
            min_range = int(marker_match.group(2)) if marker_match.group(2) else 1
            
            # Parse tile_levels if present
            tile_levels = {}
            if marker_match.group(4):
                for pair in marker_match.group(4).split(','):
                    k, v = pair.split(':')
                    tile_levels[int(k)] = int(v)
            
            # Next line should be the FOR loop
            i += 1
            if i >= len(lines):
                result.append(line)
                continue
            
            for_line = lines[i]
            for_match = re.match(r'(\s*)for \(int (\w+) = (\w+); \2 < (\w+); \2 \+= (\d+)\) \{', for_line)
            
            if not for_match:
                result.append(line)
                result.append(for_line)
                i += 1
                continue
            
            indent = for_match.group(1)
            iv = for_match.group(2)
            lb = for_match.group(3)
            ub = for_match.group(4)
            step = int(for_match.group(5))
            
            # Collect loop body
            body_lines = []
            brace_count = 1
            i += 1
            while i < len(lines) and brace_count > 0:
                body_line = lines[i]
                brace_count += body_line.count('{') - body_line.count('}')
                if brace_count > 0:
                    body_lines.append(body_line)
                i += 1
            
            # Generate binary-expanded code with unique variable names
            # Use loop_counter suffix to avoid redefinition when same iterator name used in multiple loops
            rem_var = f"{iv}_remaining_{loop_counter}"
            base_var = f"{iv}_base_{loop_counter}"
            loop_counter += 1
            
            result.append(f"{indent}// Binary-expanded loop: {iv} in [{lb}, {ub}), max_range={max_range}")
            result.append(f"{indent}int {rem_var} = {ub} - {lb};")
            result.append(f"{indent}int {base_var} = {lb};")
            
            # Generate cascading if-blocks for each power of 2
            p = max_range
            while p >= min_range:
                result.append(f"{indent}if ({rem_var} >= {p}) {{")
                result.append(f"{indent}    for (int {iv} = {base_var}; {iv} < {base_var} + {p}; {iv} += {step}) {{")
                for body_line in body_lines:
                    result.append(f"    {body_line}")
                result.append(f"{indent}    }}")
                result.append(f"{indent}    {base_var} += {p};")
                result.append(f"{indent}    {rem_var} -= {p};")
                result.append(f"{indent}}}")
                p //= 2
            
            # Handle residual iterations (remaining < min_range)
            result.append(f"{indent}// Residual loop for remaining < {min_range}")
            result.append(f"{indent}for (int {iv} = {base_var}; {iv} < {base_var} + {rem_var}; {iv} += {step}) {{")
            for body_line in body_lines:
                result.append(f"    {body_line}")
            result.append(f"{indent}}}")
            
            continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


# =============================================================================
# Loop Replay Optimization Utilities
# =============================================================================

def get_loop_replay_header() -> str:
    """Get the loop replay header code."""
    return '''
// Loop Replay Context - for record/replay optimization
// See pto_runtime.h for LoopReplayCtx structure definition
'''


def apply_loop_replay_optimization(code: str, enable_replay: bool = True) -> str:
    """
    Transform orchestration code to use loop record/replay optimization.
    """
    if not enable_replay:
        return code
    
    # Look for patterns like:
    # for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
    #     ... task scheduling code ...
    # }
    
    lines = code.split('\n')
    result = []
    
    for_pattern = re.compile(r'(\s*)for \(int (\w+) = (\d+); \2 < (\w+); \2\+\+\) \{')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        match = for_pattern.match(line)
        
        if match and 'pto_task_alloc' in '\n'.join(lines[i:i+20]):
            indent = match.group(1)
            iv = match.group(2)
            lb = match.group(3)
            ub = match.group(4)
            
            # Add loop replay context initialization before the loop
            result.append(f"{indent}// Loop replay optimization")
            result.append(f"{indent}LoopReplayCtx _loop_ctx_{iv};")
            result.append(f"{indent}pto_loop_init(&_loop_ctx_{iv}, \"{iv}_loop\", 1, OFFSET_ROW);")
            result.append(f"{indent}")
            result.append(f"{indent}for (int {iv} = {lb}; {iv} < {ub}; {iv}++) {{")
            result.append(f"{indent}    if (pto_loop_should_record(rt, &_loop_ctx_{iv}, {iv})) {{")
            
            # Find the end of the loop and wrap the body
            brace_count = 1
            i += 1
            body_lines = []
            while i < len(lines) and brace_count > 0:
                body_line = lines[i]
                brace_count += body_line.count('{') - body_line.count('}')
                if brace_count > 0:
                    body_lines.append(f"    {body_line}")
                i += 1
            
            result.extend(body_lines)
            result.append(f"{indent}        pto_loop_finish_record(rt, &_loop_ctx_{iv});")
            result.append(f"{indent}    }} else {{")
            result.append(f"{indent}        pto_loop_replay(rt, &_loop_ctx_{iv}, {iv});")
            result.append(f"{indent}    }}")
            result.append(f"{indent}}}")
            result.append(f"{indent}pto_loop_cleanup(&_loop_ctx_{iv});")
            continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


# =============================================================================
# Export commonly used items
# =============================================================================

__all__ = [
    # Error types
    'CompilerError', 'ParseError', 'TypeError', 'ValidationError',
    
    # Symbol table
    'Symbol', 'SymbolTable',
    
    # Program representation
    'PTOProgram', 'PTOModule',
    
    # Loop fusion
    'OpCategory', 'FusionTileShape', 'FusableOp', 'FusedLoop', 'FusionBarrier',
    'LoopFusionOptimizer', 'get_category', 'is_fusable', 'is_fusion_barrier',
    
    # Mock instructions
    'MockTileInfo', 'MockInstruction', 'convert_program_to_mock_instructions',
    
    # Buffer analysis
    'TileBufferInfo', 'TileBufferAnalyzer',
    
    # Type checker and optimizer
    'TypeChecker', 'Optimizer', 'CodeGenerator', 'PTOCompiler',
    
    # Utilities
    'apply_binary_expansion', 'apply_loop_replay_optimization', 'get_loop_replay_header',
]
