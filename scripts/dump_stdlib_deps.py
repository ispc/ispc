#!/usr/bin/env python3
#
#  Copyright (c) 2025, Intel Corporation
#
#  SPDX-License-Identifier: BSD-3-Clause

"""
ISPC stdlib/builtins Dependencies Dumper

Dumps function call dependencies in ISPC stdlib and builtins modules.
Follows the target hierarchy to build call graphs across different bitcode modules.
It supposes that the ISPC bitcode files are produced as part of the ISPC build
process, i.e., ISPC is build with ISPC_SLIM_BINARY=ON

Usage: python dump_stdlib_deps.py <function_name> <target> [options]
Example: python dump_stdlib_deps.py shift___vyiuni avx512spr-x4 --hide-pseudo
"""

import sys
# Subprocess is used with default shell which is False, it's safe and doesn't allow shell injection
# so it's safe to ignore Bandit warning.
import subprocess #nosec
import os
import re
import argparse
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging


class FunctionClassification(Enum):
    """Classification types for missing functions"""
    PSEUDO = "PSEUDO"
    LLVM_INTRINSIC = "LLVM_INTRINSIC"
    MANGLED_CPP = "MANGLED_CPP"
    NOT_FOUND = "NOT_FOUND"
    RECURSIVE = "RECURSIVE"


class Constants:
    """Configuration constants for the dependency analyzer"""

    DEFAULT_BITCODE_DIR = "./share/ispc"
    DEFAULT_MAX_DEPTH = 10
    DEFAULT_ARCH = "64bit"
    MAX_SUGGESTIONS = 10

    # LLVM symbol types to include
    LLVM_SYMBOL_TYPES = {'T', 't', 'W', 'w'}

    # ISPC type mappings for mangling
    TYPE_MAPPING = {
        'vyi': 'varying int32',
        'vyf': 'varying float',
        'vyd': 'varying double',
        'vyh': 'varying float16',
        'vys': 'varying int16',
        'vyt': 'varying int8',
        'vyu': 'varying uint32',
        'vyS': 'varying uint16',
        'vyT': 'varying uint8',
        'vyI': 'varying int64',
        'vyU': 'varying uint64',
        'vyb': 'varying bool',
        'uni': 'uniform int32',
        'unf': 'uniform float',
        'und': 'uniform double',
        'unh': 'uniform float16',
        'uns': 'uniform int16',
        'unt': 'uniform int8',
        'unu': 'uniform uint32',
        'unS': 'uniform uint16',
        'unT': 'uniform uint8',
        'unI': 'uniform int64',
        'unU': 'uniform uint64',
        'unb': 'uniform bool',
    }

    # Known compiler pseudo-functions that get optimized away
    PSEUDO_FUNCTIONS = {
        '__pseudo_masked_store_i8', '__pseudo_masked_store_i16', '__pseudo_masked_store_i32',
        '__pseudo_masked_store_i64', '__pseudo_masked_store_float', '__pseudo_masked_store_double',
        '__pseudo_masked_store_half',
        '__pseudo_gather32_i8', '__pseudo_gather32_i16', '__pseudo_gather32_i32',
        '__pseudo_gather32_i64', '__pseudo_gather32_float', '__pseudo_gather32_double',
        '__pseudo_gather32_half', '__pseudo_gather64_i8', '__pseudo_gather64_i16',
        '__pseudo_gather64_i32', '__pseudo_gather64_i64', '__pseudo_gather64_float',
        '__pseudo_gather64_double', '__pseudo_gather64_half',
        '__pseudo_scatter32_i8', '__pseudo_scatter32_i16', '__pseudo_scatter32_i32',
        '__pseudo_scatter32_i64', '__pseudo_scatter32_float', '__pseudo_scatter32_double',
        '__pseudo_scatter32_half', '__pseudo_scatter64_i8', '__pseudo_scatter64_i16',
        '__pseudo_scatter64_i32', '__pseudo_scatter64_i64', '__pseudo_scatter64_float',
        '__pseudo_scatter64_double', '__pseudo_scatter64_half',
        '__pseudo_gather_factored_base_offsets32_i8', '__pseudo_gather_factored_base_offsets32_i16',
        '__pseudo_gather_factored_base_offsets32_i32', '__pseudo_gather_factored_base_offsets32_i64',
        '__pseudo_gather_factored_base_offsets32_float', '__pseudo_gather_factored_base_offsets32_double',
        '__pseudo_gather_factored_base_offsets32_half', '__pseudo_gather_factored_base_offsets64_i8',
        '__pseudo_gather_factored_base_offsets64_i16', '__pseudo_gather_factored_base_offsets64_i32',
        '__pseudo_gather_factored_base_offsets64_i64', '__pseudo_gather_factored_base_offsets64_float',
        '__pseudo_gather_factored_base_offsets64_double', '__pseudo_gather_factored_base_offsets64_half',
        '__pseudo_gather_base_offsets32_i8', '__pseudo_gather_base_offsets32_i16',
        '__pseudo_gather_base_offsets32_i32', '__pseudo_gather_base_offsets32_i64',
        '__pseudo_gather_base_offsets32_float', '__pseudo_gather_base_offsets32_double',
        '__pseudo_gather_base_offsets32_half', '__pseudo_gather_base_offsets64_i8',
        '__pseudo_gather_base_offsets64_i16', '__pseudo_gather_base_offsets64_i32',
        '__pseudo_gather_base_offsets64_i64', '__pseudo_gather_base_offsets64_float',
        '__pseudo_gather_base_offsets64_double', '__pseudo_gather_base_offsets64_half',
        '__pseudo_scatter_factored_base_offsets32_i8', '__pseudo_scatter_factored_base_offsets32_i16',
        '__pseudo_scatter_factored_base_offsets32_i32', '__pseudo_scatter_factored_base_offsets32_i64',
        '__pseudo_scatter_factored_base_offsets32_float', '__pseudo_scatter_factored_base_offsets32_double',
        '__pseudo_scatter_factored_base_offsets32_half', '__pseudo_scatter_factored_base_offsets64_i8',
        '__pseudo_scatter_factored_base_offsets64_i16', '__pseudo_scatter_factored_base_offsets64_i32',
        '__pseudo_scatter_factored_base_offsets64_i64', '__pseudo_scatter_factored_base_offsets64_float',
        '__pseudo_scatter_factored_base_offsets64_double', '__pseudo_scatter_factored_base_offsets64_half',
        '__pseudo_scatter_base_offsets32_i8', '__pseudo_scatter_base_offsets32_i16',
        '__pseudo_scatter_base_offsets32_i32', '__pseudo_scatter_base_offsets32_i64',
        '__pseudo_scatter_base_offsets32_float', '__pseudo_scatter_base_offsets32_double',
        '__pseudo_scatter_base_offsets32_half', '__pseudo_scatter_base_offsets64_i8',
        '__pseudo_scatter_base_offsets64_i16', '__pseudo_scatter_base_offsets64_i32',
        '__pseudo_scatter_base_offsets64_i64', '__pseudo_scatter_base_offsets64_float',
        '__pseudo_scatter_base_offsets64_double', '__pseudo_scatter_base_offsets64_half',
        '__pseudo_prefetch_read_varying_1', '__pseudo_prefetch_read_varying_2',
        '__pseudo_prefetch_read_varying_3', '__pseudo_prefetch_read_varying_nt',
        '__pseudo_prefetch_write_varying_1', '__pseudo_prefetch_write_varying_2',
        '__pseudo_prefetch_write_varying_3', '__pseudo_prefetch_read_varying_1_native',
        '__pseudo_prefetch_read_varying_2_native', '__pseudo_prefetch_read_varying_3_native',
        '__pseudo_prefetch_read_varying_nt_native', '__pseudo_prefetch_write_varying_1_native',
        '__pseudo_prefetch_write_varying_2_native', '__pseudo_prefetch_write_varying_3_native'
    }


@dataclass
class FunctionLocation:
    """Location of a function in the target hierarchy"""
    target: str
    module_type: str


@dataclass
class CallGraphNode:
    """Node in the call graph"""
    function: str
    target: str
    module: str = "unknown"
    calls: List['CallGraphNode'] = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []


class ISPCTarget:
    """Represents an ISPC target with hierarchy information"""

    # Target hierarchy mapping based on the C++ map from src/builtins.cpp
    TARGET_HIERARCHY = {
        # NEON targets
        'neon-i8x16': 'generic-i8x16',
        'neon-i8x32': 'generic-i8x32',
        'neon-i16x8': 'generic-i16x8',
        'neon-i16x16': 'generic-i16x16',
        'neon-i32x4': 'generic-i32x4',
        'neon-i32x8': 'generic-i32x8',

        # AVX10.2 targets
        'avx10.2dmr-x4': 'avx512gnr-x4',
        'avx10.2dmr-x8': 'avx512gnr-x8',
        'avx10.2dmr-x16': 'avx512gnr-x16',
        'avx10.2dmr-x32': 'avx512gnr-x32',
        'avx10.2dmr-x64': 'avx512gnr-x64',

        # AVX512 hierarchy
        'avx512gnr-x4': 'avx512spr-x4',
        'avx512spr-x4': 'avx512icl-x4',
        'avx512icl-x4': 'avx512skx-x4',
        'avx512skx-x4': 'generic-i1x4',

        'avx512gnr-x8': 'avx512spr-x8',
        'avx512spr-x8': 'avx512icl-x8',
        'avx512icl-x8': 'avx512skx-x8',
        'avx512skx-x8': 'generic-i1x8',

        'avx512gnr-x16': 'avx512spr-x16',
        'avx512spr-x16': 'avx512icl-x16',
        'avx512icl-x16': 'avx512skx-x16',
        'avx512skx-x16': 'generic-i1x16',

        'avx512gnr-x32': 'avx512spr-x32',
        'avx512spr-x32': 'avx512icl-x32',
        'avx512icl-x32': 'avx512skx-x32',
        'avx512skx-x32': 'generic-i1x32',

        'avx512gnr-x64': 'avx512spr-x64',
        'avx512spr-x64': 'avx512icl-x64',
        'avx512icl-x64': 'avx512skx-x64',
        'avx512skx-x64': 'generic-i1x64',

        # other x86 hierarchy
        'avx2vnni-i32x4': 'avx2-i32x4',
        'avx2-i32x4': 'avx1-i32x4',
        'avx1-i32x4': 'sse4-i32x4',
        'sse4-i32x4': 'sse41-i32x4',
        'sse41-i32x4': 'sse2-i32x4',
        'sse2-i32x4': 'generic-i32x4',

        'avx2vnni-i32x8': 'avx2-i32x8',
        'avx2-i32x8': 'avx1-i32x8',
        'avx1-i32x8': 'sse4-i32x8',
        'sse4-i32x8': 'sse41-i32x8',
        'sse41-i32x8': 'sse2-i32x8',
        'sse2-i32x8': 'generic-i32x8',

        'avx2vnni-i32x16': 'avx2-i32x16',
        'avx2-i32x16': 'avx1-i32x16',
        'avx1-i32x16': 'generic-i32x16',

        'avx2-i64x4': 'avx1-i64x4',
        'avx1-i64x4': 'generic-i64x4',

        'avx2-i16x16': 'generic-i16x16',
        'avx2-i8x32': 'generic-i8x32',

        'sse4-i8x16': 'sse41-i8x16',
        'sse41-i8x16': 'generic-i8x16',

        'sse4-i16x8': 'sse41-i16x8',
        'sse41-i16x8': 'generic-i16x8',

        # WebAssembly
        'wasm-i32x4': 'generic-i32x4'
    }

    @classmethod
    def get_parent(cls, target: str) -> Optional[str]:
        """Get parent target in hierarchy"""
        return cls.TARGET_HIERARCHY.get(target)

    @classmethod
    def normalize_target_name(cls, target: str) -> str:
        """Normalize target name format (e.g., avx512spr-x4 -> avx512spr_x4, avx10.2dmr-x4 -> avx10_2dmr_x4)"""
        return target.replace('-', '_').replace('.', '_')

    @classmethod
    def denormalize_target_name(cls, target: str) -> str:
        """Convert back to hierarchy format (e.g., avx512spr_x4 -> avx512spr-x4)"""
        return target.replace('_', '-')

    @classmethod
    def get_root_targets(cls) -> List[str]:
        """Get all root targets (those that don't appear as values in the hierarchy)"""
        # Get all targets that appear as children in the hierarchy
        child_targets = set(cls.TARGET_HIERARCHY.keys())
        parent_targets = set(cls.TARGET_HIERARCHY.values())

        # Root targets are those that appear as children but never as parents
        # Plus any targets that are parent targets but not children
        root_targets = list(child_targets - parent_targets)

        # Also include standalone parent targets that don't have children
        for parent in parent_targets:
            if parent not in child_targets:
                root_targets.append(parent)

        return sorted(root_targets)

    @classmethod
    def get_non_generic_root_targets(cls) -> List[str]:
        """Get all root targets excluding generic targets"""
        root_targets = cls.get_root_targets()
        return [target for target in root_targets if not target.startswith('generic-')]

    @classmethod
    def get_targets_by_width(cls, width: str) -> List[str]:
        """Get all root targets (excluding generic) with specific width (e.g., 'x4', 'x8')"""
        root_targets = cls.get_non_generic_root_targets()
        width_targets = []

        for target in root_targets:
            # Check for targets ending with -x4, -x8, etc.
            if target.endswith(f'-{width}'):
                width_targets.append(target)
            # Check for targets with embedded width like avx2-i32x4, neon-i32x4
            elif f'{width}' in target and (target.endswith(width) or f'{width}' in target.split('-')[-1]):
                width_targets.append(target)

        return sorted(width_targets)


class LLVMBitcodeParser:
    """Parser for LLVM bitcode files to extract function information"""

    def __init__(self, bitcode_dir: str):
        self.bitcode_dir = bitcode_dir
        self._function_cache = {}  # Cache for extracted functions
        self._call_cache = {}      # Cache for function calls

    def _run_llvm_tool(self, tool: str, args: List[str], error_msg: str) -> Optional[str]:
        """Common method to run LLVM tools with error handling"""
        try:
            result = subprocess.run([tool] + args, capture_output=True, text=True, check=True, shell=False)
            return result.stdout
        except subprocess.CalledProcessError:
            print(f"Warning: {tool} failed - {error_msg}")
            return None
        except FileNotFoundError:
            print(f"Warning: {tool} not found - {error_msg}")
            return None

    def _is_ispc_function(self, func_name: str) -> bool:
        """Check if a function name appears to be an ISPC function"""
        return (not func_name.startswith('_Z') and
                ('__' in func_name or
                 func_name.startswith('shift') or
                 func_name.startswith('shuffle')))

    def get_module_path(self, module_type: str, target: str, arch: str = Constants.DEFAULT_ARCH) -> str:
        """Get path to LLVM bitcode module"""
        target_norm = ISPCTarget.normalize_target_name(target)

        # Generic targets have architecture-specific suffixes
        if target_norm.startswith('generic_'):
            # For generic targets, use x86_64 architecture suffix
            filename = f"{module_type}_{target_norm}_x86_64_unix.bc"
        else:
            filename = f"{module_type}_{target_norm}_{arch}_unix.bc"

        return os.path.join(self.bitcode_dir, filename)

    def extract_functions(self, module_path: str) -> Set[str]:
        """Extract function names from LLVM bitcode module"""
        if not os.path.exists(module_path):
            return set()

        # Check cache first
        if module_path in self._function_cache:
            return self._function_cache[module_path]

        output = self._run_llvm_tool('llvm-nm', [module_path], f"extracting symbols from {module_path}")
        if output is None:
            return self._extract_functions_with_dis(module_path)

        functions = set()
        for line in output.split('\n'):
            if line.strip():
                # Parse llvm-nm output: address type name
                parts = line.strip().split()
                if len(parts) >= 3 and parts[1] in Constants.LLVM_SYMBOL_TYPES:
                    func_name = parts[2]
                    if self._is_ispc_function(func_name):
                        functions.add(func_name)

        # Cache the result
        self._function_cache[module_path] = functions
        return functions

    def _extract_functions_with_dis(self, module_path: str) -> Set[str]:
        """Fallback method using llvm-dis to disassemble and parse"""
        output = self._run_llvm_tool('llvm-dis', ['-o', '-', module_path], f"disassembling {module_path}")
        if output is None:
            return set()

        functions = set()
        # Look for function definitions in LLVM IR
        for line in output.split('\n'):
            match = re.match(r'define.*@([a-zA-Z_][a-zA-Z0-9_]*)\(', line)
            if match:
                func_name = match.group(1)
                if self._is_ispc_function(func_name):
                    functions.add(func_name)

        return functions

    def get_function_calls(self, module_path: str, function_name: str) -> Set[str]:
        """Get functions called by a specific function"""
        if not os.path.exists(module_path):
            return set()

        # Check cache first
        cache_key = (module_path, function_name)
        if cache_key in self._call_cache:
            return self._call_cache[cache_key]

        output = self._run_llvm_tool('llvm-dis', ['-o', '-', module_path], f"disassembling {module_path} for function calls")
        if output is None:
            return set()

        calls = set()

        # Find the function definition using regex
        function_pattern = rf'define.*@{re.escape(function_name)}\('
        in_function = False
        brace_count = 0

        for line in output.split('\n'):
            if re.search(function_pattern, line):
                in_function = True
                brace_count = line.count('{') - line.count('}')
            elif in_function:
                brace_count += line.count('{') - line.count('}')

                # Look for function calls - improved regex
                call_matches = re.findall(r'call[^@]*@([a-zA-Z_][a-zA-Z0-9_]*)\(', line)
                for call_match in call_matches:
                    if self._is_ispc_function(call_match):
                        calls.add(call_match)

                if brace_count <= 0 and in_function:
                    break

        # Cache the result
        self._call_cache[cache_key] = calls
        return calls


class FunctionFinder:
    """Finds functions in the ISPC target hierarchy"""

    def __init__(self, parser: LLVMBitcodeParser):
        self.parser = parser
        self.pseudo_functions = Constants.PSEUDO_FUNCTIONS

    def find_function_in_hierarchy(self, function_name: str, target: str) -> Optional[FunctionLocation]:
        """Find function in target hierarchy"""
        current_target = target

        while current_target:
            # Try stdlib first
            stdlib_path = self.parser.get_module_path("stdlib", current_target)
            stdlib_functions = self.parser.extract_functions(stdlib_path)

            if function_name in stdlib_functions:
                return FunctionLocation(current_target, "stdlib")

            # Try builtins
            builtins_path = self.parser.get_module_path("builtins_target", current_target)
            builtins_functions = self.parser.extract_functions(builtins_path)

            if function_name in builtins_functions:
                return FunctionLocation(current_target, "builtins_target")

            # Move up the hierarchy
            current_target = ISPCTarget.get_parent(current_target)

        return None

    def classify_missing_function(self, function_name: str) -> FunctionClassification:
        """Classify why a function wasn't found"""
        if function_name in self.pseudo_functions:
            return FunctionClassification.PSEUDO
        elif function_name.startswith('__pseudo_'):
            return FunctionClassification.PSEUDO
        elif function_name.startswith('llvm.'):
            return FunctionClassification.LLVM_INTRINSIC
        elif function_name.startswith('_Z'):
            return FunctionClassification.MANGLED_CPP
        else:
            return FunctionClassification.NOT_FOUND


class DependencyVisualizer:
    """Main class for visualizing ISPC function dependencies"""

    def __init__(self, bitcode_dir: str):
        self.parser = LLVMBitcodeParser(bitcode_dir)
        self.function_finder = FunctionFinder(self.parser)
        self.visited_functions = set()

    def _show_target_hierarchy(self, target: str):
        """Display the target hierarchy chain"""
        hierarchy_chain = []
        current_target = target
        while current_target:
            hierarchy_chain.append(current_target)
            current_target = ISPCTarget.get_parent(current_target)
        print("Hierarchy: " + " -> ".join(hierarchy_chain))

    def _handle_special_function_types(self, call_graph: Dict, function_name: str, target: str, found_in_targets: List, not_found_in_targets: List):
        """Handle special function types (NOT_FOUND, PSEUDO, etc.)"""
        target_type = call_graph['target']

        if target_type == "NOT_FOUND":
            print(f"Function '{function_name}' not found in target hierarchy for '{target}'")
            not_found_in_targets.append(target)
        elif target_type == "PSEUDO":
            print(f"Function '{function_name}' is a compiler pseudo-function (placeholder)")
            found_in_targets.append((target, "PSEUDO"))
        elif target_type == "LLVM_INTRINSIC":
            print(f"Function '{function_name}' is an LLVM intrinsic")
            found_in_targets.append((target, "LLVM_INTRINSIC"))
        elif target_type == "MANGLED_CPP":
            print(f"Function '{function_name}' is a mangled C++ function")
            found_in_targets.append((target, "MANGLED_CPP"))
        else:
            print("\nDependency Tree:")
            self.print_call_graph(call_graph)
            found_in_targets.append((target, call_graph['target']))

    def get_all_functions_in_hierarchy(self, target: str) -> Dict[str, Set[str]]:
        """Get all functions available in the target hierarchy"""
        all_functions = {}
        current_target = target

        while current_target:
            # Try stdlib
            stdlib_path = self.parser.get_module_path("stdlib", current_target)
            stdlib_functions = self.parser.extract_functions(stdlib_path)
            if stdlib_functions:
                all_functions[f"{current_target}:stdlib"] = stdlib_functions

            # Try builtins
            builtins_path = self.parser.get_module_path("builtins_target", current_target)
            builtins_functions = self.parser.extract_functions(builtins_path)
            if builtins_functions:
                all_functions[f"{current_target}:builtins_target"] = builtins_functions

            # Move up the hierarchy
            current_target = ISPCTarget.get_parent(current_target)

        return all_functions

    def decode_ispc_mangling(self, mangled_name: str) -> str:
        """Decode ISPC function name mangling to show types"""
        # Use constants for type mappings
        type_mapping = Constants.TYPE_MAPPING

        # Extract base function name and parameters
        if '___' in mangled_name:
            base_name, params = mangled_name.split('___', 1)

            # Try to decode the parameter types
            decoded_params = []
            i = 0
            while i < len(params):
                found_type = False
                # Try to match type codes (longer ones first)
                for type_code, type_name in sorted(type_mapping.items(), key=len, reverse=True):
                    if params[i:].startswith(type_code):
                        decoded_params.append(type_name)
                        i += len(type_code)
                        found_type = True
                        break

                if not found_type:
                    # Skip unknown characters
                    i += 1

            if decoded_params:
                return f"{base_name}({', '.join(decoded_params)})"

        return mangled_name

    def suggest_similar_functions(self, function_name: str, target: str, max_suggestions: int = Constants.MAX_SUGGESTIONS) -> List[Tuple[str, str, str]]:
        """Suggest similar function names when a function is not found"""
        all_functions = self.get_all_functions_in_hierarchy(target)
        suggestions = []

        # Extract base name for matching (remove common prefixes/suffixes)
        base_name = function_name.lower()

        for module, functions in all_functions.items():
            for func in functions:
                func_lower = func.lower()

                # Check for various similarity patterns
                if (base_name in func_lower or
                    func_lower.startswith(base_name) or
                    any(base_name.startswith(prefix) and prefix in func_lower
                        for prefix in [base_name[:i] for i in range(3, len(base_name)+1)])):
                    decoded = self.decode_ispc_mangling(func)
                    suggestions.append((func, module, decoded))

        # Sort by similarity (prioritize exact prefix matches)
        def similarity_score(item):
            func, _, _ = item
            func_lower = func.lower()
            if func_lower.startswith(base_name):
                return 0  # Highest priority for prefix matches
            elif base_name in func_lower:
                return 1  # Medium priority for substring matches
            else:
                return 2  # Lower priority for partial matches

        suggestions.sort(key=similarity_score)
        return suggestions[:max_suggestions]


    def build_call_graph(self, function_name: str, target: str, depth: int = 0, max_depth: int = Constants.DEFAULT_MAX_DEPTH, hide_pseudo: bool = False) -> Dict:
        """Build call graph for a function starting from given target"""
        if depth > max_depth or function_name in self.visited_functions:
            return {"function": function_name, "target": "...", "calls": []}

        # If hiding pseudo functions and this is a pseudo function, return None to skip
        if hide_pseudo and self.function_finder.classify_missing_function(function_name) == FunctionClassification.PSEUDO:
            return None

        self.visited_functions.add(function_name)

        # Find where this function is defined
        location = self.function_finder.find_function_in_hierarchy(function_name, target)
        if not location:
            missing_type = self.function_finder.classify_missing_function(function_name)
            return {"function": function_name, "target": missing_type.value, "calls": []}

        found_target, module_type = location.target, location.module_type

        # Get the calls made by this function
        module_path = self.parser.get_module_path(module_type, found_target)
        called_functions = self.parser.get_function_calls(module_path, function_name)

        # Recursively build call graph for called functions
        call_graphs = []
        for called_func in sorted(called_functions):
            if called_func != function_name:  # Avoid self-recursion
                call_graph = self.build_call_graph(called_func, target, depth + 1, max_depth, hide_pseudo)

                # Skip if None (pseudo function was filtered out)
                if call_graph is None:
                    continue

                call_graphs.append(call_graph)

        self.visited_functions.remove(function_name)  # Allow revisiting in other branches

        return {
            "function": function_name,
            "target": found_target,
            "module": module_type,
            "calls": call_graphs
        }

    def print_call_graph(self, graph: Dict, prefix: str = "", is_last: bool = True):
        """Print call graph in ASCII tree format"""
        if not graph:
            return

        # Current function info
        connector = "|__ "

        # Format target info with explanation for special cases
        target = graph['target']
        module = graph.get('module', 'unknown')

        if target == "PSEUDO":
            target_info = f" [PSEUDO:compiler_placeholder]"
        elif target == "LLVM_INTRINSIC":
            target_info = f" [LLVM_INTRINSIC:builtin]"
        elif target == "MANGLED_CPP":
            target_info = f" [MANGLED_CPP:external]"
        elif target == "NOT_FOUND":
            target_info = f" [NOT_FOUND:missing]"
        elif target == "...":
            target_info = f" [RECURSIVE:pruned]"
        else:
            target_info = f" [{target}:{module}]"

        print(f"{prefix}{connector}{graph['function']}{target_info}")

        # Prepare prefix for children
        extension = "    " if is_last else "|   "
        new_prefix = prefix + extension

        # Print called functions
        calls = graph.get('calls', [])
        for i, call in enumerate(calls):
            is_last_call = (i == len(calls) - 1)
            self.print_call_graph(call, new_prefix, is_last_call)

    def analyze_all_root_targets(self, function_name: str, max_depth: int = Constants.DEFAULT_MAX_DEPTH, hide_pseudo: bool = False, width_filter: str = None):
        """Analyze function dependencies across all root targets (excluding generic targets)"""
        if width_filter:
            root_targets = ISPCTarget.get_targets_by_width(width_filter)
            print(f"Analyzing dependencies for function '{function_name}' across all root targets with width '{width_filter}'")
        else:
            root_targets = ISPCTarget.get_non_generic_root_targets()
            print(f"Analyzing dependencies for function '{function_name}' across all root targets")

        print("Note: Generic targets are skipped as they are obvious fallbacks")
        if hide_pseudo:
            print("Note: Hiding pseudo/compiler placeholder functions")
        print(f"Max depth: {max_depth}")
        print(f"Root targets: {', '.join(root_targets)}")
        print("=" * 80)

        found_in_targets = []
        not_found_in_targets = []

        for target in root_targets:
            print(f"\n[TARGET: {target}]")
            print("-" * 40)

            # Reset visited functions for each target
            self.visited_functions.clear()

            # Show target hierarchy chain
            self._show_target_hierarchy(target)

            call_graph = self.build_call_graph(function_name, target,
                                             max_depth=max_depth, hide_pseudo=hide_pseudo)

            self._handle_special_function_types(call_graph, function_name, target, found_in_targets, not_found_in_targets)

        return found_in_targets, not_found_in_targets


def main():
    parser = argparse.ArgumentParser(
        description="Dump function call dependencies in ISPC stdlib and builtins modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s shift___vyiuni avx512spr-x4
  %(prog)s shift___vyiuni avx512spr-x4 --hide-pseudo
  %(prog)s __shuffle_i32 avx512skx-x4 --bitcode-dir /path/to/bitcode
  %(prog)s shift___vyiuni avx512spr-x4 --max-depth 5 --hide-pseudo
  %(prog)s __reduce_equal all --hide-pseudo
  %(prog)s __reduce_equal all-x4 --hide-pseudo

Function Classification:
  [target:module]           - Function found in specific target and module
  [PSEUDO:compiler_placeholder] - Compiler placeholder (gets optimized away)
  [LLVM_INTRINSIC:builtin] - LLVM built-in intrinsic function
  [MANGLED_CPP:external]   - Mangled C++ function name
  [NOT_FOUND:missing]      - Function not found in target hierarchy
  [RECURSIVE:pruned]       - Already visited (prevents infinite recursion)
        """
    )

    parser.add_argument("function_name",
                       help="Name of the ISPC function to analyze")
    parser.add_argument("target",
                       help="ISPC target (e.g., avx512spr-x4, avx2-i32x8, neon-i32x4), 'all' for all root targets, or 'all-x4' for all targets with width x4")
    parser.add_argument("--hide-pseudo",
                       action="store_true",
                       help="Hide compiler pseudo-functions from output")
    parser.add_argument("--bitcode-dir",
                       default=Constants.DEFAULT_BITCODE_DIR,
                       help="Directory containing LLVM bitcode files (default: %(default)s)")
    parser.add_argument("--max-depth",
                       type=int,
                       default=Constants.DEFAULT_MAX_DEPTH,
                       help="Maximum recursion depth for call graph (default: %(default)s)")

    args = parser.parse_args()

    if not os.path.exists(args.bitcode_dir):
        print(f"Error: Bitcode directory not found: {args.bitcode_dir}")
        sys.exit(1)

    visualizer = DependencyVisualizer(args.bitcode_dir)

    # Handle 'all' target case
    if args.target.lower() == 'all':
        visualizer.analyze_all_root_targets(args.function_name,
                                          max_depth=args.max_depth,
                                          hide_pseudo=args.hide_pseudo)
        return

    # Handle 'all-x4', 'all-x8', etc. cases
    if args.target.lower().startswith('all-'):
        width_filter = args.target[4:]  # Extract width part after 'all-'
        visualizer.analyze_all_root_targets(args.function_name,
                                          max_depth=args.max_depth,
                                          hide_pseudo=args.hide_pseudo,
                                          width_filter=width_filter)
        return

    # Single target analysis
    print(f"Analyzing dependencies for function '{args.function_name}' on target '{args.target}'")
    if args.hide_pseudo:
        print("Note: Hiding pseudo/compiler placeholder functions")
    print(f"Bitcode directory: {args.bitcode_dir}")
    print(f"Max depth: {args.max_depth}")
    print("-" * 80)

    # Show target hierarchy chain
    print("Target Hierarchy Chain:")
    hierarchy_chain = []
    current_target = args.target
    while current_target:
        hierarchy_chain.append(current_target)
        current_target = ISPCTarget.get_parent(current_target)

    print("  " + " -> ".join(hierarchy_chain))
    print()

    call_graph = visualizer.build_call_graph(args.function_name, args.target,
                                            max_depth=args.max_depth, hide_pseudo=args.hide_pseudo)

    if call_graph['target'] in ["NOT_FOUND", "PSEUDO", "LLVM_INTRINSIC", "MANGLED_CPP"]:
        if call_graph['target'] == "NOT_FOUND":
            print(f"Function '{args.function_name}' not found in target hierarchy for '{args.target}'")

            # Suggest similar function names
            suggestions = visualizer.suggest_similar_functions(args.function_name, args.target)
            if suggestions:
                print(f"\nDid you mean one of these functions?")
                for func, module, decoded in suggestions:
                    print(f"  {func} [{module}] -> {decoded}")
            else:
                print("\nNo similar function names found.")
            sys.exit(1)
        elif call_graph['target'] == "PSEUDO":
            print(f"Function '{args.function_name}' is a compiler pseudo-function (placeholder)")
            sys.exit(1)
        elif call_graph['target'] == "LLVM_INTRINSIC":
            print(f"Function '{args.function_name}' is an LLVM intrinsic")
            sys.exit(1)
        elif call_graph['target'] == "MANGLED_CPP":
            print(f"Function '{args.function_name}' is a mangled C++ function")
            sys.exit(1)

    print("\nDependency Tree:")
    visualizer.print_call_graph(call_graph)


if __name__ == "__main__":
    main()
