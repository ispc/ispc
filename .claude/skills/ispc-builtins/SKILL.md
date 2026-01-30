---
name: ispc-builtins
description: Best practices for creating and modifying ISPC builtin files. Use when adding target-specific optimizations, implementing new builtin functions, or working with the hierarchical target system.
---

# ISPC Builtins Guide

Understanding the hierarchical target system is critical for correctly implementing and maintaining builtins.

---

## Architecture Overview

ISPC's standard library consist of two layers:

1. **Standard Library** (`stdlib/stdlib.ispc`) — User-visible functions written in ISPC language
2. **Builtins** (`builtins/`) — Target-specific implementations in LLVM IR that support the stdlib

---

## Hierarchical Target System

**Key Concept:** Targets are organized hierarchically. A child target inherits all functions from its parent that it doesn't explicitly override.

### Target Parent Map (`src/builtins.cpp`)

The hierarchy is defined in `targetParentMap`:

```cpp
std::unordered_map<ISPCTarget, ISPCTarget> targetParentMap = {
    // ARM NEON -> generic
    {ISPCTarget::neon_i8x16, ISPCTarget::generic_i8x16},

    // AVX512 hierarchy: dmr -> gnr -> spr -> icl -> skx -> generic
    {ISPCTarget::avx10_2dmr_x16, ISPCTarget::avx512gnr_x16},
    {ISPCTarget::avx512gnr_x16, ISPCTarget::avx512spr_x16},
    {ISPCTarget::avx512spr_x16, ISPCTarget::avx512icl_x16},
    {ISPCTarget::avx512icl_x16, ISPCTarget::avx512skx_x16},
    {ISPCTarget::avx512skx_x16, ISPCTarget::generic_i1x16},

    // AVX/SSE hierarchy: avx2vnni -> avx2 -> avx1 -> sse4 -> sse2 -> generic
    {ISPCTarget::avx2vnni_i32x8, ISPCTarget::avx2_i32x8},
    {ISPCTarget::avx2_i32x8, ISPCTarget::avx1_i32x8},
    // ...
};
```

### How Linking Works

When compiling user code:

1. Link the target-specific builtins (e.g., `avx512skx-x16`)
2. Check for unresolved symbols
3. If unresolved, link parent target's builtins (e.g., `generic-i1x16`)
4. Repeat until all symbols are resolved or error

**Implication:** You only need to implement functions that differ from the parent.

---

## File Structure (`builtins/`)

| File Pattern | Description |
|-------------|-------------|
| `target-<isa>-<variant>.ll` | Target-specific LLVM IR (e.g., `target-avx512skx-x16.ll`) |
| `target-<isa>-common.ll` | Shared code for ISA family (e.g., `target-sse4-common.ll`) |
| `target-<isa>-utils.ll` | Utility macros/functions for ISA (e.g., `target-avx512-utils.ll`) |
| `generic.ispc` | Target-independent implementations in ISPC |
| `util.m4` | M4 macros for generating LLVM IR |

Function signatures are declared in `stdlib/include/builtins.isph`.

---

## Creating/Modifying Builtins

### Step 1: Define the Function Signature

Add declaration to `stdlib/include/builtins.isph`:

```c
EXT inline READNONE varying float __my_builtin_float(varying float);
```

### Step 2: Implement Generic Fallback and Optimized Versions

**New builtins must always have a generic implementation in `generic.ispc`.** This ensures all targets work correctly via hierarchy fallback.

Then add optimized LLVM IR versions at the appropriate level — they will be inherited by all children:
- `avx512skx` — inherited by `icl`, `spr`, `gnr`, `dmr`
- `avx512icl` — inherited by `spr`, `gnr`, `dmr` (overrides `skx`)

Example optimized implementation in `target-avx512skx-x16.ll`:

```llvm
define <16 x float> @__my_builtin_float(<16 x float> %input) nounwind readnone alwaysinline {
    %result = call <16 x float> @llvm.x86.avx512.something(<16 x float> %input)
    ret <16 x float> %result
}
```

---

## Best Practices

1. **Leverage hierarchy** — Only implement what differs from parent; don't copy-paste
2. **Use generic as fallback** — Implement in `generic.ispc` first, optimize later
3. **Use `include()`** — Share code via `target-*-utils.ll` files
4. **Mark functions correctly** — Use `nounwind readnone alwaysinline` attributes

---

## Testing Builtins

See `CLAUDE.md` for build, lit test, and IR/assembly inspection commands.

### Verify Correct Builtin Was Linked

Dump IR before optimizations to see builtins as they were linked:

```bash
build/bin/ispc test.ispc --target=avx512skx-x16 --debug-phase=pre:first --dump-file=dbg -o /dev/null
# Check dbg/ir_*.ll files for the builtin implementation
```
