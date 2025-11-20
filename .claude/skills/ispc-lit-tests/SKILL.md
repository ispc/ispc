---
name: ispc-lit-tests
description: Best practices for creating ISPC lit tests. Use when writing regression tests, verifying code generation, or checking compiler diagnostics.
---

# ISPC Lit Tests

A concise guide for writing **lit tests** for the ISPC.
These tests ensure compiler correctness, verify generated code, and prevent regressions.

---

## When to Use Lit Tests

Use lit tests when validating:

- **Compiler output** — LLVM IR, assembly, or AST.
- **Diagnostics** — warnings, errors, or other emitted messages.
- **Platform behavior** — verifying cross-platform or target-specific differences.
- **Regression coverage** — reproducing and locking fixes for known compiler issues.

---

## Core Guidelines

### Always Use `--nowrap`
Prevents line wrapping in compiler output for consistent FileCheck matching:
```ispc
// RUN: %{ispc} %s --target=host --nowrap --emit-llvm-text -o - | FileCheck %s
```

### Use `--nostdlib` When Not Testing Library Code
Simplifies test output and avoids unrelated symbols:
```ispc
// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o - | FileCheck %s
```

## Avoid `export` Unless Testing It

`export` functions generate both masked and unmasked IR — doubling the verification effort.

```ispc
// Preferred
void foo() { ... }

// Avoid unless explicitly testing export behavior
export void foo() { ... }
```

## Target Specification

### Generic / Portable Tests

Use `--target=host` unless verifying target-specific codegen:

```ispc
// RUN: %{ispc} %s --target=host --nowrap -o - | FileCheck %s
```

#### Writing Portable Checks

Avoid hardcoding vector widths or variable names.  
Use named patterns like `[[WIDTH]]` and `[[TYPE]]`.

Example:
```ispc
// CHECK-NEXT:  %test = sdiv <[[WIDTH:.*]] x i32> %a, %b
// CHECK-NEXT:  ret <[[WIDTH]] x i32> %test
```

When order is flexible:
```ispc
// CHECK-DAG: {{%.*}} = shufflevector <[[WIDTH:.*]] x [[BASE_TYPE:i.*]]> {{%.*}}, <[[WIDTH]] x [[BASE_TYPE]]> {{poison|undef}}, <[[WIDTH]] x [[BASE_TYPE]]> zeroinitializer
```

**Tip:** Avoid relying on exact variable names — they differ between OS and LLVM versions.


### Target-Specific Tests

When output differs by architecture or ISA:

- Specify the **exact target and feature**.
- Include a `REQUIRES:` directive for conditional execution.

Example:
```ispc
// RUN: %{ispc} %s --target=avx512skx-x16 --emit-asm -o - | FileCheck %s
// REQUIRES: X86_ENABLED
```

## Using `REQUIRES` for Feature Dependencies

Defined in `tests/lit-tests/lit.cfg`:

- **Features:** `X86_ENABLED`, `LLVM_*_0+`, etc.
- **Substitutions:** `%{ispc}`, `%s`, `%t`
- **Test configuration:** format, suffixes, and substitutions

## Testing Intermediate IR

Use `--debug-phase` to capture output of specific optimization passes:

```ispc
// RUN: %{ispc} %s --target=avx2 --emit-llvm-text \
// RUN:   --debug-phase=325:325 --dump-file=%t -o /dev/null
// RUN: FileCheck --input-file %t/ir_325_LoadStoreVectorizerPass.ll %s
```

## Comments and Documentation

Clearly describe what the test verifies and why it exists.

Example:
```ispc
// Verifies that stmxcsr/ldmxcsr intrinsics correctly set/restore FTZ/DAZ flags
// when --opt=reset-ftz-daz is enabled.
```

## Example Template

```ispc
// Brief description of the test purpose
// RUN: %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s

// REQUIRES: <feature_if_needed>

// CHECK-LABEL: @function_name___
// CHECK: expected pattern
// CHECK-NOT: unexpected pattern

void function_name() {
    // Minimal reproducible test code here
}
```

## Test commands
Run all lit tests:
```bash
cmake --build build --target check-all -j $(nproc)
```

To test the specific test, run:
```bash
TEST=/full/path/test.ispc cmake --build build --target check-one -j $(nproc)
```

## Test names
- Regression tests: name them `####.ispc`, where #### is the GitHub issue number.
- Other tests: use a short, descriptive name. For multiple tests of one feature, add numbers (e.g., `feature-name-1.ispc`, `feature-name-2.ispc`).

## Key Takeaways

- Keep tests **minimal** — validate one behavior per test.
- Use **portable patterns** for LLVM IR.
- Add **REQUIRES** for target-dependent tests.
- Prefer **non-exported** functions unless necessary.
- Document **intent** and **expected outcome** in comments.

