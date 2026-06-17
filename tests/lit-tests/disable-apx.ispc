// Tests for the --opt=disable-apx option that turns off x86 APX sub-features.

// An unknown sub-feature name is rejected with a helpful message.
//; RUN: not %{ispc} %s --target=avx10.2dmr-x16 --opt=disable-apx=egpr,bogus --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-NAME

// Using the option when no target is APX-capable is a no-op and warns once.
//; RUN: %{ispc} %s --target=avx2-i32x8 --opt=disable-apx --nowrap -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOOP

// In a multi-target build with at least one APX-capable target the option is
// effective, so no "no effect" warning is emitted.
//; RUN: %{ispc} %s --target=avx2-i32x8,avx10.2dmr-x16 --opt=disable-apx --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=MULTI --allow-empty

// In a multi-target build where no target is APX-capable the warning is emitted
// exactly once, not once per target.
//; RUN: %{ispc} %s --target=sse4-i32x8,avx2-i32x8,avx512skx-x16 --opt=disable-apx --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=ONCE

// When the APX-capable target is derived from --cpu rather than spelled out
// with --target, the option is still effective, so no "no effect" warning is
// emitted (regression test: the check must run after target resolution).
//; RUN: %{ispc} %s --cpu=dmr --opt=disable-apx --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CPU-DERIVED --allow-empty

// Conversely, a non-APX target derived from --cpu still warns.
//; RUN: %{ispc} %s --cpu=skx --opt=disable-apx --nowrap -o %t.o 2>&1 | FileCheck %s --check-prefix=CPU-NOOP

// Empty tokens from leading/embedded commas are tolerated, not rejected.
//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --opt=disable-apx=,egpr,,nf --print-target -o /dev/null | FileCheck %s --check-prefix=EMPTY-TOK

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_20_0+

// BAD-NAME: Error: Unknown APX sub-feature "bogus" in --opt=disable-apx.

// NOOP: Warning: --opt=disable-apx has no effect: none of the selected targets is APX-capable.

// MULTI-NOT: --opt=disable-apx has no effect

// CPU-DERIVED-NOT: --opt=disable-apx has no effect

// CPU-NOOP: Warning: --opt=disable-apx has no effect: none of the selected targets is APX-capable.

// The warning must appear exactly once regardless of how many non-APX targets
// are selected.
// ONCE-COUNT-1: --opt=disable-apx has no effect
// ONCE-NOT: --opt=disable-apx has no effect

// EMPTY-TOK: Target Feature String: -egpr,-nf

void foo(uniform int a[], uniform int n) {
    for (uniform int i = 0; i < n; ++i)
        a[i] = a[i] + 1;
}
