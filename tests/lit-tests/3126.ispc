// This test verifies that when an unknown target is specified on the command line,
// the complete list of supported targets is displayed.
// It's impractical to validate the entire output without imposing strict conditions on the test,
// which is unnecessary, so we'll verify only the beginning of the output when x86 targets are enabled
// (the targets are listed in alphabetical order).
// Additionally, confirm that the message about '--support-matrix' is printed when an unknown target/OS or target is specified.

// RUN: not %{ispc} --target=unknown --nowrap %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-TARGET
// RUN: not %{ispc} --target-os=unknown --nowrap %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-OS
// RUN: not %{ispc} --arch=unknown --nowrap %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ARCH
// REQUIRES: X86_ENABLED

// CHECK-TARGET: Incorrect targets: unknown. Choices are: {{avx1-i32x16, avx1-i32x4, avx1-i32x8, avx1-i64x4, avx2-i16x16, avx2-i32x16, avx2-i32x4, avx2-i32x8, .*}}. Use --support-matrix for complete information about supported targets, archs and target OSes
// CHECK-OS: Error: Unsupported value for --target-os, supported values are: {{.*}}. Use --support-matrix for complete information about supported targets, archs and target OSes.
// CHECK-ARCH: Error: Unsupported value for --arch, supported values are: {{.*}}. Use --support-matrix for complete information about supported targets, archs and target OSes.

uniform float i;
uniform float foo() {
    return i;
}