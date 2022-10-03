// RUN: %{ispc} %s -E --nostdlib --target=sse4-i32x4 | FileCheck %s.filecheck -check-prefix=CHECK_PASS_i32x4
// RUN: %{ispc} %s -E --nostdlib --target=avx2-i32x8 | FileCheck %s.filecheck -check-prefix=CHECK_PASS_i32x8

// RUN: %{ispc} %s -E --nostdlib --target=sse4-i32x4,avx2-i32x8 -o %t.i
// RUN: cat %t_sse4.i                                | FileCheck %s.filecheck -check-prefix=CHECK_PASS_i32x4
// RUN: cat %t_avx2.i                                | FileCheck %s.filecheck -check-prefix=CHECK_PASS_i32x8

// REQUIRES: X86_ENABLED


uniform int foo() { return TARGET_WIDTH; /*target_width*/ }

uniform int bar() { return TARGET_ELEMENT_WIDTH; /*target_element_width*/ }
