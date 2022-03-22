// RUN: %{ispc} %s --nostdlib --target=sse4-i32x4            -E | grep -v -e '^// ' | FileCheck %s -check-prefix=CHECK_PASS_i32x4
// RUN: %{ispc} %s --nostdlib --target=avx2-i32x8            -E | grep -v -e '^// ' | FileCheck %s -check-prefix=CHECK_PASS_i32x8

// RUN: %{ispc} %s --nostdlib --target=sse4-i32x4,avx2-i32x8 -E -o %t.i
// RUN: cat %t_sse4.i | grep -v -e '^// ' | FileCheck %s -check-prefix=CHECK_PASS_i32x4
// RUN: cat %t_avx2.i | grep -v -e '^// ' | FileCheck %s -check-prefix=CHECK_PASS_i32x8

// CHECK_PASS_i32x4: return 4; /*target_width*/
// CHECK_PASS_i32x4: return 4; /*target_element_width*/

// CHECK_PASS_i32x8: return 8; /*target_width*/
// CHECK_PASS_i32x8: return 4; /*target_element_width*/

// REQUIRES: X86_ENABLED


uniform int foo() { return TARGET_WIDTH; /*target_width*/ }

uniform int bar() { return TARGET_ELEMENT_WIDTH; /*target_element_width*/ }
