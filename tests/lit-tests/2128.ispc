// This test checks that __cast_mask_to_i32/__cast_mask_to_i64 were inlined
// and removed from final module.

// RUN: %{ispc} --target avx512skx-x32 --emit-llvm-text %s -o - | FileCheck %s
// RUN: %{ispc} --target avx512skx-x64 --emit-llvm-text %s -o - | FileCheck %s
// REQUIRES: X86_ENABLED

// CHECK-NOT: @__cast_mask_to_i32
// CHECK-NOT: @__cast_mask_to_i64
export void A() {}
