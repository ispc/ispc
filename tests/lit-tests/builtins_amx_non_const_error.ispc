// Test that AMX builtins emit an error when tile argument is not a compile-time constant.
// AMX tile arguments must be constant values between 0-7 as required by LLVM AMX intrinsics.

// RUN: not %{ispc} %s --target=avx512spr-x8 --nowrap -o %t.o 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED && LLVM_20_0+ && !MACOS_HOST

// CHECK: Error: AMX tile argument must be compile time constant.

#include <amx.isph>

__attribute__((external_only))
export void test_amx_non_const_tile(uniform uint8 tile_id, uniform int8* uniform data) {
    // tile_id is a runtime value, not a compile-time constant
    amx_tile_zero(tile_id);
}
