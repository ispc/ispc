// Test that AMX builtins emit an error when tile argument is out of range (0-7).
// AMX tile arguments must be constant values between 0-7 as required by LLVM AMX intrinsics.

// RUN: not %{ispc} %s --target=avx512spr-x8 --nowrap -o %t.o 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED && LLVM_20_0+ && !MACOS_HOST

// CHECK: Error: AMX tile argument value must be between 0-7.

#include <amx.isph>

__attribute__((external_only))
export void test_amx_invalid_tile() {
    // tile 8 is out of range (valid range is 0-7)
    amx_tile_zero(8);
}
