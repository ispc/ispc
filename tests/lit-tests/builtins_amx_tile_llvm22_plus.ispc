// This test checks that AMX tile (amxtile) builtins are correctly handled by ISPC on NVL target

// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x4 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x8 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x16 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x32 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR
// RUN: not %{ispc} --nowrap %s --target=avx10.2nvl-x64 --emit-llvm-text -O2 -o - 2>&1 | FileCheck %s -check-prefix=CHECK_NVL_ERROR

// REQUIRES: X86_ENABLED && LLVM_22_0+ && !MACOS_HOST

// CHECK_NVL_ERROR: Error: Some AMX functions used in the source are not supported on the current target

#include <amx.isph>

__attribute__((external_only))
export void test_amx_tile(uniform int8* uniform config, uniform int8* uniform data) {
    // Base AMX tile operations - require amxtile feature
    amx_tile_loadconfig(config);
    amx_tile_storeconfig(config);
    amx_tile_zero(2);
    amx_tile_load(0, data, (uniform int64)64);
    amx_tile_load_t1(1, data, (uniform int64)64);
    amx_tile_store(0, data, (uniform int64)64);
    amx_tile_release();
}
