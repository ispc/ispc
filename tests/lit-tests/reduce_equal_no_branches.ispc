// Check that reduce_equal implementation has no additional conditional branches for avx512 targets

// RUN: %{ispc} %s --target=avx512skx-x4 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x8 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x16 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x32 --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x64 --nowrap --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-LABEL: test_reduce_equal___vyi
// CHECK-NOT: j{{[a-z]}}
// CHECK: ret
uniform bool test_reduce_equal(int32 val) {
    uniform int32 result;
    return reduce_equal(val, &result);
}
