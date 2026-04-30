// Test that all 6 VSX targets are recognized and produce valid output.
// Also tests the "vsx" alias mapping to vsx-i32x4.

// RUN: %{ispc} %s --target=vsx-i8x16 --arch=ppc64le --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-I8X16
// RUN: %{ispc} %s --target=vsx-i8x32 --arch=ppc64le --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-I8X32
// RUN: %{ispc} %s --target=vsx-i16x8 --arch=ppc64le --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-I16X8
// RUN: %{ispc} %s --target=vsx-i16x16 --arch=ppc64le --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-I16X16
// RUN: %{ispc} %s --target=vsx-i32x4 --arch=ppc64le --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-I32X4
// RUN: %{ispc} %s --target=vsx-i32x8 --arch=ppc64le --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-I32X8
// RUN: %{ispc} %s --target=vsx --arch=ppc64le --target-os=linux --nowrap --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-I32X4

// REQUIRES: PPC64_ENABLED && LINUX_ENABLED

// CHECK-I8X16: <16 x i8>
// CHECK-I8X32: <32 x i8>
// CHECK-I16X8: <8 x i16>
// CHECK-I16X16: <16 x i16>
// CHECK-I32X4: <4 x i32>
// CHECK-I32X8: <8 x i32>

void vadd(uniform int a[], uniform int b[], uniform int result[], uniform int count) {
    foreach (i = 0 ... count) {
        result[i] = a[i] + b[i];
    }
}
