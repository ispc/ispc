// RUN: %{ispc} --target=generic-i1x4 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I1X4
// RUN: %{ispc} --target=generic-i1x8 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I1X8
// RUN: %{ispc} --target=generic-i1x16 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I1X16
// RUN: %{ispc} --target=generic-i1x32 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I1X32
// TODO!: %{ispc} --target=generic-i1x64 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I1X64
// RUN: %{ispc} --target=generic-i8x16 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I8X16
// RUN: %{ispc} --target=generic-i8x32 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I8X32
// RUN: %{ispc} --target=generic-i16x8 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I16X8
// RUN: %{ispc} --target=generic-i16x16 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I16X16
// RUN: %{ispc} --target=generic-i32x4 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I32X4
// RUN: %{ispc} --target=generic-i32x8 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I32X8
// RUN: %{ispc} --target=generic-i32x16 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I32X16
// RUN: %{ispc} --target=generic-i64x4 --arch=x86 --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-I64X4

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-I1X4: define <4 x i32> @bar___vyi(<4 x i32> returned %x, <4 x i1> %__mask)
// CHECK-I1X8: define <8 x i32> @bar___vyi(<8 x i32> returned %x, <8 x i1> %__mask)
// CHECK-I1X16: define <16 x i32> @bar___vyi(<16 x i32> returned %x, <16 x i1> %__mask)
// CHECK-I1X32: define <32 x i32> @bar___vyi(<32 x i32> returned %x, <32 x i1> %__mask)
// CHECK-I1X64: define <64 x i32> @bar___vyi(<64 x i32> returned %x, <64 x i1> %__mask)
// CHECK-I8X16: define <16 x i32> @bar___vyi(<16 x i32> returned %x, <16 x i8> %__mask)
// CHECK-I8X32: define <32 x i32> @bar___vyi(<32 x i32> returned %x, <32 x i8> %__mask)
// CHECK-I16X8: define <8 x i32> @bar___vyi(<8 x i32> returned %x, <8 x i16> %__mask)
// CHECK-I16X16: define <16 x i32> @bar___vyi(<16 x i32> returned %x, <16 x i16> %__mask)
// CHECK-I32X4: define <4 x i32> @bar___vyi(<4 x i32> returned %x, <4 x i32> %__mask)
// CHECK-I32X8: define <8 x i32> @bar___vyi(<8 x i32> returned %x, <8 x i32> %__mask)
// CHECK-I32X16: define <16 x i32> @bar___vyi(<16 x i32> returned %x, <16 x i32> %__mask)
// CHECK-I64X4: define <4 x i32> @bar___vyi(<4 x i32> returned %x, <4 x i64> %__mask)
int bar(int x) { return x; }
