// The test checks that target definitions for AVX10.2 are not accepted on macOS.

//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx10.2dmr-x4  2>&1 | FileCheck %s
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx10.2dmr-x8  2>&1 | FileCheck %s
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx10.2dmr-x16 2>&1 | FileCheck %s
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx10.2dmr-x32 2>&1 | FileCheck %s
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx10.2dmr-x64 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED && MACOS_HOST && LLVM_20_0+

// CHECK: Error: avx10.2dmr-x{{.*}} target for x86-64 on macOS is not supported in current build.

uniform int j;

int foo(int i) { return i + 1; }
