// The test checks that target definitions for SPR are not accepted on macOS.

//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x4  2>&1 | FileCheck %s
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x8  2>&1 | FileCheck %s
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x16 2>&1 | FileCheck %s
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x32 2>&1 | FileCheck %s
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x64 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED && LLVM_14_0+ && MACOS_HOST

// CHECK: Error: avx512spr-x{{.*}} target for x86-64 on macOS is not supported in current build.

uniform int j;

int foo(int i) { return i + 1; }
