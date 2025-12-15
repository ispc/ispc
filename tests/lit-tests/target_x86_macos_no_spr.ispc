// The test checks that target definitions for SPR and GNR are not accepted on macOS.

//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x4  2>&1 | FileCheck %s -check-prefix=CHECK_SPR
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x8  2>&1 | FileCheck %s -check-prefix=CHECK_SPR
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x16 2>&1 | FileCheck %s -check-prefix=CHECK_SPR
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x32 2>&1 | FileCheck %s -check-prefix=CHECK_SPR
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512spr-x64 2>&1 | FileCheck %s -check-prefix=CHECK_SPR

//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512gnr-x4  2>&1 | FileCheck %s -check-prefix=CHECK_GNR
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512gnr-x8  2>&1 | FileCheck %s -check-prefix=CHECK_GNR
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512gnr-x16 2>&1 | FileCheck %s -check-prefix=CHECK_GNR
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512gnr-x32 2>&1 | FileCheck %s -check-prefix=CHECK_GNR
//; RUN: not %{ispc} %s -o %t.o --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx512gnr-x64 2>&1 | FileCheck %s -check-prefix=CHECK_GNR

// REQUIRES: X86_ENABLED && MACOS_HOST

// CHECK_SPR: Error: avx512spr-x{{.*}} target for x86-64 on macOS is not supported in current build.
// CHECK_GNR: Error: avx512gnr-x{{.*}} target for x86-64 on macOS is not supported in current build.

uniform int j;

int foo(int i) { return i + 1; }
