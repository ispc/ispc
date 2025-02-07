// The test checks that the triple contain minimum OS version if provided.

// RUN: %{ispc} %s --nostdlib --target-os=macos --target=avx2 --arch=x86-64 --emit-llvm-text --nowrap -o - | FileCheck %s --check-prefix=CHECK-MACOS-DEFAULT
// RUN: %{ispc} %s --nostdlib --target-os=macos --target=avx2 --arch=x86-64 --emit-llvm-text --nowrap --darwin-version-min=15.0 -o - | FileCheck %s --check-prefix=CHECK-MACOS-VER
// RUN: %{ispc} %s --nostdlib --target-os=macos --target=avx2 --arch=x86-64 --emit-llvm-text --nowrap --darwin-version-min="" -o - | FileCheck %s --check-prefix=CHECK-MACOS-VER-UNSET
// RUN: not %{ispc} %s --nostdlib --target-os=macos --arch=x86-64 --nowrap --target=avx2 --darwin-version-min=a.b -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-VER

// REQUIRES: MACOS_HOST && X86_ENABLED

// CHECK-MACOS-DEFAULT: target triple = "x86_64-apple-macosx10.12"
// CHECK-MACOS-VER: target triple = "x86_64-apple-macosx15.0"
// CHECK-MACOS-VER-UNSET: target triple = "x86_64-apple-macosx"

// CHECK-ERROR-VER: Error: Invalid version format: "a.b". Use <major_ver.minor_ver>.

uniform int j;

int foo(int i) { return i + 1; }

