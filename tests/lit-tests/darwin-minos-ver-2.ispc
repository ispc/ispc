// The test checks that the triple contain minimum OS version if provided.

// RUN: %{ispc} %s --nostdlib --target-os=macos --arch=aarch64 --emit-llvm-text --nowrap --target=neon-i32x4 -o - | FileCheck %s --check-prefix=CHECK-MACOS-DEFAULT
// RUN: %{ispc} %s --nostdlib --target-os=macos --arch=aarch64 --emit-llvm-text --nowrap --target=neon-i32x4 --darwin-version-min=15.0 -o - | FileCheck %s --check-prefix=CHECK-MACOS-VER
// RUN: %{ispc} %s --nostdlib --target-os=macos --arch=aarch64 --emit-llvm-text --nowrap --target=neon-i32x4 --darwin-version-min="" -o - | FileCheck %s --check-prefix=CHECK-MACOS-VER-UNSET

// RUN: not %{ispc} %s --nostdlib --target-os=macos --arch=aarch64 --nowrap --target=neon-i32x4 --darwin-version-min=a.b -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR-VER

// REQUIRES: MACOS_HOST && MACOS_ARM_ENABLED && ARM_ENABLED

// CHECK-MACOS-DEFAULT: target triple = "arm64-apple-macosx11.0"
// CHECK-MACOS-VER: target triple = "arm64-apple-macosx15.0"
// CHECK-MACOS-VER-UNSET: target triple = "arm64-apple-macosx"

// CHECK-ERROR-VER: Error: Invalid version format: "a.b". Use <major_ver.minor_ver>.

uniform int j;

int foo(int i) { return i + 1; }