// The test checks that the triple contain minimum OS version if provided.
// Note: iOS target is enabled one ARM platforms only.

// RUN: %{ispc} %s --nostdlib --target-os=ios --arch=aarch64 --emit-llvm-text --nowrap --target=neon-i32x4 -o - | FileCheck %s --check-prefix=CHECK-IOS-DEFAULT
// RUN: %{ispc} %s --nostdlib --target-os=ios --arch=aarch64 --emit-llvm-text --nowrap --target=neon-i32x4 --darwin-version-min=15.0 -o - | FileCheck %s --check-prefix=CHECK-IOS-VER
// RUN: %{ispc} %s --nostdlib --target-os=ios --arch=aarch64 --emit-llvm-text --nowrap --target=neon-i32x4 --darwin-version-min="" -o - | FileCheck %s --check-prefix=CHECK-IOS-VER-UNSET

// REQUIRES: MACOS_HOST && MACOS_ARM_ENABLED && MACOS_IOS_ENABLED

// CHECK-IOS-DEFAULT: target triple = "arm64-apple-ios11.0"
// CHECK-IOS-VER: target triple = "arm64-apple-ios15.0"
// CHECK-IOS-VER-UNSET: target triple = "arm64-apple-ios"

uniform int j;

int foo(int i) { return i + 1; }
