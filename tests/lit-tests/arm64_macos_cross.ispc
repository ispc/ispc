// The test checks correct features for ARM/AARCH64 for cross compilation on macOS. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=ios --print-target  | FileCheck %s -check-prefixes=CHECK_ARM64_IOS
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=linux --print-target  | FileCheck %s -check-prefixes=CHECK_AARCH64_LINUX
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --print-target  | FileCheck %s -check-prefixes=CHECK_AARCH64_CUSTOM_LINUX
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --target-os=linux --print-target  | FileCheck %s -check-prefixes=CHECK_ARM_LINUX
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --target-os=custom_linux --print-target  | FileCheck %s -check-prefixes=CHECK_ARM_CUSTOM_LINUX

// There are slightly different features before LLVM 17.0
// REQUIRES: ARM_ENABLED && MACOS_ARM_ENABLED && LINUX_ENABLED && LLVM_17_0+

// CHECK_ARM64_IOS: Triple: arm64-apple-ios
// CHECK_ARM64_IOS-NEXT: CPU: apple-a7
// CHECK_ARM64_IOS-NEXT: Feature String:
// CHECK_ARM64_IOS-SAME: +aes
// CHECK_ARM64_IOS-SAME: +fp-armv8
// CHECK_ARM64_IOS-SAME: +neon
// CHECK_ARM64_IOS-SAME: +sha2

// CHECK_AARCH64_LINUX: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_LINUX-NEXT: CPU: cortex-a35
// CHECK_AARCH64_LINUX-NEXT: Feature String:
// CHECK_AARCH64_LINUX-SAME: +aes
// CHECK_AARCH64_LINUX-SAME: +crc
// CHECK_AARCH64_LINUX-SAME: +fp-armv8
// CHECK_AARCH64_LINUX-SAME: +neon
// CHECK_AARCH64_LINUX-SAME: +sha2

// CHECK_AARCH64_CUSTOM_LINUX: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_CUSTOM_LINUX-NEXT: CPU: cortex-a57
// CHECK_AARCH64_CUSTOM_LINUX-NEXT: Feature String:
// CHECK_AARCH64_CUSTOM_LINUX-SAME: +aes
// CHECK_AARCH64_CUSTOM_LINUX-SAME: +crc
// CHECK_AARCH64_CUSTOM_LINUX-SAME: +crypto
// CHECK_AARCH64_CUSTOM_LINUX-SAME: +fp-armv8
// CHECK_AARCH64_CUSTOM_LINUX-SAME: +neon
// CHECK_AARCH64_CUSTOM_LINUX-SAME: +sha2

// CHECK_ARM_LINUX: Triple: armv8a-unknown-linux-gnueabihf
// CHECK_ARM_LINUX: CPU: cortex-a9
// CHECK_ARM_LINUX-NEXT: Feature String:
// CHECK_ARM_LINUX-SAME: +fp16
// CHECK_ARM_LINUX-SAME: +neon

// CHECK_ARM_CUSTOM_LINUX: Triple: armv8a-unknown-linux-gnueabihf
// CHECK_ARM_CUSTOM_LINUX: CPU: cortex-a57
// CHECK_ARM_CUSTOM_LINUX-NEXT: Feature String:
// CHECK_ARM_CUSTOM_LINUX-SAME: +crypto
// CHECK_ARM_CUSTOM_LINUX-SAME: +fp-armv8
// CHECK_ARM_CUSTOM_LINUX-SAME: +neon
// CHECK_ARM_CUSTOM_LINUX-SAME: +sha2

uniform int i;

void foo() {}
