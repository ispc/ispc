// The test checks correct features for ARM (32-bit) for cross compilation on macOS. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --target-os=linux --print-target  | FileCheck %s -check-prefixes=CHECK_ARM_LINUX
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --target-os=custom_linux --print-target  | FileCheck %s -check-prefixes=CHECK_ARM_CUSTOM_LINUX

// REQUIRES: ARM32_ENABLED && MACOS_ARM_ENABLED && LINUX_ENABLED

// CHECK_ARM_LINUX: Triple: armv8a-unknown-linux-gnueabihf
// CHECK_ARM_LINUX: CPU: cortex-a35
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
