// The test checks correct features for ARM/AARCH64 for cross compilation on Windows. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --target-os=linux --print-target 2>&1 | FileCheck %s -check-prefix=CHECK_AARCH64_LINUX
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --target-os=custom_linux --print-target 2>&1 | FileCheck %s -check-prefix=CHECK_AARCH64_CUSTOM_LINUX
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --target-os=linux --print-target 2>&1 | FileCheck %s -check-prefix=CHECK_ARM_LINUX
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --target-os=custom_linux --print-target 2>&1 | FileCheck %s -check-prefix=CHECK_ARM_CUSTOM_LINUX

// There are slightly different features before LLVM 17.0
// REQUIRES: ARM_ENABLED && WINDOWS_HOST && LINUX_ENABLED && LLVM_17_0+

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