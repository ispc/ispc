// The test checks correct features for ARM (32-bit) for different cpus on Linux. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --print-target | FileCheck %s -check-prefix=CHECK_ARM
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --cpu=cortex-a57 --print-target | FileCheck %s -check-prefix=CHECK_ARM_A57

// arm supported on Linux only
// REQUIRES: ARM32_ENABLED && LINUX_HOST

// CHECK_ARM: Triple: armv8a-unknown-linux-gnueabihf
// CHECK_ARM: CPU: {{cortex-a*}}
// CHECK_ARM-NEXT: Feature String:
// CHECK_ARM-SAME: +fp16
// CHECK_ARM-SAME: +neon

// CHECK_ARM_A57: Triple: armv8a-unknown-linux-gnueabihf
// CHECK_ARM_A57: CPU: cortex-a57
// CHECK_ARM_A57-NEXT: Feature String:
// CHECK_ARM_A57-SAME: +fp16
// CHECK_ARM_A57-SAME: +neon

uniform int i;

void foo() {}
