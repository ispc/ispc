// The test checks correct features for different linux cpus for arm (32-bit) custom_linux.

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=arm --print-target | FileCheck %s -check-prefix=CHECK_ARM
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=arm --cpu=cortex-a35 --print-target | FileCheck %s -check-prefix=CHECK_ARM_A35

// REQUIRES: LINUX_ENABLED && ARM32_ENABLED && !MACOS_HOST

// CHECK_ARM: Triple: armv8a-unknown-linux-gnueabihf
// CHECK_ARM-NEXT: CPU: cortex-a57
// CHECK_ARM-NEXT: Feature String: +crypto,+fp-armv8,+neon,+sha2

// CHECK_ARM_A35: Triple: armv8a-unknown-linux-gnueabihf
// CHECK_ARM_A35-NEXT: CPU: cortex-a35
// CHECK_ARM_A35-NEXT: Feature String: +crypto,+fp-armv8,+neon,+sha2

uniform int i;

void foo() {}
