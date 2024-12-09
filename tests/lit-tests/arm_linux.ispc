// The test checks correct features for ARM/AARCH64 for different cpus on Linux. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a15 --print-target 2>&1 | FileCheck %s -check-prefix=CHECK_AARCH64_A15
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a35 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A35
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a57 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A57
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a55 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A55
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a78 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A78
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a510 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A510

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --print-target | FileCheck %s -check-prefix=CHECK_ARM
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --cpu=cortex-a9 --print-target | FileCheck %s -check-prefix=CHECK_ARM_A9
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --cpu=cortex-a15 --print-target | FileCheck %s -check-prefix=CHECK_ARM_A15
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=arm --cpu=cortex-a57 --print-target | FileCheck %s -check-prefix=CHECK_ARM_A57

// arm supported on Linux only
// REQUIRES: ARM_ENABLED && LINUX_HOST && LLVM_17_0+

// CHECK_DEFAULT: Triple: aarch64-unknown-linux-gnu
// CHECK_DEFAULT: CPU: {{cortex-a*}}
// CHECK_DEFAULT-NEXT: Feature String:
// CHECK_DEFAULT-SAME: +aes
// CHECK_DEFAULT-SAME: +crc
// CHECK_DEFAULT-SAME: +fp-armv8
// CHECK_DEFAULT-SAME: +neon
// CHECK_DEFAULT-SAME: +sha2

// CHECK_AARCH64: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64: CPU: {{cortex-a*}}
// CHECK_AARCH64-NEXT: Feature String:
// CHECK_AARCH64-SAME: +aes
// CHECK_AARCH64-SAME: +crc
// CHECK_AARCH64-SAME: +fp-armv8
// CHECK_AARCH64-SAME: +neon
// CHECK_AARCH64-SAME: +sha2

// CHECK_AARCH64_A15: Error: Invalid CPU name for AArch64 architecture: cortex-a15

// CHECK_AARCH64_A35: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_A35: CPU: cortex-a35
// CHECK_AARCH64_A35-NEXT: Feature String:
// CHECK_AARCH64_A35-SAME: +aes
// CHECK_AARCH64_A35-SAME: +crc
// CHECK_AARCH64_A35-SAME: +fp-armv8
// CHECK_AARCH64_A35-SAME: +neon
// CHECK_AARCH64_A35-SAME: +sha2

// CHECK_AARCH64_A57: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_A57: CPU: cortex-a57
// CHECK_AARCH64_A57-NEXT: Feature String:
// CHECK_AARCH64_A57-SAME: +aes
// CHECK_AARCH64_A57-SAME: +crc
// CHECK_AARCH64_A57-SAME: +fp-armv8
// CHECK_AARCH64_A57-SAME: +neon
// CHECK_AARCH64_A57-SAME: +sha2

// CHECK_AARCH64_A55: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_A55-NEXT: CPU: cortex-a55
// CHECK_AARCH64_A55-NEXT: Feature String:
// CHECK_AARCH64_A55-SAME: +aes
// CHECK_AARCH64_A55-SAME: +crc
// CHECK_AARCH64_A55-SAME: +dotprod
// CHECK_AARCH64_A55-SAME: +fp-armv8
// CHECK_AARCH64_A55-SAME: +fullfp16
// CHECK_AARCH64_A55-SAME: +lse
// CHECK_AARCH64_A55-SAME: +neon
// CHECK_AARCH64_A55-SAME: +rcpc
// CHECK_AARCH64_A55-SAME: +sha2

// CHECK_AARCH64_A78: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_A78-NEXT: CPU: cortex-a78
// CHECK_AARCH64_A78-NEXT: Feature String:
// CHECK_AARCH64_A78-SAME: +aes
// CHECK_AARCH64_A78-SAME: +crc
// CHECK_AARCH64_A78-SAME: +dotprod
// CHECK_AARCH64_A78-SAME: +fp-armv8
// CHECK_AARCH64_A78-SAME: +fullfp16
// CHECK_AARCH64_A78-SAME: +lse
// CHECK_AARCH64_A78-SAME: +neon
// CHECK_AARCH64_A78-SAME: +rcpc
// CHECK_AARCH64_A78-SAME: +sha2

// CHECK_AARCH64_A510: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_A510-NEXT: CPU: cortex-a510
// CHECK_AARCH64_A510-NEXT: Feature String:
// CHECK_AARCH64_A510-SAME: +crc
// CHECK_AARCH64_A510-SAME: +dotprod
// CHECK_AARCH64_A510-SAME: +fp-armv8
// CHECK_AARCH64_A510-SAME: +fp16fml
// CHECK_AARCH64_A510-SAME: +fullfp16
// CHECK_AARCH64_A510-SAME: +i8mm
// CHECK_AARCH64_A510-SAME: +lse
// CHECK_AARCH64_A510-SAME: +neon
// CHECK_AARCH64_A510-SAME: +rcpc
// CHECK_AARCH64_A510-SAME: +sve
// CHECK_AARCH64_A510-SAME: +sve2

// CHECK_ARM: Triple: armv7-unknown-linux-gnueabihf
// CHECK_ARM: CPU: {{cortex-a*}}
// CHECK_ARM-NEXT: Feature String:
// CHECK_ARM-SAME: +fp16
// CHECK_ARM-SAME: +neon

// CHECK_ARM_A9: Triple: armv7-unknown-linux-gnueabihf
// CHECK_ARM_A9: CPU: cortex-a9
// CHECK_ARM_A9-NEXT: Feature String:
// CHECK_ARM_A9-SAME: +fp16
// CHECK_ARM_A9-SAME: +neon

// CHECK_ARM_A15: Triple: armv7-unknown-linux-gnueabihf
// CHECK_ARM_A15: CPU: cortex-a15
// CHECK_ARM_A15-NEXT: Feature String:
// CHECK_ARM_A15-SAME: +fp16
// CHECK_ARM_A15-SAME: +neon

// CHECK_ARM_A57: Triple: armv7-unknown-linux-gnueabihf
// CHECK_ARM_A57: CPU: cortex-a57
// CHECK_ARM_A57-NEXT: Feature String:
// CHECK_ARM_A57-SAME: +fp16
// CHECK_ARM_A57-SAME: +neon

uniform int i;

void foo() {}