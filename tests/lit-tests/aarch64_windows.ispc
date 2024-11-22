// The test checks correct features for AARCH64 for different cpus on Windows. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --print-target | FileCheck %s -check-prefix=CHECK_DEFAULT
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a15 --print-target 2>&1 | FileCheck %s -check-prefix=CHECK_AARCH64_A15
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a35 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A35
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --arch=aarch64 --cpu=cortex-a57 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A57

// There are slightly different features before LLVM 17.0
// REQUIRES: ARM_ENABLED && WINDOWS_HOST && LLVM_17_0+

// CHECK_DEFAULT: Triple: aarch64-pc-windows-msvc
// CHECK_DEFAULT-NEXT: CPU: {{cortex-a*}}
// CHECK_DEFAULT-NEXT: Feature String:
// CHECK_DEFAULT-SAME: +aes
// CHECK_DEFAULT-SAME: +crc
// CHECK_DEFAULT-SAME: +fp-armv8
// CHECK_DEFAULT-SAME: +neon
// CHECK_DEFAULT-SAME: +sha2

// CHECK_AARCH64: Triple: aarch64-pc-windows-msvc
// CHECK_AARCH64-NEXT: CPU: {{cortex-a*}}
// CHECK_AARCH64-NEXT: Feature String:
// CHECK_AARCH64-SAME: +aes
// CHECK_AARCH64-SAME: +crc
// CHECK_AARCH64-SAME: +fp-armv8
// CHECK_AARCH64-SAME: +neon
// CHECK_AARCH64-SAME: +sha2

// CHECK_AARCH64_A15: Error: Invalid CPU name for AArch64 architecture: cortex-a15

// CHECK_AARCH64_A35: Triple: aarch64-pc-windows-msvc
// CHECK_AARCH64_A35-NEXT: CPU: cortex-a35
// CHECK_AARCH64_A35-NEXT: Feature String:
// CHECK_AARCH64_A35-SAME: +aes
// CHECK_AARCH64_A35-SAME: +crc
// CHECK_AARCH64_A35-SAME: +fp-armv8
// CHECK_AARCH64_A35-SAME: +neon
// CHECK_AARCH64_A35-SAME: +sha2

// CHECK_AARCH64_A57: Triple: aarch64-pc-windows-msvc
// CHECK_AARCH64_A57-NEXT: CPU: cortex-a57
// CHECK_AARCH64_A57-NEXT: Feature String:
// CHECK_AARCH64_A57-SAME: +aes
// CHECK_AARCH64_A57-SAME: +crc
// CHECK_AARCH64_A57-SAME: +fp-armv8
// CHECK_AARCH64_A57-SAME: +neon
// CHECK_AARCH64_A57-SAME: +sha2


uniform int i;

void foo() {}