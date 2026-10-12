// The test checks correct features for different linux cpus for aarch64 custom_linux.

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --print-target | FileCheck %s -check-prefix=CHECK_DEFAULT
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=aarch64 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=aarch64 --cpu=cortex-a35 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A35

// REQUIRES: LINUX_ENABLED && ARM_ENABLED && !MACOS_HOST

// CHECK_DEFAULT: Triple: aarch64-unknown-linux-gnu
// CHECK_DEFAULT-NEXT: CPU: cortex-a57
// CHECK_DEFAULT-NEXT: Feature String: +aes,+crc,+crypto,+fp-armv8,+neon,+sha2

// CHECK_AARCH64: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64-NEXT: CPU: cortex-a57
// CHECK_AARCH64-NEXT: Feature String: +aes,+crc,+crypto,+fp-armv8,+neon,+sha2

// CHECK_AARCH64_A35: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_A35-NEXT: CPU: cortex-a35
// CHECK_AARCH64_A35-NEXT: Feature String: +aes,+crc,+crypto,+fp-armv8,+neon,+sha2

uniform int i;

void foo() {}
