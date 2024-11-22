// The test checks correct features for diferent linux cpus/archs for custom_linux.

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --print-target | FileCheck %s -check-prefix=CHECK_DEFAULT
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=aarch64 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=aarch64 --cpu=cortex-a15 --print-target 2>&1 | FileCheck %s -check-prefix=CHECK_AARCH64_A15
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=aarch64 --cpu=cortex-a35 --print-target | FileCheck %s -check-prefix=CHECK_AARCH64_A35
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=arm --print-target | FileCheck %s -check-prefix=CHECK_ARM
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=arm --cpu=cortex-a15 --print-target | FileCheck %s -check-prefix=CHECK_ARM_A15
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --target-os=custom_linux --arch=arm --cpu=cortex-a35 --print-target | FileCheck %s -check-prefix=CHECK_ARM_A35

// REQUIRES: ARM_ENABLED && !MACOS_HOST

// CHECK_DEFAULT: Triple: aarch64-unknown-linux-gnu
// CHECK_DEFAULT-NEXT: CPU: cortex-a57
// CHECK_DEFAULT-NEXT: Feature String: +aes,+crc,+crypto,+fp-armv8,+neon,+sha2

// CHECK_AARCH64: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64-NEXT: CPU: cortex-a57
// CHECK_AARCH64-NEXT: Feature String: +aes,+crc,+crypto,+fp-armv8,+neon,+sha2

// CHECK_AARCH64_A15: 'cortex-a15' is not a recognized processor for this target (ignoring processor)

// CHECK_AARCH64_A35: Triple: aarch64-unknown-linux-gnu
// CHECK_AARCH64_A35-NEXT: CPU: cortex-a35
// CHECK_AARCH64_A35-NEXT: Feature String: +aes,+crc,+crypto,+fp-armv8,+neon,+sha2

// CHECK_ARM: Triple: armv7-unknown-linux-gnueabihf
// CHECK_ARM-NEXT: CPU: cortex-a57
// CHECK_ARM-NEXT: Feature String: +crypto,+fp-armv8,+neon,+sha2

// CHECK_ARM_A15: Triple: armv7-unknown-linux-gnueabihf
// CHECK_ARM_A15-NEXT: CPU: cortex-a15
// CHECK_ARM_A15-NEXT: Feature String: +crypto,+fp-armv8,+neon,+sha2

// CHECK_ARM_A35: Triple: armv7-unknown-linux-gnueabihf
// CHECK_ARM_A35-NEXT: CPU: cortex-a35
// CHECK_ARM_A35-NEXT: Feature String: +crypto,+fp-armv8,+neon,+sha2

uniform int i;

void foo() {}