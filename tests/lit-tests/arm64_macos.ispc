// The test checks correct features for ARM64 for different cpus on macOS. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a7 --print-target  | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A7
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a10 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A10
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a11 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A11
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a12 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A12
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a13 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A13
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a14 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A14
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a15 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A15
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a16 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A16
// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a17 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A17

// REQUIRES: ARM_ENABLED && MACOS_ARM_ENABLED

// CHECK_ALL: Triple: arm64-apple-macosx

// CHECK_ARM64_APPLE_A7: CPU: apple-a7
// CHECK_ARM64_APPLE_A7-NEXT: Feature String
// CHECK_ARM64_APPLE_A7-SAME: +aes
// CHECK_ARM64_APPLE_A7-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A7-SAME: +neon
// CHECK_ARM64_APPLE_A7-SAME: +sha2

// CHECK_ARM64_APPLE_A10: CPU: apple-a10
// CHECK_ARM64_APPLE_A10-NEXT: Feature String
// CHECK_ARM64_APPLE_A10-SAME: +aes
// CHECK_ARM64_APPLE_A10-SAME: +crc
// CHECK_ARM64_APPLE_A10-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A10-SAME: +neon
// CHECK_ARM64_APPLE_A10-SAME: +sha2

// CHECK_ARM64_APPLE_A11: CPU: apple-a11
// CHECK_ARM64_APPLE_A11-NEXT: Feature String
// CHECK_ARM64_APPLE_A11-SAME: +aes
// CHECK_ARM64_APPLE_A11-SAME: +crc
// CHECK_ARM64_APPLE_A11-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A11-SAME: +fullfp16
// CHECK_ARM64_APPLE_A11-SAME: +lse
// CHECK_ARM64_APPLE_A11-SAME: +neon
// CHECK_ARM64_APPLE_A11-SAME: +sha2

// CHECK_ARM64_APPLE_A12: CPU: apple-a12
// CHECK_ARM64_APPLE_A12-NEXT: Feature String
// CHECK_ARM64_APPLE_A12-SAME: +aes
// CHECK_ARM64_APPLE_A12-SAME: +crc
// CHECK_ARM64_APPLE_A12-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A12-SAME: +fullfp16
// CHECK_ARM64_APPLE_A12-SAME: +lse
// CHECK_ARM64_APPLE_A12-SAME: +neon
// CHECK_ARM64_APPLE_A12-SAME: +rcpc
// CHECK_ARM64_APPLE_A12-SAME: +sha2

// CHECK_ARM64_APPLE_A13: CPU: apple-a13
// CHECK_ARM64_APPLE_A13-NEXT: Feature String
// CHECK_ARM64_APPLE_A13-SAME: +aes
// CHECK_ARM64_APPLE_A13-SAME: +crc
// CHECK_ARM64_APPLE_A13-SAME: +dotprod
// CHECK_ARM64_APPLE_A13-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A13-SAME: +fp16fml
// CHECK_ARM64_APPLE_A13-SAME: +fullfp16
// CHECK_ARM64_APPLE_A13-SAME: +lse
// CHECK_ARM64_APPLE_A13-SAME: +neon
// CHECK_ARM64_APPLE_A13-SAME: +rcpc
// CHECK_ARM64_APPLE_A13-SAME: +sha2
// CHECK_ARM64_APPLE_A13-SAME: +sha3

// CHECK_ARM64_APPLE_A14: CPU: apple-a14
// CHECK_ARM64_APPLE_A14-NEXT: Feature String
// CHECK_ARM64_APPLE_A14-SAME: +aes
// CHECK_ARM64_APPLE_A14-SAME: +crc
// CHECK_ARM64_APPLE_A14-SAME: +dotprod
// CHECK_ARM64_APPLE_A14-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A14-SAME: +fp16fml
// CHECK_ARM64_APPLE_A14-SAME: +fullfp16
// CHECK_ARM64_APPLE_A14-SAME: +lse
// CHECK_ARM64_APPLE_A14-SAME: +neon
// CHECK_ARM64_APPLE_A14-SAME: +rcpc
// CHECK_ARM64_APPLE_A14-SAME: +sha2
// CHECK_ARM64_APPLE_A14-SAME: +sha3

// CHECK_ARM64_APPLE_A15: CPU: apple-a15
// CHECK_ARM64_APPLE_A15-NEXT: Feature String
// CHECK_ARM64_APPLE_A15-SAME: +aes
// CHECK_ARM64_APPLE_A15-SAME: +bf16
// CHECK_ARM64_APPLE_A15-SAME: +crc
// CHECK_ARM64_APPLE_A15-SAME: +dotprod
// CHECK_ARM64_APPLE_A15-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A15-SAME: +fp16fml
// CHECK_ARM64_APPLE_A15-SAME: +fullfp16
// CHECK_ARM64_APPLE_A15-SAME: +i8mm
// CHECK_ARM64_APPLE_A15-SAME: +lse
// CHECK_ARM64_APPLE_A15-SAME: +neon
// CHECK_ARM64_APPLE_A15-SAME: +rcpc
// CHECK_ARM64_APPLE_A15-SAME: +sha2
// CHECK_ARM64_APPLE_A15-SAME: +sha3

// CHECK_ARM64_APPLE_A16: CPU: apple-a16
// CHECK_ARM64_APPLE_A16-NEXT: Feature String
// CHECK_ARM64_APPLE_A16-SAME: +aes
// CHECK_ARM64_APPLE_A16-SAME: +bf16
// CHECK_ARM64_APPLE_A16-SAME: +crc
// CHECK_ARM64_APPLE_A16-SAME: +dotprod
// CHECK_ARM64_APPLE_A16-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A16-SAME: +fp16fml
// CHECK_ARM64_APPLE_A16-SAME: +fullfp16
// CHECK_ARM64_APPLE_A16-SAME: +i8mm
// CHECK_ARM64_APPLE_A16-SAME: +lse
// CHECK_ARM64_APPLE_A16-SAME: +neon
// CHECK_ARM64_APPLE_A16-SAME: +rcpc
// CHECK_ARM64_APPLE_A16-SAME: +sha2
// CHECK_ARM64_APPLE_A16-SAME: +sha3

// CHECK_ARM64_APPLE_A17: CPU: apple-a17
// CHECK_ARM64_APPLE_A17-NEXT: Feature String
// CHECK_ARM64_APPLE_A17-SAME: +aes
// CHECK_ARM64_APPLE_A17-SAME: +bf16
// CHECK_ARM64_APPLE_A17-SAME: +crc
// CHECK_ARM64_APPLE_A17-SAME: +dotprod
// CHECK_ARM64_APPLE_A17-SAME: +fp-armv8
// CHECK_ARM64_APPLE_A17-SAME: +fp16fml
// CHECK_ARM64_APPLE_A17-SAME: +fullfp16
// CHECK_ARM64_APPLE_A17-SAME: +i8mm
// CHECK_ARM64_APPLE_A17-SAME: +lse
// CHECK_ARM64_APPLE_A17-SAME: +neon
// CHECK_ARM64_APPLE_A17-SAME: +rcpc
// CHECK_ARM64_APPLE_A17-SAME: +sha2
// CHECK_ARM64_APPLE_A17-SAME: +sha3


uniform int i;

void foo() {}
