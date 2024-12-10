// The test checks correct features for ARM64 for different cpus on macOS starting LLVM18. We don't check full set of features, just the ones used in lGetHostARMDeviceType

// RUN: %{ispc} %s -o %t.o --nostdlib --target=neon-i32x4 --nowrap --cpu=apple-a17 --print-target | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ARM64_APPLE_A17

// REQUIRES: ARM_ENABLED && MACOS_ARM_ENABLED && LLVM_18_0+

// CHECK_ALL: Triple: arm64-apple-macosx

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
