// Tests checks that correct registries are used to set/restore FTZ/DAZ flags on ARM

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --nostdlib --emit-asm --opt=reset-ftz-daz -o - | FileCheck --check-prefixes=CHECK_AARCH64 %s

// CHECK_AARCH64: mrs
// CHECK_AARCH64: FPCR

// ARM must be enabled in order to test it.
// REQUIRES: ARM_ENABLED
// Not every ISPC build on macOS supports ARM targets.
// They are supported only if macOS SDK supports them.
// REQUIRES: !MACOS_HOST || MACOS_ARM_ENABLED

export uniform float export_test_ftz_daz(){
    uniform float x = 0x1p-149;
    uniform float y = 0x0.fffffep-126f;
    return y - x;
}
