// Tests checks that correct registries are used to set/restore FTZ/DAZ flags on ARM

// RUN: %{ispc} %s --arch=arm --target=neon-i32x8 --nostdlib --emit-asm --opt=reset-ftz-daz -o - | FileCheck --check-prefixes=CHECK_ARM %s

// CHECK_ARM: vmrs
// CHECK_ARM: fpscr

// ARM must be enabled in order to test it.
// Windows and Mac do not support 32 bit ARM. so test on Linux only (Android assumed to be the same)
// REQUIRES: ARM_ENABLED && LINUX_HOST

export uniform float export_test_ftz_daz(){
    uniform float x = 0x1p-149;
    uniform float y = 0x0.fffffep-126f;
    return y - x;
}
