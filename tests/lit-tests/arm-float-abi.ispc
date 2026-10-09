// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --target-os=linux --nostdlib --emit-asm -o - | FileCheck %s --check-prefix=ARM
// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --target-os=linux --nostdlib --opt=disable-fma --emit-asm -o - | FileCheck %s --check-prefix=ARM-NO-FMA --implicit-check-not=vfma
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --nostdlib --emit-asm -o - | FileCheck %s --check-prefix=AARCH64
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --nostdlib --opt=disable-fma --emit-asm -o - | FileCheck %s --check-prefix=AARCH64-NO-FMA --implicit-check-not=fmadd
// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --target-os=linux --nostdlib --emit-llvm-text -o - | FileCheck %s --check-prefix=IR --implicit-check-not='"float-abi"' --implicit-check-not=contract
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --target-os=linux --nostdlib --emit-llvm-text -o - | FileCheck %s --check-prefix=IR --implicit-check-not='"float-abi"' --implicit-check-not=contract
// REQUIRES: ARM32_ENABLED && LINUX_HOST

// Exported uniform floats must use floating-point registers for arguments
// and the return value. LLVM 24 requires a "float-abi" module flag for this,
// which must be added only during code generation, just like FMA contraction.
// ARM-LABEL: multiply_add:
// ARM: vfma.f32 s0, s0, s1
// ARM-NO-FMA-LABEL: multiply_add:
// ARM-NO-FMA: vmla.f32 s0, s0, s1

// AARCH64-LABEL: multiply_add:
// AARCH64: fmadd s0, s0, s1, s0
// AARCH64-NO-FMA-LABEL: multiply_add:
// AARCH64-NO-FMA: fmul s1, s0, s1
// AARCH64-NO-FMA: fadd s0, s0, s1

// IR-LABEL: define float @multiply_add(
// IR: fmul float
// IR: fadd float
export uniform float multiply_add(uniform float a, uniform float b) {
    return a * b + a;
}
