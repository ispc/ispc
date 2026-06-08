// Verifies that the "avx10" entry in ISPC's CPUFeatures map lets an AVX10
// LLVM intrinsic be used from ISPC source on the diamondrapids (DMR) and
// novalake (NVL) CPUs. Without the "avx10" CPUFeatures entry for these CPUs,
// Target::checkIntrinsticSupport rejects @llvm.x86.avx10.* with
// "Target specific LLVM intrinsic ... not supported on ... CPU".
// The @llvm.x86.avx10.mask.vminmaxps256 intrinsic exists in LLVM 22 and
// lowers to the vminmaxps instruction, proving the intrinsic is both accepted
// (feature gate works) and selected by the backend.

// RUN: %{ispc} %s --target=avx10.2dmr-x16 --enable-llvm-intrinsics --nowrap --emit-asm -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx10.2nvl-x16 --enable-llvm-intrinsics --nowrap --emit-asm -o - | FileCheck %s

// REQUIRES: X86_ENABLED && LLVM_22_0+ && !MACOS_HOST

// CHECK-LABEL: f___
// CHECK: vminmaxps
export void f(uniform float a[8], uniform float b[8], uniform float passthru[8], uniform float out[8]) {
    uniform float<8> va = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
    uniform float<8> vb = {b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]};
    uniform float<8> vp = {passthru[0], passthru[1], passthru[2], passthru[3],
                           passthru[4], passthru[5], passthru[6], passthru[7]};
    uniform float<8> r = @llvm.x86.avx10.mask.vminmaxps256(va, vb, (uniform int)0, vp, (uniform int8)-1);
    for (uniform int i = 0; i < 8; i++)
        out[i] = r[i];
}
