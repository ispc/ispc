// RUN: %{ispc} %s --target=avx512spr-x4 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SPR
// RUN: %{ispc} %s --target=avx512spr-x8 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SPR

// -x16 target has fails for half -> int16/uint16 converts with LLVM 14. For 15&16 backport of
// the fix is available, and it's part of upstream in 17.0. So x16 is disabled for now.
// %{ispc} %s --target=avx512spr-x16 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SPR

// RUN: %{ispc} %s --target=avx512spr-x32 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SPR
// RUN: %{ispc} %s --target=avx512spr-x64 --emit-asm -o - | FileCheck %s -check-prefix=CHECK_SPR

// REQUIRES: X86_ENABLED && LLVM_14_0+ && !MACOS_HOST

// CHECK_SPR-LABEL: fp16_add
// CHECK_SPR: vaddph
unmasked float16 fp16_add(float16 a, float16 b) {
  return a+b;
}

// CHECK_SPR-LABEL: fp16_sub
// CHECK_SPR: vsubph
unmasked float16 fp16_sub(float16 a, float16 b) {
  return a-b;
}

// CHECK_SPR-LABEL: fp16_mul
// CHECK_SPR: vmulph
unmasked float16 fp16_mul(float16 a, float16 b) {
  return a*b;
}

// CHECK_SPR-LABEL: fp16_div
// CHECK_SPR: vdivph
unmasked float16 fp16_div(float16 a, float16 b) {
  return a/b;
}

// CHECK_SPR-LABEL: fp16_sqrt
// CHECK_SPR: vsqrtph
unmasked float16 fp16_sqrt(float16 a) {
  return sqrt(a);
}

// CHECK_SPR-LABEL: fp16_rsqrt
// CHECK_SPR: vrsqrtph
unmasked float16 fp16_rsqrt(float16 a) {
  return rsqrt(a);
}

// CHECK_SPR-LABEL: fp16_rcp
// CHECK_SPR: vrcpph
unmasked float16 fp16_rcp(float16 a) {
  return rcp(a);
}

// CHECK_SPR-LABEL: fp16_min
// CHECK_SPR: vminph
unmasked float16 fp16_min(float16 a, float16 b) {
  return min(a, b);
}

// CHECK_SPR-LABEL: fp16_max
// CHECK_SPR: vmaxph
unmasked float16 fp16_max(float16 a, float16 b) {
  return max(a, b);
}

// CHECK_SPR-LABEL: fp16_cmp
// CHECK_SPR: vcmp{{lt|gt}}ph
unmasked bool fp16_cmp(float16 a, float16 b) {
  return a>b;
}

// CHECK_SPR-LABEL: fp16_fma
// CHECK_SPR: vfmadd213ph
unmasked float16 fp16_fma(float16 a, float16 b, float16 c) {
  return a*b+c;
}

// CHECK_SPR-LABEL: fp16_fnma
// CHECK_SPR: vfnmadd213ph
unmasked float16 fp16_fnma(float16 a, float16 b, float16 c) {
  return -a*b+c;
}

// CHECK_SPR-LABEL: fp16_fms
// CHECK_SPR: vfmsub213ph
unmasked float16 fp16_fms(float16 a, float16 b, float16 c) {
  return a*b-c;
}

// CHECK_SPR-LABEL: fp16_fnms
// CHECK_SPR: vfnmsub213ph
unmasked float16 fp16_fnms(float16 a, float16 b, float16 c) {
  return -a*b-c;
}

// CHECK_SPR-LABEL: fp16_round
// CHECK_SPR: vrndscalesh $8
unmasked float16 fp16_round(float16 a) {
  return round(a);
}

// CHECK_SPR-LABEL: fp16_floor
// CHECK_SPR: vrndscaleph $9
unmasked float16 fp16_floor(float16 a) {
  return floor(a);
}

// CHECK_SPR-LABEL: fp16_ceil
// CHECK_SPR: vrndscaleph $10
unmasked float16 fp16_ceil(float16 a) {
  return ceil(a);
}

// CHECK_SPR-LABEL: fp16_trunc
// CHECK_SPR: vrndscaleph $11
unmasked float16 fp16_trunc(float16 a) {
  return trunc(a);
}

///////////////////////////////////////////////
// Converts - floating point

// double

// CHECK_SPR-LABEL: fp16_cvt_pd2ph
// CHECK_SPR: vcvtpd2ph
unmasked float16 fp16_cvt_pd2ph(double a) {
  return a;
}

// CHECK_SPR-LABEL: fp16_cvt_ph2pd
// CHECK_SPR: vcvtph2pd
unmasked double fp16_cvt_ph2pd(float16 a) {
  return a;
}

// float

// CHECK_SPR-LABEL: fp16_cvt_ps2ph
// CHECK_SPR: vcvtps2ph
unmasked float16 fp16_cvt_ps2ph(float a) {
  return a;
}

// CHECK_SPR-LABEL: fp16_cvt_ph2ps
// CHECK_SPR: vcvtph2ps
unmasked float fp16_cvt_ph2ps(float16 a) {
  return a;
}

// int64

// CHECK_SPR-LABEL: fp16_cvt_qq2ph
// CHECK_SPR: vcvtqq2ph
unmasked float16 fp16_cvt_qq2ph(int64 a) {
  return a;
}

// CHECK_SPR-LABEL: fp16_cvt_ph2qq
// CHECK_SPR: vcvttph2qq
unmasked int64 fp16_cvt_ph2qq(float16 a) {
  return a;
}

// uint64

// CHECK_SPR-LABEL: fp16_cvt_uqq2ph
// CHECK_SPR: vcvtuqq2ph
unmasked float16 fp16_cvt_uqq2ph(uint64 a) {
  return a;
}

// CHECK_SPR-LABEL: fp16_cvt_ph2uqq
// CHECK_SPR: vcvttph2uqq
unmasked uint64 fp16_cvt_ph2uqq(float16 a) {
  return a;
}

// int32

// CHECK_SPR-LABEL: fp16_cvt_dq2ph
// CHECK_SPR: vcvtdq2ph
unmasked float16 fp16_cvt_dq2ph(int32 a) {
  return a;
}

// CHECK_SPR-LABEL: fp16_cvt_ph2dq
// CHECK_SPR: vcvttph2dq
unmasked int fp16_cvt_ph2dq(float16 a) {
  return a;
}

// uint32

// CHECK_SPR-LABEL: fp16_cvt_udq2ph
// CHECK_SPR: vcvtudq2ph
unmasked float16 fp16_cvt_udq2ph(uint32 a) {
  return a;
}

// CHECK_SPR-LABEL: fp16_cvt_ph2udq
// CHECK_SPR: vcvttph2udq
unmasked uint32 fp16_cvt_ph2udq(float16 a) {
  return a;
}

// int16

// CHECK_SPR-LABEL: fp16_cvt_w2ph
// CHECK_SPR: vcvtw2ph
unmasked float16 fp16_cvt_w2ph(int16 a) {
  return a;
}

// CHECK_SPR-LABEL: fp16_cvt_ph2w
// CHECK_SPR: vcvttph2w
unmasked int16 fp16_cvt_ph2w(float16 a) {
  return a;
}

// uint16

// CHECK_SPR-LABEL: fp16_cvt_uw2ph
// CHECK_SPR: vcvtuw2ph
unmasked float16 fp16_cvt_uw2ph(uint16 a) {
  return a;
}

// CHECK_SPR-LABEL: fp16_cvt_ph2uw
// CHECK_SPR: vcvttph2uw
unmasked uint16 fp16_cvt_ph2uw(float16 a) {
  return a;
}

