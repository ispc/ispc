;;
;; target-neon-common.ll
;;
;;  Copyright(c) 2013-2015 Google, Inc.
;;  Copyright(c) 2019-2026 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause


declare i1 @__is_compile_time_constant_mask(<WIDTH x MASK> %mask)

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
declare i64 @llvm.ctpop.i64(i64) nounwind readnone

define(`NEON_PREFIX',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon',
        RUNTIME, `32', `llvm.arm.neon')')

define(`NEON_PREFIX_RECPEQ',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.frecpe',
        RUNTIME, `32', `llvm.arm.neon.vrecpe')')

define(`NEON_PREFIX_RECPSQ',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.frecps',
        RUNTIME, `32', `llvm.arm.neon.vrecps')')

define(`NEON_PREFIX_RSQRTEQ',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.frsqrte',
        RUNTIME, `32', `llvm.arm.neon.vrsqrte')')

define(`NEON_PREFIX_RSQRTSQ',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.frsqrts',
        RUNTIME, `32', `llvm.arm.neon.vrsqrts')')

define(`NEON_PREFIX_UDOT',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.udot',
        RUNTIME, `32', `llvm.arm.neon.udot')')

define(`NEON_PREFIX_SDOT',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.sdot',
        RUNTIME, `32', `llvm.arm.neon.sdot')')

define(`NEON_PREFIX_USDOT',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.usdot',
        RUNTIME, `32', `llvm.arm.neon.usdot')')

define(`NEON_PREFIX_PADDLU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.uaddlp',
        RUNTIME, `32', `llvm.arm.neon.vpaddlu')')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

define(`half_uniform_conversions',
`declare <4 x i16> @NEON_PREFIX.vcvtfp2hf(<4 x float>) nounwind readnone
declare <4 x float> @NEON_PREFIX.vcvthf2fp(<4 x i16>) nounwind readnone

define float @__half_to_float_uniform(i16 %v) nounwind readnone alwaysinline {
  %v1 = bitcast i16 %v to <1 x i16>
  %vec = shufflevector <1 x i16> %v1, <1 x i16> undef,
           <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %h = call <4 x float> @NEON_PREFIX.vcvthf2fp(<4 x i16> %vec)
  %r = extractelement <4 x float> %h, i32 0
  ret float %r
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone alwaysinline {
  %v1 = bitcast float %v to <1 x float>
  %vec = shufflevector <1 x float> %v1, <1 x float> undef,
           <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  %h = call <4 x i16> @NEON_PREFIX.vcvtfp2hf(<4 x float> %vec)
  %r = extractelement <4 x i16> %h, i32 0
  ret i16 %r
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; math

define(`math_flags_functions',
`ifelse(RUNTIME, `32',
`
  declare void @llvm.arm.set.fpscr(i32) nounwind
  declare i32 @llvm.arm.get.fpscr() nounwind

  define void @__fastmath() nounwind alwaysinline {
    %x = call i32 @llvm.arm.get.fpscr()
    ; Turn on FTZ (bit 24) and default NaN (bit 25)
    %y = or i32 %x, 50331648
    call void @llvm.arm.set.fpscr(i32 %y)
    ret void
  }

  define i32 @__set_ftz_daz_flags() nounwind alwaysinline {
    %x = call i32 @llvm.arm.get.fpscr()
    ; Turn on FTZ (bit 24) and default NaN (bit 25)
    %y = or i32 %x, 50331648
    call void @llvm.arm.set.fpscr(i32 %y)
    ret i32 %x
  }

  define void @__restore_ftz_daz_flags(i32 %oldVal) nounwind alwaysinline {
    ; restore value to previously saved
    call void @llvm.arm.set.fpscr(i32 %oldVal)
    ret void
  }
',
  RUNTIME, `64',
`
  declare void @llvm.aarch64.set.fpcr(i64) nounwind
  declare i64 @llvm.aarch64.get.fpcr() nounwind

  define void @__fastmath() nounwind alwaysinline {
    %x = call i64 @llvm.aarch64.get.fpcr()
    ; Turn on FTZ (bit 24) and default NaN (bit 25)
    %y = or i64 %x, 50331648
    call void @llvm.aarch64.set.fpcr(i64 %y)
    ret void
  }

  define i64 @__set_ftz_daz_flags() nounwind alwaysinline {
    %x = call i64 @llvm.aarch64.get.fpcr()
    ; Turn on FTZ (bit 24) and default NaN (bit 25)
    %y = or i64 %x, 50331648
    call void @llvm.aarch64.set.fpcr(i64 %y)
    ret i64 %x
  }

  define void @__restore_ftz_daz_flags(i64 %oldVal) nounwind alwaysinline {
    ; restore value to previously saved
    call void @llvm.aarch64.set.fpcr(i64 %oldVal)
    ret void
  }
')
')

aossoa()
packed_load_and_store(FALSE)
math_flags_functions()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

declare void @__masked_store_blend_i8(<WIDTH x i8>* nocapture %ptr, <WIDTH x i8> %new,
                                     <WIDTH x MASK> %mask) nounwind alwaysinline

declare void @__masked_store_blend_i16(<WIDTH x i16>* nocapture %ptr, <WIDTH x i16> %new,
                                      <WIDTH x MASK> %mask) nounwind alwaysinline

declare void @__masked_store_blend_i32(<WIDTH x i32>* nocapture %ptr, <WIDTH x i32> %new,
                                      <WIDTH x MASK> %mask) nounwind alwaysinline

declare void @__masked_store_blend_i64(<WIDTH x i64>* nocapture %ptr,
                            <WIDTH x i64> %new, <WIDTH x MASK> %mask) nounwind alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(i32)
gen_gather_factored(half)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; assert

declare i32 @puts(i8*)
declare void @abort() noreturn

define void @__do_assert_uniform(i8 *%str, i1 %test, <WIDTH x MASK> %mask) {
  br i1 %test, label %ok, label %fail

fail:
  %call = call i32 @puts(i8* %str)
  call void @abort() noreturn
  unreachable

ok:
  ret void
}


define void @__do_assert_varying(i8 *%str, <WIDTH x MASK> %test,
                                 <WIDTH x MASK> %mask) {
  %nottest = xor <WIDTH x MASK> %test,
                 < forloop(i, 1, eval(WIDTH-1), `MASK -1, ') MASK -1 >
  %nottest_and_mask = and <WIDTH x MASK> %nottest, %mask
  %mm = call i64 @__movmsk(<WIDTH x MASK> %nottest_and_mask)
  %all_ok = icmp eq i64 %mm, 0
  br i1 %all_ok, label %ok, label %fail

fail:
  %call = call i32 @puts(i8* %str)
  call void @abort() noreturn
  unreachable

ok:
  ret void
}