;;
;; target-neon-common.ll
;;
;;  Copyright(c) 2013-2015 Google, Inc.
;;  Copyright(c) 2019-2025 Intel
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
