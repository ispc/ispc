;;
;; target-neon-common.ll
;;
;;  Copyright(c) 2013-2015 Google, Inc.
;;  Copyright(c) 2019-2024 Intel
;;
;;  SPDX-License-Identifier: BSD-3-Clause



define(`NEON_PREFIX',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon',
        RUNTIME, `32', `llvm.arm.neon')')

define(`NEON_PREFIX_FMIN',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.fmin',
        RUNTIME, `32', `llvm.arm.neon.vmins')')

define(`NEON_PREFIX_IMINS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.smin',
        RUNTIME, `32', `llvm.arm.neon.vmins')')

define(`NEON_PREFIX_IMINU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.umin',
        RUNTIME, `32', `llvm.arm.neon.vminu')')

define(`NEON_PREFIX_PMINF',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.fminp',
        RUNTIME, `32', `llvm.arm.neon.vpmins')')

define(`NEON_PREFIX_PMINS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.sminp',
        RUNTIME, `32', `llvm.arm.neon.vpmins')')

define(`NEON_PREFIX_PMINU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.uminp',
        RUNTIME, `32', `llvm.arm.neon.vpminu')')

define(`NEON_PREFIX_FMAX',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.fmax',
        RUNTIME, `32', `llvm.arm.neon.vmaxs')')

define(`NEON_PREFIX_IMAXS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.smax',
        RUNTIME, `32', `llvm.arm.neon.vmaxs')')

define(`NEON_PREFIX_IMAXU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.umax',
        RUNTIME, `32', `llvm.arm.neon.vmaxu')')

define(`NEON_PREFIX_PMAXF',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.fmaxp',
        RUNTIME, `32', `llvm.arm.neon.vpmaxs')')

define(`NEON_PREFIX_PMAXS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.smaxp',
        RUNTIME, `32', `llvm.arm.neon.vpmaxs')')

define(`NEON_PREFIX_PMAXU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.umaxp',
        RUNTIME, `32', `llvm.arm.neon.vpmaxu')')

define(`NEON_PREFIX_FPADD',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.addp',
        RUNTIME, `32', `llvm.arm.neon.vpadd')')

define(`NEON_PREFIX_PADDLS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.saddlp',
        RUNTIME, `32', `llvm.arm.neon.vpaddls')')

define(`NEON_PREFIX_PADDLU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.uaddlp',
        RUNTIME, `32', `llvm.arm.neon.vpaddlu')')

define(`NEON_PREFIX_RHADDLS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.srhadd',
        RUNTIME, `32', `llvm.arm.neon.vrhadds')')

define(`NEON_PREFIX_RHADDLU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.urhadd',
        RUNTIME, `32', `llvm.arm.neon.vrhaddu')')

define(`NEON_PREFIX_HADDLS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.shadd',
        RUNTIME, `32', `llvm.arm.neon.vhadds')')

define(`NEON_PREFIX_HADDLU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.uhadd',
        RUNTIME, `32', `llvm.arm.neon.vhaddu')')

define(`NEON_PREFIX_QADDS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.sqadd',
        RUNTIME, `32', `llvm.arm.neon.vqadds')')

define(`NEON_PREFIX_QADDU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.uqadd',
        RUNTIME, `32', `llvm.arm.neon.vqaddu')')

define(`NEON_PREFIX_QSUBS',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.sqsub',
        RUNTIME, `32', `llvm.arm.neon.vqsubs')')

define(`NEON_PREFIX_QSUBU',
`ifelse(RUNTIME, `64', `llvm.aarch64.neon.uqsub',
        RUNTIME, `32', `llvm.arm.neon.vqsubu')')

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

stdlib_core()
scans()
reduce_equal(WIDTH)
rdrand_decls()
define_shuffles()
aossoa()
ctlztz()
popcnt()
halfTypeGenericImplementation()
dot_product_vnni_decl()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare <4 x i16> @NEON_PREFIX.vcvtfp2hf(<4 x float>) nounwind readnone
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; math

ifelse(RUNTIME, `32',
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
;; round/floor/ceil

;; FIXME: grabbed these from the sse2 target, which does not have native
;; instructions for these.  Is there a better approach for NEON?

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast float %0 to i32
  %bitop.i.i = and i32 %float_to_int_bitcast.i.i.i.i, -2147483648
  %bitop.i = xor i32 %bitop.i.i, %float_to_int_bitcast.i.i.i.i
  %int_to_float_bitcast.i.i40.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %int_to_float_bitcast.i.i40.i, 8.388608e+06
  %binop21.i = fadd float %binop.i, -8.388608e+06
  %float_to_int_bitcast.i.i.i = bitcast float %binop21.i to i32
  %bitop31.i = xor i32 %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop31.i to float
  ret float %int_to_float_bitcast.i.i.i
}

define float @__floor_uniform_float(float) nounwind readonly alwaysinline {
  %calltmp.i = tail call float @__round_uniform_float(float %0) nounwind
  %bincmp.i = fcmp ogt float %calltmp.i, %0
  %selectexpr.i = sext i1 %bincmp.i to i32
  %bitop.i = and i32 %selectexpr.i, -1082130432
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %calltmp.i, %int_to_float_bitcast.i.i.i
  ret float %binop.i
}

define float @__ceil_uniform_float(float) nounwind readonly alwaysinline {
  %calltmp.i = tail call float @__round_uniform_float(float %0) nounwind
  %bincmp.i = fcmp olt float %calltmp.i, %0
  %selectexpr.i = sext i1 %bincmp.i to i32
  %bitop.i = and i32 %selectexpr.i, 1065353216
  %int_to_float_bitcast.i.i.i = bitcast i32 %bitop.i to float
  %binop.i = fadd float %calltmp.i, %int_to_float_bitcast.i.i.i
  ret float %binop.i
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare double @round(double)
declare double @floor(double)
declare double @ceil(double)

define double @__round_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @round(double %0)
  ret double %r
}

define double @__floor_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @floor(double %0)
  ret double %r
}

define double @__ceil_uniform_double(double) nounwind readonly alwaysinline {
  %r = call double @ceil(double %0)
  ret double %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

define float @__max_uniform_float(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ugt float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define float @__min_uniform_float(float, float) nounwind readnone alwaysinline {
  %cmp = fcmp ult float %0, %1
  %r = select i1 %cmp, float %0, float %1
  ret float %r
}

define i32 @__min_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp slt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__max_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp sgt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__min_uniform_uint32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ult i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readnone alwaysinline {
  %cmp = icmp ugt i32 %0, %1
  %r = select i1 %cmp, i32 %0, i32 %1
  ret i32 %r
}

define i64 @__min_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp slt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp sgt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__min_uniform_uint64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp ult i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define i64 @__max_uniform_uint64(i64, i64) nounwind readnone alwaysinline {
  %cmp = icmp ugt i64 %0, %1
  %r = select i1 %cmp, i64 %0, i64 %1
  ret i64 %r
}

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp olt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %0, %1
  %r = select i1 %cmp, double %0, double %1
  ret double %r
}

define <WIDTH x i64> @__min_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp slt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp sgt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__min_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp ult <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x i64> @__max_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone alwaysinline {
  %m = icmp ugt <WIDTH x i64> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x i64> %0, <WIDTH x i64> %1
  ret <WIDTH x i64> %r
}

define <WIDTH x double> @__min_varying_double(<WIDTH x double>,
                                              <WIDTH x double>) nounwind readnone alwaysinline {
  %m = fcmp olt <WIDTH x double> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %r
}

define <WIDTH x double> @__max_varying_double(<WIDTH x double>,
                                              <WIDTH x double>) nounwind readnone alwaysinline {
  %m = fcmp ogt <WIDTH x double> %0, %1
  %r = select <WIDTH x i1> %m, <WIDTH x double> %0, <WIDTH x double> %1
  ret <WIDTH x double> %r
}

;; sqrt/rsqrt/rcp

declare float @llvm.sqrt.f32(float)

define float @__sqrt_uniform_float(float) nounwind readnone alwaysinline {
  %r = call float @llvm.sqrt.f32(float %0)
  ret float %r
}

declare double @llvm.sqrt.f64(double)

define double @__sqrt_uniform_double(double) nounwind readnone alwaysinline {
  %r = call double @llvm.sqrt.f64(double %0)
  ret double %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

masked_load(i8,  1)
masked_load(i16, 2)
masked_load(half, 2)
masked_load(i32, 4)
masked_load(float, 4)
masked_load(i64, 8)
masked_load(double, 8)

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(i64)
masked_store_float_double()

define void @__masked_store_blend_i8(<WIDTH x i8>* nocapture %ptr, <WIDTH x i8> %new,
                                     <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i8> ')  %ptr
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %result = select <WIDTH x i1> %mask1, <WIDTH x i8> %new, <WIDTH x i8> %old
  store <WIDTH x i8> %result, <WIDTH x i8> * %ptr
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture %ptr, <WIDTH x i16> %new, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i16> ')  %ptr
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %result = select <WIDTH x i1> %mask1, <WIDTH x i16> %new, <WIDTH x i16> %old
  store <WIDTH x i16> %result, <WIDTH x i16> * %ptr
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture %ptr, <WIDTH x i32> %new, 
                                      <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i32> ')  %ptr
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %result = select <WIDTH x i1> %mask1, <WIDTH x i32> %new, <WIDTH x i32> %old
  store <WIDTH x i32> %result, <WIDTH x i32> * %ptr
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture %ptr,
                            <WIDTH x i64> %new, <WIDTH x MASK> %mask) nounwind alwaysinline {
  %old = load PTR_OP_ARGS(`<WIDTH x i64> ')  %ptr
  %mask1 = trunc <WIDTH x MASK> %mask to <WIDTH x i1>
  %result = select <WIDTH x i1> %mask1, <WIDTH x i64> %new, <WIDTH x i64> %old
  store <WIDTH x i64> %result, <WIDTH x i64> * %ptr
  ret void
}

;; yuck.  We need declarations of these, even though we shouldnt ever
;; actually generate calls to them for the NEON target...


include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(i32)
gen_gather_factored(half)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(half)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)

packed_load_and_store(FALSE)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

define_prefetches()
