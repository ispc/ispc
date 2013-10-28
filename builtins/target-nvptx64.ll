;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Define the standard library builtins for the NOVEC target
define(`MASK',`i1')
define(`WIDTH',`1')

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() nounwind readnone

define i32 @__tid_x()  nounwind readnone alwaysinline
{
 %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 ret i32 %tid
}
define i32 @__warpsize()  nounwind readnone alwaysinline
{
 %tid = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
 ret i32 %tid
}


define i32 @__ctaid_x()  nounwind readnone alwaysinline
{
 %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 ret i32 %bid
}
define i32 @__ctaid_y()  nounwind readnone alwaysinline
{
 %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
 ret i32 %bid
}
define i32 @__ctaid_z()  nounwind readnone alwaysinline
{
 %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
 ret i32 %bid
}

define i32 @__nctaid_x()  nounwind readnone alwaysinline
{
 %nb = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
 ret i32 %nb
}
define i32 @__nctaid_y()  nounwind readnone alwaysinline
{
 %nb = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
 ret i32 %nb
}
define i32 @__nctaid_z()  nounwind readnone alwaysinline
{
 %nb = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
 ret i32 %nb
}

;;;;;;;;;;;;;;


include(`util.m4')

stdlib_core()
packed_load_and_store()
int64minmax()
scans()
rdrand_decls()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shuffle

define_shuffles()

;; declare <WIDTH x float> @__smear_float(float) nounwind readnone
;; declare <WIDTH x double> @__smear_double(double) nounwind readnone
;; declare <WIDTH x i8> @__smear_i8(i8) nounwind readnone
;; declare <WIDTH x i16> @__smear_i16(i16) nounwind readnone
;; declare <WIDTH x i32> @__smear_i32(i32) nounwind readnone
;; declare <WIDTH x i64> @__smear_i64(i64) nounwind readnone

;; declare <WIDTH x float> @__setzero_float() nounwind readnone
;; declare <WIDTH x double> @__setzero_double() nounwind readnone
;; declare <WIDTH x i8> @__setzero_i8() nounwind readnone
;; declare <WIDTH x i16> @__setzero_i16() nounwind readnone
;; declare <WIDTH x i32> @__setzero_i32() nounwind readnone
;; declare <WIDTH x i64> @__setzero_i64() nounwind readnone

;; declare <WIDTH x float> @__undef_float() nounwind readnone
;; declare <WIDTH x double> @__undef_double() nounwind readnone
;; declare <WIDTH x i8> @__undef_i8() nounwind readnone
;; declare <WIDTH x i16> @__undef_i16() nounwind readnone
;; declare <WIDTH x i32> @__undef_i32() nounwind readnone
;; declare <WIDTH x i64> @__undef_i64() nounwind readnone


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; aos/soa

aossoa()

;; dummy 1 wide vector ops
define  void
@__aos_to_soa4_float1(<1 x float> %v0, <1 x float> %v1, <1 x float> %v2,
        <1 x float> %v3, <1 x float> * noalias %out0, 
        <1 x float> * noalias %out1, <1 x float> * noalias %out2, 
        <1 x float> * noalias %out3) nounwind alwaysinline { 

  store <1 x float> %v0, <1 x float > * %out0
  store <1 x float> %v1, <1 x float > * %out1
  store <1 x float> %v2, <1 x float > * %out2
  store <1 x float> %v3, <1 x float > * %out3

  ret void
}

define  void
@__soa_to_aos4_float1(<1 x float> %v0, <1 x float> %v1, <1 x float> %v2,
        <1 x float> %v3, <1 x float> * noalias %out0, 
        <1 x float> * noalias %out1, <1 x float> * noalias %out2, 
        <1 x float> * noalias %out3) nounwind alwaysinline { 
  call void @__aos_to_soa4_float1(<1 x float> %v0, <1 x float> %v1, 
    <1 x float> %v2, <1 x float> %v3, <1 x float> * %out0, 
    <1 x float> * %out1, <1 x float> * %out2, <1 x float> * %out3)
  ret void
}

define  void
@__aos_to_soa3_float1(<1 x float> %v0, <1 x float> %v1,
         <1 x float> %v2, <1 x float> * %out0, <1 x float> * %out1,
         <1 x float> * %out2) {
  store <1 x float> %v0, <1 x float > * %out0
  store <1 x float> %v1, <1 x float > * %out1
  store <1 x float> %v2, <1 x float > * %out2

  ret void
}

define  void
@__soa_to_aos3_float1(<1 x float> %v0, <1 x float> %v1,
         <1 x float> %v2, <1 x float> * %out0, <1 x float> * %out1,
         <1 x float> * %out2) {
  call void @__aos_to_soa3_float1(<1 x float> %v0, <1 x float> %v1,
         <1 x float> %v2, <1 x float> * %out0, <1 x float> * %out1,
         <1 x float> * %out2)
  ret void
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare float @__half_to_float_uniform(i16 %v) nounwind readnone
declare <WIDTH x float> @__half_to_float_varying(<WIDTH x i16> %v) nounwind readnone
declare i16 @__float_to_half_uniform(float %v) nounwind readnone
declare <WIDTH x i16> @__float_to_half_varying(<WIDTH x float> %v) nounwind readnone

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; math

declare void @__fastmath() nounwind 

;; round/floor/ceil

declare float @__round_uniform_float(float) nounwind readnone 
declare float @__floor_uniform_float(float) nounwind readnone 
declare float @__ceil_uniform_float(float) nounwind readnone 

declare double @__round_uniform_double(double) nounwind readnone 
declare double @__floor_uniform_double(double) nounwind readnone 
declare double @__ceil_uniform_double(double) nounwind readnone 

declare <WIDTH x float> @__round_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__floor_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__ceil_varying_float(<WIDTH x float>) nounwind readnone 

declare <WIDTH x double> @__round_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__floor_varying_double(<WIDTH x double>) nounwind readnone 
declare <WIDTH x double> @__ceil_varying_double(<WIDTH x double>) nounwind readnone 

;; min/max uniform

;; declare float @__max_uniform_float(float, float) nounwind readnone 
;; declare float @__min_uniform_float(float, float) nounwind readnone 
define  float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
  %d = fcmp ogt float %0, %1 
  %r = select i1 %d, float %0, float %1
  ret float %r

}
define  float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
  %d = fcmp olt float %0, %1 
  %r = select i1 %d, float %0, float %1
  ret float %r

}

;; declare i32 @__min_uniform_int32(i32, i32) nounwind readnone 
;; declare i32 @__max_uniform_int32(i32, i32) nounwind readnone 
define  i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp slt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}
define  i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp sgt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

;; declare i32 @__min_uniform_uint32(i32, i32) nounwind readnone 
;; declare i32 @__max_uniform_uint32(i32, i32) nounwind readnone 
define  i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp ult i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}
define  i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %c = icmp ugt i32 %0, %1
  %r = select i1 %c, i32 %0, i32 %1
  ret i32 %r
}

;; declare i64 @__min_uniform_int64(i64, i64) nounwind readnone 
;; declare i64 @__max_uniform_int64(i64, i64) nounwind readnone 
;; declare i64 @__min_uniform_uint64(i64, i64) nounwind readnone 
;; declare i64 @__max_uniform_uint64(i64, i64) nounwind readnone 

;; declare double @__min_uniform_double(double, double) nounwind readnone 
;; declare double @__max_uniform_double(double, double) nounwind readnone 
define  double @__max_uniform_double(double, double) nounwind readonly alwaysinline {
  %d = fcmp ogt double %0, %1 
  %r = select i1 %d, double %0, double %1
  ret double %r
}
define  double @__min_uniform_double(double, double) nounwind readonly alwaysinline {
  %d = fcmp olt double %0, %1 
  %r = select i1 %d, double %0, double %1
  ret double %r
}

;; min/max uniform

declare <WIDTH x float> @__max_varying_float(<WIDTH x float>, <WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__min_varying_float(<WIDTH x float>, <WIDTH x float>) nounwind readnone 
declare <WIDTH x i32> @__min_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone 
declare <WIDTH x i32> @__max_varying_int32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone 
declare <WIDTH x i32> @__min_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone 
declare <WIDTH x i32> @__max_varying_uint32(<WIDTH x i32>, <WIDTH x i32>) nounwind readnone 
;; declare <WIDTH x i64> @__min_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone 
;; declare <WIDTH x i64> @__max_varying_int64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone 
;; declare <WIDTH x i64> @__min_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone 
;; declare <WIDTH x i64> @__max_varying_uint64(<WIDTH x i64>, <WIDTH x i64>) nounwind readnone 
declare <WIDTH x double> @__min_varying_double(<WIDTH x double>,
                                               <WIDTH x double>) nounwind readnone
declare <WIDTH x double> @__max_varying_double(<WIDTH x double>,
                                               <WIDTH x double>) nounwind readnone 

;; sqrt/rsqrt/rcp

declare float     @llvm.nvvm.rsqrt.approx.f(float %f) nounwind readonly alwaysinline
declare float     @llvm.nvvm.sqrt.f(float %f) nounwind readonly alwaysinline
declare double    @llvm.nvvm.rsqrt.approx.d(double %f) nounwind readonly alwaysinline
declare double    @llvm.nvvm.sqrt.d(double %f) nounwind readonly alwaysinline

;; declare float @__rcp_uniform_float(float) nounwind readnone 
define  float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
;    uniform float iv = extract(__rcp_u(v), 0);
;    return iv * (2. - v * iv);
  %r = fdiv float 1.,%0
  ret float %r
}
;; declare float @__sqrt_uniform_float(float) nounwind readnone 
define  float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %ret = call float @llvm.nvvm.sqrt.f(float %0)
  ret float %ret
}
;; declare float @__rsqrt_uniform_float(float) nounwind readnone 
define  float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline 
{
  %ret = call float @llvm.nvvm.rsqrt.approx.f(float %0)
  ret float %ret
}

declare <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float>) nounwind readnone 
declare <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readnone 

;; declare double @__sqrt_uniform_double(double) nounwind readnone
define  double @__sqrt_uniform_double(double) nounwind readonly alwaysinline {
  %ret = call double @llvm.nvvm.sqrt.d(double %0)
  ret double %ret
}
declare <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone

;; bit ops

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
define  i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
  %call = call i32 @llvm.ctpop.i32(i32 %0)
  ret i32 %call
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone
define  i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %call = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %call
}

ctlztz()

; FIXME: need either to wire these up to the 8-wide SVML entrypoints,
; or, use the macro to call the 4-wide ones twice with our 8-wide
; vectors...

;; svml

include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; population count;




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define  i64 @__movmsk(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %v64 = zext i1 %v to i64
  ret i64 %v64
}

define  i1 @__any(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %cmp = icmp ne i1 %v, 0
  ret i1 %cmp
}

define  i1 @__all(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %cmp = icmp eq i1 %v, 1
  ret i1 %cmp
}

define  i1 @__none(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %cmp = icmp eq i1 %v, 0
  ret i1 %cmp
}

declare i16 @__reduce_add_int8(<WIDTH x i8>) nounwind readnone
declare i32 @__reduce_add_int16(<WIDTH x i16>) nounwind readnone

define  float @__reduce_add_float(<1 x float> %v) nounwind readonly alwaysinline {
  %r = extractelement <1 x float> %v, i32 0
  ret float %r
}

define  float @__reduce_min_float(<1 x float>) nounwind readnone {
  %r = extractelement <1 x float> %0, i32 0
  ret float %r
}

define  float @__reduce_max_float(<1 x float>) nounwind readnone {
  %r = extractelement <1 x float> %0, i32 0
  ret float %r
}

define  i32 @__reduce_add_int32(<1 x i32> %v) nounwind readnone {
  %r = extractelement <1 x i32> %v, i32 0
  ret i32 %r
}

define  i32 @__reduce_min_int32(<1 x i32>) nounwind readnone {
  %r = extractelement <1 x i32> %0, i32 0
  ret i32 %r
}

define  i32 @__reduce_max_int32(<1 x i32>) nounwind readnone {
  %r = extractelement <1 x i32> %0, i32 0
  ret i32 %r
}

define  i32 @__reduce_min_uint32(<1 x i32>) nounwind readnone {
  %r = extractelement <1 x i32> %0, i32 0
  ret i32 %r
}

define  i32 @__reduce_max_uint32(<1 x i32>) nounwind readnone {
  %r = extractelement <1 x i32> %0, i32 0
  ret i32 %r
 }


define  double @__reduce_add_double(<1 x double>) nounwind readnone {
  %m = extractelement <1 x double> %0, i32 0
  ret double %m
}

define  double @__reduce_min_double(<1 x double>) nounwind readnone {
  %m = extractelement <1 x double> %0, i32 0
  ret double %m
}

define  double @__reduce_max_double(<1 x double>) nounwind readnone {
  %m = extractelement <1 x double> %0, i32 0
  ret double %m
}

define  i64 @__reduce_add_int64(<1 x i64>) nounwind readnone {
  %m = extractelement <1 x i64> %0, i32 0
  ret i64 %m
}

define  i64 @__reduce_min_int64(<1 x i64>) nounwind readnone {
  %m = extractelement <1 x i64> %0, i32 0
  ret i64 %m
}

define  i64 @__reduce_max_int64(<1 x i64>) nounwind readnone {
  %m = extractelement <1 x i64> %0, i32 0
  ret i64 %m
}

define  i64 @__reduce_min_uint64(<1 x i64>) nounwind readnone {
  %m = extractelement <1 x i64> %0, i32 0
  ret i64 %m
}

define  i64 @__reduce_max_uint64(<1 x i64>) nounwind readnone {
  %m = extractelement <1 x i64> %0, i32 0
  ret i64 %m
}

define  i1 @__reduce_equal_int32(<1 x i32> %vv, i32 * %samevalue,
                                      <1 x i32> %mask) nounwind alwaysinline {
  %v=extractelement <1 x i32> %vv, i32 0
  store i32 %v, i32 * %samevalue
  ret i1 true

}

define  i1 @__reduce_equal_float(<1 x float> %vv, float * %samevalue,
                                      <1 x i32> %mask) nounwind alwaysinline {
  %v=extractelement <1 x float> %vv, i32 0
  store float %v, float * %samevalue
  ret i1 true

}

define  i1 @__reduce_equal_int64(<1 x i64> %vv, i64 * %samevalue,
                                      <1 x i32> %mask) nounwind alwaysinline {
  %v=extractelement <1 x i64> %vv, i32 0
  store i64 %v, i64 * %samevalue
  ret i1 true

}

define  i1 @__reduce_equal_double(<1 x double> %vv, double * %samevalue,
                                      <1 x i32> %mask) nounwind alwaysinline {
  %v=extractelement <1 x double> %vv, i32 0
  store double %v, double * %samevalue
  ret i1 true

}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts


masked_load(i8,  1)
masked_load(i16, 2)
masked_load(i32, 4)
masked_load(float, 4)
masked_load(i64, 8)
masked_load(double, 8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

gen_masked_store(i8)
gen_masked_store(i16)
gen_masked_store(i32)
gen_masked_store(float)
gen_masked_store(i64)
gen_masked_store(double)

define void @__masked_store_blend_i8(<WIDTH x i8>* nocapture, <WIDTH x i8>, 
                                     <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x i8> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i8> %1, <WIDTH x i8> %v
  store <WIDTH x i8> %v1, <WIDTH x i8> * %0
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x i16> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i16> %1, <WIDTH x i16> %v
  store <WIDTH x i16> %v1, <WIDTH x i16> * %0
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x i32> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i32> %1, <WIDTH x i32> %v
  store <WIDTH x i32> %v1, <WIDTH x i32> * %0
  ret void
}

define void @__masked_store_blend_float(<WIDTH x float>* nocapture, <WIDTH x float>, 
                                        <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x float> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x float> %1, <WIDTH x float> %v
  store <WIDTH x float> %v1, <WIDTH x float> * %0
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture,
                            <WIDTH x i64>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x i64> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i64> %1, <WIDTH x i64> %v
  store <WIDTH x i64> %v1, <WIDTH x i64> * %0
  ret void
}

define void @__masked_store_blend_double(<WIDTH x double>* nocapture,
                            <WIDTH x double>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load <WIDTH x double> * %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x double> %1, <WIDTH x double> %v
  store <WIDTH x double> %v1, <WIDTH x double> * %0
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather/scatter

; define these with the macros from stdlib.m4

gen_gather_factored(i8)
gen_gather_factored(i16)
gen_gather_factored(i32)
gen_gather_factored(float)
gen_gather_factored(i64)
gen_gather_factored(double)

gen_scatter(i8)
gen_scatter(i16)
gen_scatter(i32)
gen_scatter(float)
gen_scatter(i64)
gen_scatter(double)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch

;; define void @__prefetch_read_uniform_1(i8 * nocapture) nounwind alwaysinline { }
;; define void @__prefetch_read_uniform_2(i8 * nocapture) nounwind alwaysinline { }
;; define void @__prefetch_read_uniform_3(i8 * nocapture) nounwind alwaysinline { }
;; define void @__prefetch_read_uniform_nt(i8 * nocapture) nounwind alwaysinline { }

define_prefetches()
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

