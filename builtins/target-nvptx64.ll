;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Define the standard library builtins for the NOVEC target
define(`MASK',`i1')
define(`WIDTH',`1')

;; target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

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
define i32 @__shfl_i32(i32, i32) nounwind readnone alwaysinline
{
  %shfl = tail call i32 asm sideeffect "shfl.idx.b32  $0, $1, $2, 0x1f;", "=r,r,r"(i32 %0, i32 %1) nounwind readnone alwaysinline
  ret i32 %shfl
}
define float @__shfl_xor_float(float, i32) nounwind readnone alwaysinline
{
  %shfl = tail call float asm sideeffect "shfl.bfly.b32  $0, $1, $2, 0x1f;", "=f,f,r"(float %0, i32 %1) nounwind readnone alwaysinline
  ret float %shfl
}
define float @__fminf(float,float) nounwind readnone alwaysinline
{
  %min = tail call float asm sideeffect "min.f32 $0, $1, $2;", "=f,f,f"(float %0, float %1) nounwind readnone alwaysinline
  ret float %min
}
define float @__fmaxf(float,float) nounwind readnone alwaysinline
{
  %max = tail call float asm sideeffect "max.f32 $0, $1, $2;", "=f,f,f"(float %0, float %1) nounwind readnone alwaysinline
  ret float %max
}
define i32 @__ballot(i1) nounwind readnone alwaysinline
{
  %conv = zext i1 %0 to i32
  %res = tail call i32 asm sideeffect 
      "{ .reg .pred %p1; 
         setp.ne.u32 %p1, $1, 0; 
         vote.ballot.b32  $0, %p1; 
      }", "=r,r"(i32 %conv) nounwind readnone alwaysinline
  ret i32 %res
}
define i32 @__lanemask_lt() nounwind readnone alwaysinline
{
  %mask = tail call i32 asm sideeffect "mov.u32 $0, %lanemask_lt;", "=r"() nounwind readnone alwaysinline
  ret i32 %mask
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; tasking

module asm "
.extern .func  (.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b64 cudaLaunchDevice_param_0,
  .param .b64 cudaLaunchDevice_param_1,
  .param .align 4 .b8 cudaLaunchDevice_param_2[12],
  .param .align 4 .b8 cudaLaunchDevice_param_3[12],
  .param .b32 cudaLaunchDevice_param_4,
  .param .b64 cudaLaunchDevice_param_5
);
"
define i8* @ISPCAlloc(i8**, i64, i32) nounwind alwaysinline
{
  ret i8* null
}
define void @ISPCLaunch(i8**, i8* %func_ptr, i8** %func_args, i32 %ntx, i32 %nty, i32 %ntz) nounwind alwaysinline
{
  %func_i64 = ptrtoint i8*  %func_ptr  to i64
  %args_i64 = ptrtoint i8** %func_args to i64
;; nbx = (%ntx-1)/(blocksize/warpsize) + 1  for blocksize=128 & warpsize=32
  %sub = add nsw i32 %ntx, -1
  %div = sdiv i32 %sub, 4
  %nbx = add nsw i32 %div, 1

  %res = call i32 asm sideeffect "{
      .reg .s32 %r<8>;
      .reg .s64 %rd<3>;
     .param .b64 param0;
     st.param.b64	[param0+0], $1; //%rd0;
     .param .b64 param1;
     st.param.b64	[param1+0], $2; //%rd1;
     .param .align 4 .b8 param2[12];
     st.param.b32	[param2+0], $3; //%r0;
     st.param.b32	[param2+4], $4; //%r1;
     st.param.b32	[param2+8], $5; //%r2;
     .param .align 4 .b8 param3[12];
     st.param.b32	[param3+0], $6; //%r3;
     st.param.b32	[param3+4], $7; //%r4;
     st.param.b32	[param3+8], $8; //%r5;
     .param .b32 param4;
     st.param.b32	[param4+0], $9; //%r6;
     .param .b64 param5;
     st.param.b64	[param5+0], $10; //%rd2;

     .param .b32 retval0;
     call.uni (retval0), 
       cudaLaunchDevice,
       (
        param0, 
        param1, 
        param2, 
        param3, 
        param4, 
        param5
       );
     ld.param.b32	$0, [retval0+0];
  }
  ", 
"=r, l,l, r,r,r, r,r,r, r,l"(i64 %func_i64,i64 %args_i64, i32 %nbx,i32 %nty,i32 %ntz, i32 128,i32 1,i32 1, i32 0,i64 0);
  ret void
}
declare i32 @cudaDeviceSynchronize() nounwind
define void @ISPCSync(i8*) nounwind alwaysinline
{
  call i32 @cudaDeviceSynchronize()
  ret void;
}


;;;;;;;;;;;;;;



include(`util_ptx.m4')

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

declare float @llvm.convert.from.fp16(i16) nounwind readnone
declare i16   @llvm.convert.to.fp16(float) nounwind readnone
define float @__half_to_float_uniform(i16 %v) nounwind readnone alwaysinline
{
  ;; %res = call float @llvm.convert.from.fp16(i16 %v)
  %res = tail call float asm sideeffect 
      "{ .reg .b16 %tmp; 
         mov.b16 %tmp, $1;
         cvt.f32.f16 $0, %tmp;
      }", "=f,h"(i16 %v) nounwind readnone alwaysinline
  ret float %res
}
define i16 @__float_to_half_uniform(float %v) nounwind readnone alwaysinline
{
 ;; this will break the compiler, use inline asm similarly to above case
  %half = call i16 @llvm.convert.to.fp16(float %v)
  ret i16 %half
}
define <WIDTH x float> @__half_to_float_varying(<WIDTH x i16> %v) nounwind readnone alwaysinline
{
  %el = extractelement <1 x i16> %v, i32 0
  %sf = call float @__half_to_float_uniform(i16 %el)
  %vf = insertelement <1 x float> undef, float %sf, i32 0
  ret <1 x float> %vf;
}
define <WIDTH x i16> @__float_to_half_varying(<WIDTH x float> %v) nounwind readnone alwaysinline
{
  %el = extractelement <1 x float> %v, i32 0
  %sh = call i16 @__float_to_half_uniform(float %el)
  %vh = insertelement <1 x i16> undef, i16 %sh, i32 0
  ret <1 x i16> %vh;
}

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

define  <1 x float> @__round_varying_float(<1 x float>) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast <1 x float> %0 to <1 x i32>
  %bitop.i.i = and <1 x i32> %float_to_int_bitcast.i.i.i.i, <i32 -2147483648>
  %bitop.i = xor <1 x i32> %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast <1 x i32> %bitop.i to <1 x float>
  %binop.i = fadd <1 x float> %int_to_float_bitcast.i.i40.i, <float 8.388608e+06>
  %binop21.i = fadd <1 x float> %binop.i, <float -8.388608e+06>
  %float_to_int_bitcast.i.i.i = bitcast <1 x float> %binop21.i to <1 x i32>
  %bitop31.i = xor <1 x i32> %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast <1 x i32> %bitop31.i to <1 x float>
  ret <1 x float> %int_to_float_bitcast.i.i.i
}
define  <1 x float> @__floor_varying_float(<1 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <1 x float> @__round_varying_float(<1 x float> %0) nounwind
  %bincmp.i = fcmp ogt <1 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <1 x i1> %bincmp.i to <1 x i32>
  %bitop.i = and <1 x i32> %val_to_boolvec32.i, <i32 -1082130432>
  %int_to_float_bitcast.i.i.i = bitcast <1 x i32> %bitop.i to <1 x float>
  %binop.i = fadd <1 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <1 x float> %binop.i
}

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

;; /* float */
define  <1 x float> @__max_varying_float(<1 x float>, <1 x float>) nounwind readonly alwaysinline {
  %a = extractelement <1 x float> %0, i32 0
  %b = extractelement <1 x float> %1, i32 0
  %r = call float @__max_uniform_float(float %a, float %b)
  %rv = insertelement <1 x float> undef, float %r, i32 0
  ret <1 x float> %rv    
}
define  <1 x float> @__min_varying_float(<1 x float>, <1 x float>) nounwind readonly alwaysinline {
  %a = extractelement <1 x float> %0, i32 0
  %b = extractelement <1 x float> %1, i32 0
  %r = call float @__min_uniform_float(float %a, float %b)
  %rv = insertelement <1 x float> undef, float %r, i32 0
  ret <1 x float> %rv    

}

;; /* int32 */
define  <1 x i32> @__max_varying_int32(<1 x i32>, <1 x i32>) nounwind readonly alwaysinline {
  %a = extractelement <1 x i32> %0, i32 0
  %b = extractelement <1 x i32> %1, i32 0
  %r = call i32 @__max_uniform_int32(i32 %a, i32 %b)
  %rv = insertelement <1 x i32> undef, i32 %r, i32 0
  ret <1 x i32> %rv
}
define  <1 x i32> @__min_varying_int32(<1 x i32>, <1 x i32>) nounwind readonly alwaysinline {
  %a = extractelement <1 x i32> %0, i32 0
  %b = extractelement <1 x i32> %1, i32 0
  %r = call i32 @__min_uniform_int32(i32 %a, i32 %b)
  %rv = insertelement <1 x i32> undef, i32 %r, i32 0
  ret <1 x i32> %rv
}

;; /* uint32 */
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

define <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readnone  alwaysinline
{
  %v = extractelement <1 x float> %0, i32 0
  %r = call float @__rcp_uniform_float(float %v)
  %rv = insertelement <1 x float> undef, float %r, i32 0 
  ret <WIDTH x float> %rv
}
define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float>) nounwind readnone alwaysinline
{
  %v = extractelement <1 x float> %0, i32 0
  %r = call float @__rsqrt_uniform_float(float %v)
  %rv = insertelement <1 x float> undef, float %r, i32 0 
  ret <WIDTH x float> %rv
}
define <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readnone alwaysinline
{
  %v = extractelement <1 x float> %0, i32 0
  %r = call float @__sqrt_uniform_float(float %v)
  %rv = insertelement <1 x float> undef, float %r, i32 0 
  ret <WIDTH x float> %rv
}

;; declare double @__sqrt_uniform_double(double) nounwind readnone
define  double @__sqrt_uniform_double(double) nounwind readonly alwaysinline {
  %ret = call double @llvm.nvvm.sqrt.d(double %0)
  ret double %ret
}
declare <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone

;; bit ops

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
define  i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
;;  %call = call i32 @llvm.ctpop.i32(i32 %0)
;;  ret i32 %call
  %res = tail call i32 asm sideeffect "popc.b32 $0, $1;", "=r,r"(i32 %0) nounwind readnone alwaysinline
  ret i32 %res
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone
define  i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %call = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %call
}

define i64 @__warpBinExclusiveScan(i1 %p) nounwind readonly alwaysinline 
{
entry:
  %call = call i32 @__ballot(i1 zeroext %p)
  %call1 = call i32 @__popcnt_int32(i32 %call)
  %call2 = call i32 @__lanemask_lt()
  %and = and i32 %call2, %call
  %call3 = call i32 @__popcnt_int32(i32 %and)
  %retval.sroa.1.4.insert.ext.i = zext i32 %call3 to i64
  %retval.sroa.1.4.insert.shift.i = shl nuw i64 %retval.sroa.1.4.insert.ext.i, 32
  %retval.sroa.0.0.insert.ext.i = zext i32 %call1 to i64
  %retval.sroa.0.0.insert.insert.i = or i64 %retval.sroa.1.4.insert.shift.i, %retval.sroa.0.0.insert.ext.i
  ret i64 %retval.sroa.0.0.insert.insert.i
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
  %res = call i32 @__ballot(i1 %v)
  %cmp = icmp ne i32 %res, 0
  ret i1 %cmp
}

define  i1 @__all(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %res = call i32 @__ballot(i1 %v)
  %cmp = icmp eq i32 %res, 31
  ret i1 %cmp
}

define  i1 @__none(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %res = call i32 @__ballot(i1 %v)
  %cmp = icmp eq i32 %res, 0
  ret i1 %cmp
}

declare i16 @__reduce_add_int8(<WIDTH x i8>) nounwind readnone
declare i32 @__reduce_add_int16(<WIDTH x i16>) nounwind readnone

define  float @__reduce_add_float(<1 x float> %v) nounwind readonly alwaysinline {
  %r = extractelement <1 x float> %v, i32 0
  ret float %r
}

define  float @__reduce_min_float(<1 x float>) nounwind readnone {
  %value = extractelement <1 x float> %0, i32 0
  %call = tail call float @__shfl_xor_float(float %value, i32 16)
  %call1 = tail call float @__fminf(float %value, float %call) #4
  %call.1 = tail call float @__shfl_xor_float(float %call1, i32 8)
  %call1.1 = tail call float @__fminf(float %call1, float %call.1) #4
  %call.2 = tail call float @__shfl_xor_float(float %call1.1, i32 4)
  %call1.2 = tail call float @__fminf(float %call1.1, float %call.2) #4
  %call.3 = tail call float @__shfl_xor_float(float %call1.2, i32 2)
  %call1.3 = tail call float @__fminf(float %call1.2, float %call.3) #4
  %call.4 = tail call float @__shfl_xor_float(float %call1.3, i32 1)
  %call1.4 = tail call float @__fminf(float %call1.3, float %call.4) #4
  ret float %call1.4
}

define  float @__reduce_max_float(<1 x float>) nounwind readnone 
{
  %value = extractelement <1 x float> %0, i32 0
  %call = tail call float @__shfl_xor_float(float %value, i32 16)
  %call1 = tail call float @__fmaxf(float %value, float %call) 
  %call.1 = tail call float @__shfl_xor_float(float %call1, i32 8)
  %call1.1 = tail call float @__fmaxf(float %call1, float %call.1) 
  %call.2 = tail call float @__shfl_xor_float(float %call1.1, i32 4)
  %call1.2 = tail call float @__fmaxf(float %call1.1, float %call.2) 
  %call.3 = tail call float @__shfl_xor_float(float %call1.2, i32 2)
  %call1.3 = tail call float @__fmaxf(float %call1.2, float %call.3) 
  %call.4 = tail call float @__shfl_xor_float(float %call1.3, i32 1)
  %call1.4 = tail call float @__fmaxf(float %call1.3, float %call.4) 
  ret float %call1.4
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
                                      <1 x i1> %mask) nounwind alwaysinline {
  %v=extractelement <1 x i32> %vv, i32 0
  store i32 %v, i32 * %samevalue
  ret i1 true

}

define  i1 @__reduce_equal_float(<1 x float> %vv, float * %samevalue,
                                      <1 x i1> %mask) nounwind alwaysinline {
  %v=extractelement <1 x float> %vv, i32 0
  store float %v, float * %samevalue
  ret i1 true

}

define  i1 @__reduce_equal_int64(<1 x i64> %vv, i64 * %samevalue,
                                      <1 x i1> %mask) nounwind alwaysinline {
  %v=extractelement <1 x i64> %vv, i32 0
  store i64 %v, i64 * %samevalue
  ret i1 true

}

define  i1 @__reduce_equal_double(<1 x double> %vv, double * %samevalue,
                                      <1 x i1> %mask) nounwind alwaysinline {
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

