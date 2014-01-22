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
;; %tid = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
;; ret i32 %tid
  ret i32 32
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
;;;;;;;;
declare i64* @llvm.nvvm.ptr.shared.to.gen.p0i64.p3i64(i64 addrspace(3)*)
declare i64* @llvm.nvvm.ptr.shared.to.gen.p0i64.p4i64(i64 addrspace(4)*)
define i64* @__cvt_loc2gen(i64 addrspace(3)*) nounwind readnone alwaysinline
{
  %ptr =  tail call i64* @llvm.nvvm.ptr.shared.to.gen.p0i64.p3i64(i64 addrspace(3)* %0)
  ret i64* %ptr
}
define i64* @__cvt_loc2gen_var(i64 addrspace(3)*) nounwind readnone alwaysinline
{
  %ptr =  tail call i64* @llvm.nvvm.ptr.shared.to.gen.p0i64.p3i64(i64 addrspace(3)* %0)
  ret i64* %ptr
}
define i64* @__cvt_const2gen(i64 addrspace(4)*) nounwind readnone alwaysinline
{
  %ptr =  tail call i64* @llvm.nvvm.ptr.shared.to.gen.p0i64.p4i64(i64 addrspace(4)* %0)
  ret i64* %ptr
}

;;;;;;;;
;; i32
define i32 @__shfl_i32_nvptx(i32, i32) nounwind readnone alwaysinline
{
  %shfl = tail call i32 asm sideeffect "shfl.idx.b32  $0, $1, $2, 0x1f;", "=r,r,r"(i32 %0, i32 %1) nounwind readnone alwaysinline
  ret i32 %shfl
}
define i32 @__shfl_xor_i32_nvptx(i32, i32) nounwind readnone alwaysinline
{
  %shfl = tail call i32 asm sideeffect "shfl.bfly.b32  $0, $1, $2, 0x1f;", "=r,r,r"(i32 %0, i32 %1) nounwind readnone alwaysinline
  ret i32 %shfl
}
;; float
define float @__shfl_float_nvptx(float, i32) nounwind readnone alwaysinline
{
  %shfl = tail call float asm sideeffect "shfl.idx.b32  $0, $1, $2, 0x1f;", "=f,f,r"(float %0, i32 %1) nounwind readnone alwaysinline
  ret float %shfl
}
define float @__shfl_xor_float_nvptx(float, i32) nounwind readnone alwaysinline
{
  %shfl = tail call float asm sideeffect "shfl.bfly.b32  $0, $1, $2, 0x1f;", "=f,f,r"(float %0, i32 %1) nounwind readnone alwaysinline
  ret float %shfl
}

;;;;;;;;;;; min/max
;; float/double
define float @__fminf_nvptx(float,float) nounwind readnone alwaysinline
{
  %min = tail call float asm sideeffect "min.f32 $0, $1, $2;", "=f,f,f"(float %0, float %1) nounwind readnone alwaysinline
  ret float %min
}
define float @__fmaxf_nvptx(float,float) nounwind readnone alwaysinline
{
  %max = tail call float asm sideeffect "max.f32 $0, $1, $2;", "=f,f,f"(float %0, float %1) nounwind readnone alwaysinline
  ret float %max
}

;; int
define(`int_minmax',`
define $1 @__min_$1_signed($1,$1) nounwind readnone alwaysinline {
  %c = icmp slt $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
define $1 @__max_$1_signed($1,$1) nounwind readnone alwaysinline {
  %c = icmp sgt $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
define $1 @__min_$1_unsigned($1,$1) nounwind readnone alwaysinline  {
  %c = icmp ult $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
define $1 @__max_$1_unsigned($1,$1) nounwind readnone alwaysinline {
  %c = icmp ugt $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
')
int_minmax(i8);
int_minmax(i16);
int_minmax(i32);
int_minmax(i64);

;; float/double
define(`fp_minmax',`
define $1 @__min_$1($1,$1) nounwind readnone alwaysinline {
  %c = fcmp olt $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
define $1 @__max_$1($1,$1) nounwind readnone alwaysinline {
  %c = fcmp ogt $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
')
fp_minmax(float)
fp_minmax(double)

;;;;;;;;; __shfl/__shfl_xor intrinsics
;;  i8/i16/i64 
define(`shfl32',`
define $2 @$1_$2_nvptx($2, i32) nounwind readnone alwaysinline
{
  %ext = zext $2 %0 to i32
  %res = tail call i32 @$1_i32_nvptx(i32 %ext, i32 %1)
  %ret = trunc i32 %res to $2
  ret $2 %ret
}
')
shfl32(__shfl,     i8);
shfl32(__shfl_xor, i8);
shfl32(__shfl,     i16);
shfl32(__shfl_xor, i16);


define(`shfl64',`
define $2 @$1_$2_nvptx($2, i32) nounwind readnone alwaysinline
{
  %in   = bitcast $2 %0 to <2 x i32>
  %in0  = extractelement <2 x i32> %in, i32 0
  %in1  = extractelement <2 x i32> %in, i32 1
  %out0 = tail call i32 @$1_i32_nvptx(i32 %in0, i32 %1)
  %out1 = tail call i32 @$1_i32_nvptx(i32 %in1, i32 %1)
  %out2 = insertelement <2 x i32> undef, i32 %out0, i32 0
  %out  = insertelement <2 x i32> %out2, i32 %out1, i32 1
  %ret  = bitcast <2 x i32> %out to $2
  ret $2 %ret
}
')
shfl64(__shfl,     i64)
shfl64(__shfl_xor, i64)
shfl64(__shfl,     double)
shfl64(__shfl_xor, double)

;;;;;;;;;;;;;
define i32 @__ballot_nvptx(i1) nounwind readnone alwaysinline
{
  %conv = zext i1 %0 to i32
  %res = tail call i32 asm sideeffect 
      "{ .reg .pred %p1; 
         setp.ne.u32 %p1, $1, 0; 
         vote.ballot.b32  $0, %p1; 
      }", "=r,r"(i32 %conv) nounwind readnone alwaysinline
  ret i32 %res
}
define i32 @__lanemask_lt_nvptx() nounwind readnone alwaysinline
{
  %mask = tail call i32 asm sideeffect "mov.u32 $0, %lanemask_lt;", "=r"() nounwind readnone alwaysinline
  ret i32 %mask
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; tasking

;; this call allocate parameter buffer for kernel launch
declare i64 @cudaGetParameterBuffer(i64, i64) nounwind
define i8* @ISPCAlloc(i8**, i64 %size, i32 %align32) nounwind alwaysinline
{
entry:
  %call = tail call i32 @__tid_x()
  %call1 = tail call i32 @__warpsize()
  %sub = add nsw i32 %call1, -1
  %and = and i32 %sub, %call
  %cmp = icmp eq i32 %and, 0
  %align = zext i32 %align32 to i64
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %ptri64tmp = call i64 @cudaGetParameterBuffer(i64 %align, i64 %size);
  br label %if.end

if.end:
  %ptri64 = phi i64 [ %ptri64tmp, %if.then ], [ 0, %entry ]
  %ptr = inttoptr i64 %ptri64 to i8*
  ret i8* %ptr
}

;; this actually launches kernel a kernel
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
define void @ISPCLaunch(i8**, i8* %func_ptr, i8* %func_args, i32 %ntx, i32 %nty, i32 %ntz) nounwind alwaysinline
{
entry:
;;  only 1 lane must launch the kernel  !!!
 %func_i64 = ptrtoint i8*  %func_ptr  to i64
 %args_i64 = ptrtoint i8*  %func_args to i64

;; nbx = (%ntx-1)/(blocksize/warpsize) + 1  for blocksize=128 & warpsize=32
  %ntxm1   = add nsw i32 %ntx, -1
;;  %ntxm1d4 = sdiv i32 %ntxm1, 4
  %ntxm1d4 = ashr i32 %ntxm1, 2
  %nbx     = add nsw i32 %ntxm1d4, 1
  %call = tail call i32 @__tid_x()
  %call1 = tail call i32 @__warpsize()
  %sub = add nsw i32 %call1, -1
  %and = and i32 %sub, %call
;; if (laneIdx == 0)
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:

 %res_tmp = call i32 asm sideeffect "{
     .param .b64 param0;
     st.param.b64	[param0+0], $1;
     .param .b64 param1;
     st.param.b64	[param1+0], $2;
     .param .align 4 .b8 param2[12];
     st.param.b32	[param2+0], $3; 
     st.param.b32	[param2+4], $4; 
     st.param.b32	[param2+8], $5; 
     .param .align 4 .b8 param3[12];
     st.param.b32	[param3+0], $6; 
     st.param.b32	[param3+4], $7; 
     st.param.b32	[param3+8], $8; 
     .param .b32 param4;
     st.param.b32	[param4+0], $9; 
     .param .b64 param5;
     st.param.b64	[param5+0], $10; 

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
"=r, l,l, r,r,r, r,r,r, r,l"(
          i64 %func_i64,i64 %args_i64, 
          i32 %nbx,i32 %nty,i32 %ntz, 
          i32 128,i32 1,i32 1, i32 0,i64 0);
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
;;  %res = phi i32 [ %res_tmp, %if.then ], [ undef, %entry ]

  ret void
}

;; this synchronizes kernel
declare i32 @cudaDeviceSynchronize() nounwind
define void @ISPCSync(i8*) nounwind alwaysinline
{
  call i32 @cudaDeviceSynchronize()
  ret void;
}


;;;;;;;;;;;;;;



include(`util-nvptx.m4')

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
  %ret = fdiv float 1.,%0
;  %ret = tail call float asm sideeffect "rcp.approx.ftz.f32  $0, $1;", "=f,f"(float %0) nounwind readnone alwaysinline
  ret float %ret
}
;; declare float @__sqrt_uniform_float(float) nounwind readnone 
define  float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %ret = call float @llvm.nvvm.sqrt.f(float %0)
;  %ret = tail call float asm sideeffect "sqrt.approx.ftz.f32  $0, $1;", "=f,f"(float %0) nounwind readnone alwaysinline
  ret float %ret
}
;; declare float @__rsqrt_uniform_float(float) nounwind readnone 
define  float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline 
{
  %ret = call float @llvm.nvvm.rsqrt.approx.f(float %0)
;  %ret = tail call float asm sideeffect "rsqrt.approx.ftz.f32  $0, $1;", "=f,f"(float %0) nounwind readnone alwaysinline
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
  %call = call i32 @__ballot_nvptx(i1 zeroext %p)
  %call1 = call i32 @__popcnt_int32(i32 %call)
  %call2 = call i32 @__lanemask_lt_nvptx()
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
  %res = call i32 @__ballot_nvptx(i1 %v)
  %cmp = icmp ne i32 %res, 0
  ret i1 %cmp
}

define  i1 @__all(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %res = call i32 @__ballot_nvptx(i1 %v)
  %cmp = icmp eq i32 %res, 31
  ret i1 %cmp
}

define  i1 @__none(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %res = call i32 @__ballot_nvptx(i1 %v)
  %cmp = icmp eq i32 %res, 0
  ret i1 %cmp
}

;;;;;;;;; reductions i8
define i16 @__reduce_add_int8(<1 x i8> %v) nounwind readnone alwaysinline {
  %value8 = extractelement <1 x i8> %v, i32 0
  %value  = zext i8 %value8 to i16
  %call = tail call i16 @__shfl_xor_i16_nvptx(i16 %value, i32 16)
  %call1 = add i16 %call, %value 
  %call.1 = tail call i16 @__shfl_xor_i16_nvptx(i16 %call1, i32 8)
  %call1.1 = add i16 %call1, %call.1 
  %call.2 = tail call i16 @__shfl_xor_i16_nvptx(i16 %call1.1, i32 4)
  %call1.2 = add i16 %call1.1, %call.2
  %call.3 = tail call i16 @__shfl_xor_i16_nvptx(i16 %call1.2, i32 2)
  %call1.3 = add i16 %call1.2, %call.3 
  %call.4 = tail call i16 @__shfl_xor_i16_nvptx(i16 %call1.3, i32 1)
  %call1.4 = add i16 %call1.3, %call.4 
  ret i16 %call1.4
}
;;;;;;;;; reductions i16
define i32 @__reduce_add_int16(<1 x i16> %v) nounwind readnone alwaysinline {
  %value16 = extractelement <1 x i16> %v, i32 0
  %value  = zext i16 %value16 to i32
  %call = tail call i32 @__shfl_xor_i32_nvptx(i32 %value, i32 16)
  %call1 = add i32 %call, %value 
  %call.1 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1, i32 8)
  %call1.1 = add i32 %call1, %call.1 
  %call.2 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.1, i32 4)
  %call1.2 = add i32 %call1.1, %call.2
  %call.3 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.2, i32 2)
  %call1.3 = add i32 %call1.2, %call.3 
  %call.4 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.3, i32 1)
  %call1.4 = add i32 %call1.3, %call.4 
  ret i32 %call1.4
}

;;;;;;;;; reductions float
define float @__reduce_add_float(<1 x float> %v) nounwind readonly alwaysinline {
  %value = extractelement <1 x float> %v, i32 0
  %call = tail call float @__shfl_xor_float_nvptx(float %value, i32 16)
  %call1 = fadd float %call, %value 
  %call.1 = tail call float @__shfl_xor_float_nvptx(float %call1, i32 8)
  %call1.1 = fadd float %call1, %call.1 
  %call.2 = tail call float @__shfl_xor_float_nvptx(float %call1.1, i32 4)
  %call1.2 = fadd float %call1.1, %call.2
  %call.3 = tail call float @__shfl_xor_float_nvptx(float %call1.2, i32 2)
  %call1.3 = fadd float %call1.2, %call.3 
  %call.4 = tail call float @__shfl_xor_float_nvptx(float %call1.3, i32 1)
  %call1.4 = fadd float %call1.3, %call.4 
  ret float %call1.4
}
define  float @__reduce_min_float(<1 x float>) nounwind readnone alwaysinline {
  %value = extractelement <1 x float> %0, i32 0
  %call = tail call float @__shfl_xor_float_nvptx(float %value, i32 16)
  %call1 = tail call float @__fminf_nvptx(float %value, float %call) 
  %call.1 = tail call float @__shfl_xor_float_nvptx(float %call1, i32 8)
  %call1.1 = tail call float @__fminf_nvptx(float %call1, float %call.1) 
  %call.2 = tail call float @__shfl_xor_float_nvptx(float %call1.1, i32 4)
  %call1.2 = tail call float @__fminf_nvptx(float %call1.1, float %call.2) 
  %call.3 = tail call float @__shfl_xor_float_nvptx(float %call1.2, i32 2)
  %call1.3 = tail call float @__fminf_nvptx(float %call1.2, float %call.3) 
  %call.4 = tail call float @__shfl_xor_float_nvptx(float %call1.3, i32 1)
  %call1.4 = tail call float @__fminf_nvptx(float %call1.3, float %call.4) 
  ret float %call1.4
}
define  float @__reduce_max_float(<1 x float>) nounwind readnone alwaysinline {
  %value = extractelement <1 x float> %0, i32 0
  %call = tail call float @__shfl_xor_float_nvptx(float %value, i32 16)
  %call1 = tail call float @__fmaxf_nvptx(float %value, float %call) 
  %call.1 = tail call float @__shfl_xor_float_nvptx(float %call1, i32 8)
  %call1.1 = tail call float @__fmaxf_nvptx(float %call1, float %call.1) 
  %call.2 = tail call float @__shfl_xor_float_nvptx(float %call1.1, i32 4)
  %call1.2 = tail call float @__fmaxf_nvptx(float %call1.1, float %call.2) 
  %call.3 = tail call float @__shfl_xor_float_nvptx(float %call1.2, i32 2)
  %call1.3 = tail call float @__fmaxf_nvptx(float %call1.2, float %call.3) 
  %call.4 = tail call float @__shfl_xor_float_nvptx(float %call1.3, i32 1)
  %call1.4 = tail call float @__fmaxf_nvptx(float %call1.3, float %call.4) 
  ret float %call1.4
}

;;;;;;;;; reductions int32
define  i32 @__reduce_add_int32(<1 x i32>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i32> %0, i32 0
  %call = tail call i32 @__shfl_xor_i32_nvptx(i32 %value, i32 16)
  %call1 = add i32 %call, %value 
  %call.1 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1, i32 8)
  %call1.1 =add i32 %call1, %call.1 
  %call.2 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.1, i32 4)
  %call1.2 = add i32 %call1.1, %call.2
  %call.3 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.2, i32 2)
  %call1.3 = add i32 %call1.2, %call.3 
  %call.4 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.3, i32 1)
  %call1.4 = add i32 %call1.3, %call.4 
  ret i32 %call1.4
}
define  i32 @__reduce_min_int32(<1 x i32>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i32> %0, i32 0
  %call = tail call i32 @__shfl_xor_i32_nvptx(i32 %value, i32 16)
  %call1 = tail call i32 @__min_i32_signed(i32 %value, i32 %call) 
  %call.1 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1, i32 8)
  %call1.1 = tail call i32 @__min_i32_signed(i32 %call1, i32 %call.1) 
  %call.2 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.1, i32 4)
  %call1.2 = tail call i32 @__min_i32_signed(i32 %call1.1, i32 %call.2) 
  %call.3 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.2, i32 2)
  %call1.3 = tail call i32 @__min_i32_signed(i32 %call1.2, i32 %call.3) 
  %call.4 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.3, i32 1)
  %call1.4 = tail call i32 @__min_i32_signed(i32 %call1.3, i32 %call.4) 
  ret i32 %call1.4
}
define  i32 @__reduce_max_int32(<1 x i32>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i32> %0, i32 0
  %call = tail call i32 @__shfl_xor_i32_nvptx(i32 %value, i32 16)
  %call1 = tail call i32 @__max_i32_signed(i32 %value, i32 %call) 
  %call.1 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1, i32 8)
  %call1.1 = tail call i32 @__max_i32_signed(i32 %call1, i32 %call.1) 
  %call.2 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.1, i32 4)
  %call1.2 = tail call i32 @__max_i32_signed(i32 %call1.1, i32 %call.2) 
  %call.3 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.2, i32 2)
  %call1.3 = tail call i32 @__max_i32_signed(i32 %call1.2, i32 %call.3) 
  %call.4 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.3, i32 1)
  %call1.4 = tail call i32 @__max_i32_signed(i32 %call1.3, i32 %call.4) 
  ret i32 %call1.4
}

;;;;;;;;; reductions uint32
define  i32 @__reduce_min_uint32(<1 x i32>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i32> %0, i32 0
  %call = tail call i32 @__shfl_xor_i32_nvptx(i32 %value, i32 16)
  %call1 = tail call i32 @__min_i32_unsigned(i32 %value, i32 %call) 
  %call.1 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1, i32 8)
  %call1.1 = tail call i32 @__min_i32_unsigned(i32 %call1, i32 %call.1) 
  %call.2 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.1, i32 4)
  %call1.2 = tail call i32 @__min_i32_unsigned(i32 %call1.1, i32 %call.2) 
  %call.3 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.2, i32 2)
  %call1.3 = tail call i32 @__min_i32_unsigned(i32 %call1.2, i32 %call.3) 
  %call.4 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.3, i32 1)
  %call1.4 = tail call i32 @__min_i32_unsigned(i32 %call1.3, i32 %call.4) 
  ret i32 %call1.4
}
define  i32 @__reduce_max_uint32(<1 x i32>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i32> %0, i32 0
  %call = tail call i32 @__shfl_xor_i32_nvptx(i32 %value, i32 16)
  %call1 = tail call i32 @__max_i32_unsigned(i32 %value, i32 %call) 
  %call.1 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1, i32 8)
  %call1.1 = tail call i32 @__max_i32_unsigned(i32 %call1, i32 %call.1) 
  %call.2 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.1, i32 4)
  %call1.2 = tail call i32 @__max_i32_unsigned(i32 %call1.1, i32 %call.2) 
  %call.3 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.2, i32 2)
  %call1.3 = tail call i32 @__max_i32_unsigned(i32 %call1.2, i32 %call.3) 
  %call.4 = tail call i32 @__shfl_xor_i32_nvptx(i32 %call1.3, i32 1)
  %call1.4 = tail call i32 @__max_i32_unsigned(i32 %call1.3, i32 %call.4) 
  ret i32 %call1.4
 }

;;;;;;;;; reductions double
define  double @__reduce_add_double(<1 x double>) nounwind readnone alwaysinline {
  %value = extractelement <1 x double> %0, i32 0
  %call = tail call double @__shfl_xor_double_nvptx(double %value, i32 16)
  %call1 = fadd double %call, %value 
  %call.1 = tail call double @__shfl_xor_double_nvptx(double %call1, i32 8)
  %call1.1 = fadd double %call1, %call.1 
  %call.2 = tail call double @__shfl_xor_double_nvptx(double %call1.1, i32 4)
  %call1.2 = fadd double %call1.1, %call.2
  %call.3 = tail call double @__shfl_xor_double_nvptx(double %call1.2, i32 2)
  %call1.3 = fadd double %call1.2, %call.3 
  %call.4 = tail call double @__shfl_xor_double_nvptx(double %call1.3, i32 1)
  %call1.4 = fadd double %call1.3, %call.4 
  ret double %call1.4
}
define  double @__reduce_min_double(<1 x double>) nounwind readnone alwaysinline {
  %value = extractelement <1 x double> %0, i32 0
  %call = tail call double @__shfl_xor_double_nvptx(double %value, i32 16)
  %call1 = tail call double @__min_double(double %value, double %call) 
  %call.1 = tail call double @__shfl_xor_double_nvptx(double %call1, i32 8)
  %call1.1 = tail call double @__min_double(double %call1, double %call.1) 
  %call.2 = tail call double @__shfl_xor_double_nvptx(double %call1.1, i32 4)
  %call1.2 = tail call double @__min_double(double %call1.1, double %call.2) 
  %call.3 = tail call double @__shfl_xor_double_nvptx(double %call1.2, i32 2)
  %call1.3 = tail call double @__min_double(double %call1.2, double %call.3) 
  %call.4 = tail call double @__shfl_xor_double_nvptx(double %call1.3, i32 1)
  %call1.4 = tail call double @__min_double(double %call1.3, double %call.4) 
  ret double %call1.4
}
define  double @__reduce_max_double(<1 x double>) nounwind readnone alwaysinline {
  %value = extractelement <1 x double> %0, i32 0
  %call = tail call double @__shfl_xor_double_nvptx(double %value, i32 16)
  %call1 = tail call double @__max_double(double %value, double %call) 
  %call.1 = tail call double @__shfl_xor_double_nvptx(double %call1, i32 8)
  %call1.1 = tail call double @__max_double(double %call1, double %call.1) 
  %call.2 = tail call double @__shfl_xor_double_nvptx(double %call1.1, i32 4)
  %call1.2 = tail call double @__max_double(double %call1.1, double %call.2) 
  %call.3 = tail call double @__shfl_xor_double_nvptx(double %call1.2, i32 2)
  %call1.3 = tail call double @__max_double(double %call1.2, double %call.3) 
  %call.4 = tail call double @__shfl_xor_double_nvptx(double %call1.3, i32 1)
  %call1.4 = tail call double @__max_double(double %call1.3, double %call.4) 
  ret double %call1.4
}


;;;;;;;;; reductions int64
define  i64 @__reduce_add_int64(<1 x i64>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i64> %0, i32 0
  %call = tail call i64 @__shfl_xor_i64_nvptx(i64 %value, i32 16)
  %call1 = add i64 %call, %value 
  %call.1 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1, i32 8)
  %call1.1 =add i64 %call1, %call.1 
  %call.2 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.1, i32 4)
  %call1.2 = add i64 %call1.1, %call.2
  %call.3 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.2, i32 2)
  %call1.3 = add i64 %call1.2, %call.3 
  %call.4 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.3, i32 1)
  %call1.4 = add i64 %call1.3, %call.4 
  ret i64 %call1.4
}
define  i64 @__reduce_min_int64(<1 x i64>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i64> %0, i32 0
  %call = tail call i64 @__shfl_xor_i64_nvptx(i64 %value, i32 16)
  %call1 = tail call i64 @__min_i64_signed(i64 %value, i64 %call) 
  %call.1 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1, i32 8)
  %call1.1 = tail call i64 @__min_i64_signed(i64 %call1, i64 %call.1) 
  %call.2 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.1, i32 4)
  %call1.2 = tail call i64 @__min_i64_signed(i64 %call1.1, i64 %call.2) 
  %call.3 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.2, i32 2)
  %call1.3 = tail call i64 @__min_i64_signed(i64 %call1.2, i64 %call.3) 
  %call.4 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.3, i32 1)
  %call1.4 = tail call i64 @__min_i64_signed(i64 %call1.3, i64 %call.4) 
  ret i64 %call1.4
}
define  i64 @__reduce_max_int64(<1 x i64>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i64> %0, i32 0
  %call = tail call i64 @__shfl_xor_i64_nvptx(i64 %value, i32 16)
  %call1 = tail call i64 @__max_i64_signed(i64 %value, i64 %call) 
  %call.1 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1, i32 8)
  %call1.1 = tail call i64 @__max_i64_signed(i64 %call1, i64 %call.1) 
  %call.2 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.1, i32 4)
  %call1.2 = tail call i64 @__max_i64_signed(i64 %call1.1, i64 %call.2) 
  %call.3 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.2, i32 2)
  %call1.3 = tail call i64 @__max_i64_signed(i64 %call1.2, i64 %call.3) 
  %call.4 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.3, i32 1)
  %call1.4 = tail call i64 @__max_i64_signed(i64 %call1.3, i64 %call.4) 
  ret i64 %call1.4
}
define  i64 @__reduce_min_uint64(<1 x i64>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i64> %0, i32 0
  %call = tail call i64 @__shfl_xor_i64_nvptx(i64 %value, i32 16)
  %call1 = tail call i64 @__min_i64_unsigned(i64 %value, i64 %call) 
  %call.1 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1, i32 8)
  %call1.1 = tail call i64 @__min_i64_unsigned(i64 %call1, i64 %call.1) 
  %call.2 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.1, i32 4)
  %call1.2 = tail call i64 @__min_i64_unsigned(i64 %call1.1, i64 %call.2) 
  %call.3 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.2, i32 2)
  %call1.3 = tail call i64 @__min_i64_unsigned(i64 %call1.2, i64 %call.3) 
  %call.4 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.3, i32 1)
  %call1.4 = tail call i64 @__min_i64_unsigned(i64 %call1.3, i64 %call.4) 
  ret i64 %call1.4
}
define  i64 @__reduce_max_uint64(<1 x i64>) nounwind readnone alwaysinline {
  %value = extractelement <1 x i64> %0, i32 0
  %call = tail call i64 @__shfl_xor_i64_nvptx(i64 %value, i32 16)
  %call1 = tail call i64 @__max_i64_unsigned(i64 %value, i64 %call) 
  %call.1 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1, i32 8)
  %call1.1 = tail call i64 @__max_i64_unsigned(i64 %call1, i64 %call.1) 
  %call.2 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.1, i32 4)
  %call1.2 = tail call i64 @__max_i64_unsigned(i64 %call1.1, i64 %call.2) 
  %call.3 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.2, i32 2)
  %call1.3 = tail call i64 @__max_i64_unsigned(i64 %call1.2, i64 %call.3) 
  %call.4 = tail call i64 @__shfl_xor_i64_nvptx(i64 %call1.3, i32 1)
  %call1.4 = tail call i64 @__max_i64_unsigned(i64 %call1.3, i64 %call.4) 
  ret i64 %call1.4
}

;;;; reduce equal
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

;;;;;;;;;;; shuffle
define(`shuffle1', `
define <1 x $1> @__shuffle_$1(<1 x $1>, <1 x i32>) nounwind readnone alwaysinline 
{
  %val  = extractelement <1 x $1> %0, i32 0
  %lane = extractelement <1 x i32> %1, i32 0
  %rets = tail call $1 @__shfl_$1_nvptx($1 %val, i32 %lane)
  %retv = insertelement <1 x $1> undef, $1 %rets, i32 0
  ret <1 x $1> %retv
}
')
shuffle1(i8)
shuffle1(i16)
shuffle1(i32)
shuffle1(i64)
shuffle1(float)
shuffle1(double)

define(`shuffle2',`
define <1 x $1> @__shuffle2_$1(<1 x $1>, <1 x $1>, <1 x i32>) nounwind readnone alwaysinline
{
  %val1 = extractelement <1 x $1> %0, i32 0
  %val2 = extractelement <1 x $1> %1, i32 0
  %lane = extractelement <1 x i32> %2, i32 0
  %c    = icmp slt i32 %lane, 32              
  %val  = select i1 %c, $1 %val1, $1 %val2
  %lane_mask = and i32 %lane, 31
  %rets = tail call $1 @__shfl_$1_nvptx($1 %val, i32 %lane_mask);
  %retv = insertelement <1 x $1> undef, $1 %rets, i32 0
  ret <1 x $1> %retv
}
')
shuffle2(i8)
shuffle2(i16)
shuffle2(i32)
shuffle2(i64)
shuffle2(float)
shuffle2(double)

define(`shift',`
define <1 x $1> @__shift_$1(<1 x $1>, i32) nounwind readnone alwaysinline
{
  %val  = extractelement <1 x $1> %0, i32 0
  %tid  = tail call i32 @__tid_x()
  %lane = and i32 %tid,  31
  %src  = add i32 %lane, %1
  %ret  = tail call $1 @__shfl_$1_nvptx($1 %val, i32 %src)
  %c1   = icmp sge i32 %src, 0
  %c2   = icmp slt i32 %src, 32
  %c    = and i1 %c1, %c2
  %rets = select i1 %c, $1 %ret, $1 zeroinitializer
  %retv = insertelement <1 x $1> undef, $1 %rets, i32 0
  ret <1 x $1> %retv
}
')
shift(i8)
shift(i16)
shift(i32)
shift(i64)
shift(float)
shift(double)

define(`rotate', `
define <1 x $1> @__rotate_$1(<1 x $1>, i32) nounwind readnone alwaysinline 
{
  %val  = extractelement <1 x $1> %0, i32 0
  %tid  = tail call i32 @__tid_x()
  %src  = add i32 %tid, %1
  %lane = and i32 %src, 31
  %rets = tail call $1 @__shfl_$1_nvptx($1 %val, i32 %lane)
  %retv = insertelement <1 x $1> undef, $1 %rets, i32 0
  ret <1 x $1> %retv
}
')
rotate(i8)
rotate(i16)
rotate(i32)
rotate(i64)
rotate(float)
rotate(double)

define(`broadcast', `
define <1 x $1> @__broadcast_$1(<1 x $1>, i32) nounwind readnone alwaysinline 
{
  %val  = extractelement <1 x $1> %0, i32 0
  %rets = tail call $1 @__shfl_$1_nvptx($1 %val, i32 %1)
  %retv = insertelement <1 x $1> undef, $1 %rets, i32 0
  ret <1 x $1> %retv
}
')
broadcast(i8)
broadcast(i16)
broadcast(i32)
broadcast(i64)
broadcast(float)
broadcast(double)

define i32 @__shfl_scan_add_step_i32(i32 %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call i32 asm sideeffect  
      "{.reg .u32 r0;
       .reg .pred p;
       shfl.up.b32 r0|p, $1, $2, 0;
       @p add.u32 r0, r0, $3;
       mov.u32 $0, r0;
       }", "=r,r,r,r"(i32 %partial, i32 %up_offset, i32 %partial) nounwind readnone alwaysinline
  ret i32 %result;
}
define <1 x i32> @__exclusive_scan_add_i32(<1 x i32>, <1 x i1>) nounwind readnone alwaysinline
{
  %v0   = extractelement <1 x i32> %0, i32 0
  %mask = extractelement <1 x i1 > %1, i32 0
  %v    = select i1 %mask, i32 %v0, i32 0
  
  %s1 = tail call i32 @__shfl_scan_add_step_i32(i32 %v,  i32  1);
  %s2 = tail call i32 @__shfl_scan_add_step_i32(i32 %s1, i32  2);
  %s3 = tail call i32 @__shfl_scan_add_step_i32(i32 %s2, i32  4);
  %s4 = tail call i32 @__shfl_scan_add_step_i32(i32 %s3, i32  8);
  %s5 = tail call i32 @__shfl_scan_add_step_i32(i32 %s4, i32 16);
  %rets = sub i32 %s5, %v
  %retv = insertelement <1 x i32> undef, i32 %rets, i32 0
  ret <1 x i32> %retv
}
;;
define i32 @__shfl_scan_or_step_i32(i32 %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call i32 asm sideeffect  
      "{.reg .u32 r0;
       .reg .pred p;
       shfl.up.b32 r0|p, $1, $2, 0;
       @p or.b32 r0, r0, $3;
       mov.u32 $0, r0;
       }", "=r,r,r,r"(i32 %partial, i32 %up_offset, i32 %partial) nounwind readnone alwaysinline
  ret i32 %result;
}
define <1 x i32> @__exclusive_scan_or_i32(<1 x i32>, <1 x i1>) nounwind readnone alwaysinline
{
  %shft = tail call <1 x i32> @__shift_i32(<1 x i32> %0, i32 -1)
  %v0   = extractelement <1 x i32> %shft, i32 0
  %mask = extractelement <1 x i1 > %1, i32 0
  %v    = select i1 %mask, i32 %v0, i32 0
  
  %s1 = tail call i32 @__shfl_scan_or_step_i32(i32 %v,  i32  1);
  %s2 = tail call i32 @__shfl_scan_or_step_i32(i32 %s1, i32  2);
  %s3 = tail call i32 @__shfl_scan_or_step_i32(i32 %s2, i32  4);
  %s4 = tail call i32 @__shfl_scan_or_step_i32(i32 %s3, i32  8);
  %s5 = tail call i32 @__shfl_scan_or_step_i32(i32 %s4, i32 16);
  %retv = insertelement <1 x i32> undef, i32 %s5, i32 0
  ret <1 x i32> %retv
}
;;
define i32 @__shfl_scan_and_step_i32(i32 %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call i32 asm sideeffect  
      "{.reg .u32 r0;
       .reg .pred p;
       shfl.up.b32 r0|p, $1, $2, 0;
       @p and.b32 r0, r0, $3;
       mov.u32 $0, r0;
       }", "=r,r,r,r"(i32 %partial, i32 %up_offset, i32 %partial) nounwind readnone alwaysinline
  ret i32 %result;
}
define <1 x i32> @__exclusive_scan_and_i32(<1 x i32>, <1 x i1>) nounwind readnone alwaysinline
{
  %shft = tail call <1 x i32> @__shift_i32(<1 x i32> %0, i32 -1)
  %v0   = extractelement <1 x i32> %shft, i32 0
  %m0   = extractelement <1 x i1 > %1,    i32 0

  %tid  = tail call i32 @__tid_x()
  %lane = and i32 %tid, 31
  %m1   = icmp eq i32 %lane, 0

  %mask = and i1 %m0, %m1
  %v    = select i1 %mask, i32 %v0, i32 -1
  
  %s1 = tail call i32 @__shfl_scan_and_step_i32(i32 %v,  i32  1);
  %s2 = tail call i32 @__shfl_scan_and_step_i32(i32 %s1, i32  2);
  %s3 = tail call i32 @__shfl_scan_and_step_i32(i32 %s2, i32  4);
  %s4 = tail call i32 @__shfl_scan_and_step_i32(i32 %s3, i32  8);
  %s5 = tail call i32 @__shfl_scan_and_step_i32(i32 %s4, i32 16);
  %retv = insertelement <1 x i32> undef, i32 %s5, i32 0
  ret <1 x i32> %retv
}

define float @__shfl_scan_add_step_float(float %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call float asm sideeffect  
      "{.reg .f32 f0;
       .reg .pred p;
       shfl.up.b32 f0|p, $1, $2, 0;
       @p add.f32 f0, f0, $3;
       mov.f32 $0, f0;
       }", "=f,f,r,f"(float %partial, i32 %up_offset, float %partial) nounwind readnone alwaysinline
  ret float %result;
}
define <1 x float> @__exclusive_scan_add_float(<1 x float>, <1 x i1>) nounwind readnone alwaysinline
{
  %v0   = extractelement <1 x float> %0, i32 0
  %mask = extractelement <1 x i1 > %1, i32 0
  %v    = select i1 %mask, float %v0, float zeroinitializer

  %s1 = tail call float @__shfl_scan_add_step_float(float %v,  i32  1);
  %s2 = tail call float @__shfl_scan_add_step_float(float %s1, i32  2);
  %s3 = tail call float @__shfl_scan_add_step_float(float %s2, i32  4);
  %s4 = tail call float @__shfl_scan_add_step_float(float %s3, i32  8);
  %s5 = tail call float @__shfl_scan_add_step_float(float %s4, i32 16);
  %rets = fsub float %s5, %v
  %retv = insertelement <1 x float> undef, float %rets, i32 0
  ret <1 x float> %retv
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

