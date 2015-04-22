;;  Copyright (c) 2014-2015, Intel Corporation
;;  All rights reserved.
;;
;;  Redistribution and use in source and binary forms, with or without
;;  modification, are permitted provided that the following conditions are
;;  met:
;;
;;    * Redistributions of source code must retain the above copyright
;;      notice, this list of conditions and the following disclaimer.
;;
;;    * Redistributions in binary form must reproduce the above copyright
;;      notice, this list of conditions and the following disclaimer in the
;;      documentation and/or other materials provided with the distribution.
;;
;;    * Neither the name of Intel Corporation nor the names of its
;;      contributors may be used to endorse or promote products derived from
;;      this software without specific prior written permission.
;;
;;
;;   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
;;   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
;;   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
;;   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
;;   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
;;   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
;;   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
;;   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
;;   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;;   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;;   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

;;;;;;;;;;

define i32 @__program_index()  nounwind readnone alwaysinline
{
 %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
 %program_index = and i32 %tid, 31
 ret i32 %program_index
}
define i32 @__program_count()  nounwind readnone alwaysinline
{
;; %tid = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
;; ret i32 %tid
  ret i32 32
}
define i32 @__warp_index() nounwind readnone alwaysinline
{
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %warp_index = lshr i32 %tid, 5
  ret i32 %warp_index
}

;;;;;;;;;;;;

define i32 @__task_index0()  nounwind readnone alwaysinline
{
 %bid  = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
 %bid4 = shl i32 %bid, 2
 %warp_index = call i32 @__warp_index()
 %task_index0 = add i32 %bid4, %warp_index
 ret i32 %task_index0
}
define i32 @__task_index1()  nounwind readnone alwaysinline
{
 %task_index1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
 ret i32 %task_index1
}
define i32 @__task_index2()  nounwind readnone alwaysinline
{
 %task_index2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
 ret i32 %task_index2
}
define i32 @__task_index()  nounwind readnone alwaysinline
{
  %ti0 = call i32 @__task_index0()
  %ti1 = call i32 @__task_index1()
  %ti2 = call i32 @__task_index2()
  %tc0 = call i32 @__task_count0()
  %tc1 = call i32 @__task_count1()
  %mul1 = mul i32 %tc1, %ti2
  %add1 = add i32 %mul1, %ti1
  %mul2 = mul i32 %add1, %tc0
  %task_index = add i32 %mul2, %ti0
  ret i32 %task_index
}

;;;;;

define i32 @__task_count0()  nounwind readnone alwaysinline
{
 %nb = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
 %task_count0 = shl i32 %nb, 2
 ret i32 %task_count0
}
define i32 @__task_count1()  nounwind readnone alwaysinline
{
 %task_count1 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
 ret i32 %task_count1
}
define i32 @__task_count2()  nounwind readnone alwaysinline
{
 %task_count2 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
 ret i32 %task_count2
}
define i32 @__task_count()  nounwind readnone alwaysinline
{
  %tc0 = call i32 @__task_count0()
  %tc1 = call i32 @__task_count1()
  %tc2 = call i32 @__task_count2()
  %mul1 = mul i32 %tc1, %tc2
  %task_count = mul i32 %mul1, %tc0
  ret i32 %task_count
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
define internal i32 @__shfl_i32_nvptx(i32, i32) nounwind readnone alwaysinline
{
  %shfl = tail call i32 asm sideeffect "shfl.idx.b32  $0, $1, $2, 0x1f;", "=r,r,r"(i32 %0, i32 %1) 
  ret i32 %shfl
}
define internal i32 @__shfl_xor_i32_nvptx(i32, i32) nounwind readnone alwaysinline
{
  %shfl = tail call i32 asm sideeffect "shfl.bfly.b32  $0, $1, $2, 0x1f;", "=r,r,r"(i32 %0, i32 %1) 
  ret i32 %shfl
}
;; float
define internal float @__shfl_float_nvptx(float, i32) nounwind readnone alwaysinline
{
  %shfl = tail call float asm sideeffect "shfl.idx.b32  $0, $1, $2, 0x1f;", "=f,f,r"(float %0, i32 %1)
  ret float %shfl
}
define internal float @__shfl_xor_float_nvptx(float, i32) nounwind readnone alwaysinline
{
  %shfl = tail call float asm sideeffect "shfl.bfly.b32  $0, $1, $2, 0x1f;", "=f,f,r"(float %0, i32 %1) 
  ret float %shfl
}

;;;;;;;;;;; min/max
;; float/double
define internal float @__fminf_nvptx(float,float) nounwind readnone alwaysinline
{
  %min = tail call float asm sideeffect "min.f32 $0, $1, $2;", "=f,f,f"(float %0, float %1)
  ret float %min
}
define internal float @__fmaxf_nvptx(float,float) nounwind readnone alwaysinline
{
  %max = tail call float asm sideeffect "max.f32 $0, $1, $2;", "=f,f,f"(float %0, float %1)
  ret float %max
}

;; int
define(`int_minmax',`
define internal $1 @__min_$1_signed($1,$1) nounwind readnone alwaysinline {
  %c = icmp slt $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
define internal $1 @__max_$1_signed($1,$1) nounwind readnone alwaysinline {
  %c = icmp sgt $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
define internal $1 @__min_$1_unsigned($1,$1) nounwind readnone alwaysinline  {
  %c = icmp ult $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
define internal $1 @__max_$1_unsigned($1,$1) nounwind readnone alwaysinline {
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
define internal $1 @__min_$1($1,$1) nounwind readnone alwaysinline {
  %c = fcmp olt $1 %0, %1
  %r = select i1 %c, $1 %0, $1 %1
  ret $1 %r
}
define internal $1 @__max_$1($1,$1) nounwind readnone alwaysinline {
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
define internal $2 @$1_$2_nvptx($2, i32) nounwind readnone alwaysinline
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
define internal $2 @$1_$2_nvptx($2, i32) nounwind readnone alwaysinline
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
define internal i32 @__ballot_nvptx(i1) nounwind readnone alwaysinline
{
  %conv = zext i1 %0 to i32
  %res = tail call i32 asm sideeffect 
      "{ .reg .pred %p1; 
         setp.ne.u32 %p1, $1, 0; 
         vote.ballot.b32  $0, %p1; 
      }", "=r,r"(i32 %conv) 
  ret i32 %res
}
define internal i32 @__lanemask_lt_nvptx() nounwind readnone alwaysinline
{
  %mask = tail call i32 asm sideeffect "mov.u32 $0, %lanemask_lt;", "=r"() 
  ret i32 %mask
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; tasking

;; this call allocate parameter buffer for kernel launch
declare i64 @cudaGetParameterBuffer(i64, i64) nounwind
define i8* @ISPCAlloc(i8**, i64 %size, i32 %align32) nounwind alwaysinline
{
entry:
  %and = call i32 @__program_index()
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
  %and = call i32 @__program_index()
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
rdrand_decls()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; broadcast/rotate/shuffle

define_shuffles()


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; aos/soa

aossoa()

;; dummy 1 wide vector ops
declare  void
@__aos_to_soa4_float1(<1 x float> %v0, <1 x float> %v1, <1 x float> %v2,
        <1 x float> %v3, <1 x float> * noalias %out0, 
        <1 x float> * noalias %out1, <1 x float> * noalias %out2, 
        <1 x float> * noalias %out3) nounwind alwaysinline ;

declare  void
@__soa_to_aos4_float1(<1 x float> %v0, <1 x float> %v1, <1 x float> %v2,
        <1 x float> %v3, <1 x float> * noalias %out0, 
        <1 x float> * noalias %out1, <1 x float> * noalias %out2, 
        <1 x float> * noalias %out3) nounwind alwaysinline ;

declare  void
@__aos_to_soa3_float1(<1 x float> %v0, <1 x float> %v1,
         <1 x float> %v2, <1 x float> * %out0, <1 x float> * %out1,
         <1 x float> * %out2);

declare  void
@__soa_to_aos3_float1(<1 x float> %v0, <1 x float> %v1,
         <1 x float> %v2, <1 x float> * %out0, <1 x float> * %out1,
         <1 x float> * %out2);

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; half conversion routines

declare float @llvm.convert.from.fp16(i16) nounwind readnone
declare i16   @llvm.convert.to.fp16(float) nounwind readnone
define float @__half_to_float_uniform(i16 %v) nounwind readnone alwaysinline
{
  ;; %res = call float @llvm.convert.from.fp16(i16 %v)
  %res = tail call float asm sideeffect 
      "{ .reg .f16 tmp; 
        mov.b16 tmp, $1;
        cvt.f32.f16 $0, tmp;
     }", "=f,h"(i16 %v) 
  ret float %res
}
define i16 @__float_to_half_uniform(float %v) nounwind readnone alwaysinline
{
 ;; this will break the compiler, use inline asm similarly to above case
 ;; %half = call i16 @llvm.convert.to.fp16(float %v)
  %half = tail call i16 asm sideeffect 
      "{ .reg .f16 tmp; 
        cvt.rn.f16.f32 tmp, $1;
        mov.b16 $0, tmp;
     }", "=h,f"(float %v) 
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

define internal float @__round_uniform_float_ptx(float) nounwind readnone alwaysinline
{
  %2 = tail call float asm sideeffect
        "{ .reg .pred p<3>; .reg .s32 r<4>; .reg .f32 f<10>;
           mov.f32 f4, $1;
           abs.f32 f5, f4;
           mov.b32 r1, f4;
           and.b32 r2, r1, -2147483648;
           or.b32  r3, r2, 1056964608;
           mov.b32 f6, r3;
           add.f32 f7, f6, f4;
           cvt.rzi.f32.f32	f8, f7;
           setp.gt.f32	p1, f5, 0f4B000000;
           selp.f32	f9, f4, f8, p1;
           setp.geu.f32	p2, f5, 0f3F000000;
           @p2 bra BB2_2;
           cvt.rzi.f32.f32	f9, f4;
BB2_2:
           mov.f32 $0, f9;
        }", "=f,f"(float %0) 
  ret float %2
}
define  float @__round_uniform_float(float) nounwind readonly alwaysinline {
  %float_to_int_bitcast.i.i.i.i = bitcast float %0 to <1 x i32>
  %bitop.i.i = and <1 x i32> %float_to_int_bitcast.i.i.i.i, <i32 -2147483648>
  %bitop.i = xor <1 x i32> %float_to_int_bitcast.i.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i40.i = bitcast <1 x i32> %bitop.i to <1 x float>
  %binop.i = fadd <1 x float> %int_to_float_bitcast.i.i40.i, <float 8.388608e+06>
  %binop21.i = fadd <1 x float> %binop.i, <float -8.388608e+06>
  %float_to_int_bitcast.i.i.i = bitcast <1 x float> %binop21.i to <1 x i32>
  %bitop31.i = xor <1 x i32> %float_to_int_bitcast.i.i.i, %bitop.i.i
  %int_to_float_bitcast.i.i.i = bitcast <1 x i32> %bitop31.i to float
  ret float %int_to_float_bitcast.i.i.i
}
define float @__floor_uniform_float(float) nounwind readnone alwaysinline
{
  %2 = tail call float asm sideeffect "cvt.rmi.f32.f32 $0, $1;", "=f,f"(float %0) 
  ret float %2
}
define float @__ceil_uniform_float(float) nounwind readnone alwaysinline
{
  %2 = tail call float asm sideeffect "cvt.rpi.f32.f32 $0, $1;", "=f,f"(float %0)
  ret float %2
}

define double @__round_uniform_double(double) nounwind readnone alwaysinline
{
  %2 = tail call double asm sideeffect
        "{ 
          .reg .pred 	p<3>;
          .reg .s32 	r<6>;
          .reg .f64 	fd<9>;

          mov.f64 	fd8, $1
          abs.f64 	fd1, fd8;
          setp.ge.f64	p1, fd1, 0d4330000000000000;
          @p1 bra 	BB5_2;

          add.f64 	fd5, fd1, 0d3FE0000000000000;
          cvt.rzi.f64.f64	fd6, fd5;
          setp.lt.f64	p2, fd1, 0d3FE0000000000000;
          selp.f64	fd7, 0d0000000000000000, fd6, p2;
          {
            .reg .b32 temp; 
            mov.b64 	{r1, temp}, fd7;
          }
          {
            .reg .b32 temp; 
            mov.b64 	{temp, r2}, fd7;
          }
          {
            .reg .b32 temp; 
            mov.b64 	{temp, r3}, fd8;
          }
          and.b32  	r4, r3, -2147483648;
          or.b32  	r5, r2, r4;
          mov.b64 	fd8, {r1, r5};

BB5_2:
          mov.f64	$0, fd8;
        }", "=d,d"(double %0)
  ret double %2
}
define double @__floor_uniform_double(double) nounwind readnone alwaysinline
{
  %2 = tail call double asm sideeffect "cvt.rmi.f64.f64 $0, $1;", "=f,f"(double %0)
  ret double %2
}
define double @__ceil_uniform_double(double) nounwind readnone alwaysinline
{
  %2 = tail call double asm sideeffect "cvt.rpi.f64.f64 $0, $1;", "=f,f"(double %0)
  ret double %2
}

define  internal <1 x float> @__floor_varying_floatX(<1 x float>) nounwind readonly alwaysinline {
  %calltmp.i = tail call <1 x float> @__round_varying_float(<1 x float> %0) nounwind
  %bincmp.i = fcmp ogt <1 x float> %calltmp.i, %0
  %val_to_boolvec32.i = sext <1 x i1> %bincmp.i to <1 x i32>
  %bitop.i = and <1 x i32> %val_to_boolvec32.i, <i32 -1082130432>
  %int_to_float_bitcast.i.i.i = bitcast <1 x i32> %bitop.i to <1 x float>
  %binop.i = fadd <1 x float> %calltmp.i, %int_to_float_bitcast.i.i.i
  ret <1 x float> %binop.i
}

define(`rfc_varying',`
define <1 x $2> @__$1_varying_$2(<1 x $2>) nounwind readonly alwaysinline
{
   %val = extractelement <1 x $2> %0, i32 0
   %res = call $2 @__$1_uniform_$2($2 %val)
   %ret = insertelement <1 x $2> undef, $2 %res, i32 0
   ret <1 x $2> %ret
}
')
rfc_varying(round, float)
rfc_varying(floor, float)
rfc_varying(ceil,  float)
rfc_varying(round, double)
rfc_varying(floor, double)
rfc_varying(ceil,  double)

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
define  internal i64 @__min_uniform_int64X(i64, i64) nounwind readonly alwaysinline {
  %c = icmp slt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}
define  internal i64 @__max_uniform_int64X(i64, i64) nounwind readonly alwaysinline {
  %c = icmp sgt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

;; declare i64 @__min_uniform_uint64(i64, i64) nounwind readnone 
;; declare i64 @__max_uniform_uint64(i64, i64) nounwind readnone 
define  internal i64 @__min_uniform_uint64X(i64, i64) nounwind readonly alwaysinline {
  %c = icmp ult i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}
define  internal i64 @__max_uniform_uint64X(i64, i64) nounwind readonly alwaysinline {
  %c = icmp ugt i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

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


define(`minmax_vy',`
define <1 x $2> @__$1_varying_$3(<1 x $2>, <1 x $2>) nounwind readnone alwaysinline
{
  %v0 = extractelement <1 x $2> %0, i32 0
  %v1 = extractelement <1 x $2> %1, i32 0
  %r = call $2 @__$1_uniform_$3($2 %v0, $2 %v1)
  %ret = insertelement <1 x $2> undef, $2 %r, i32 0
  ret <1 x $2> %ret;
}
')
minmax_vy(min, i32,  int32)
minmax_vy(max, i32,  int32)
minmax_vy(min, i32, uint32)
minmax_vy(max, i32, uint32)
minmax_vy(min, float, float)
minmax_vy(max, float, float)
minmax_vy(min, double, double)
minmax_vy(max, double, double)

;; sqrt/rsqrt/rcp

define  float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  %ret = fdiv float 1.,%0
  ret float %ret
}
declare double @__nv_drcp_rn(double)
define  double @__rcp_uniform_double(double) nounwind readonly alwaysinline 
{
  %ret  = call double @__nv_drcp_rn(double %0)
  ret double %ret
}
declare float @__nv_sqrtf(float)
define  float @__sqrt_uniform_float(float) nounwind readonly alwaysinline 
{
  %ret = call float @__nv_sqrtf(float %0)
  ret float %ret
}
declare double @__nv_sqrt(double)
define  double @__sqrt_uniform_double(double) nounwind readonly alwaysinline {
  %ret = call double @__nv_sqrt(double %0)
  ret double %ret
}
declare float @__nv_rsqrtf(float)
define  float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline 
{
  %ret = call float @__nv_rsqrtf(float %0)
  ret float %ret
}
declare double @__nv_rsqrt(double)
define  double @__rsqrt_uniform_double(double) nounwind readonly alwaysinline 
{
  %ret = call double @__nv_rsqrt(double %0)
  ret double %ret
}

;;;;;; varying
define <WIDTH x float> @__rcp_varying_float(<WIDTH x float>) nounwind readnone  alwaysinline
{
  %v = extractelement <1 x float> %0, i32 0
  %r = call float @__rcp_uniform_float(float %v)
  %rv = insertelement <1 x float> undef, float %r, i32 0 
  ret <WIDTH x float> %rv
}
define <WIDTH x double> @__rcp_varying_double(<WIDTH x double>) nounwind readnone  alwaysinline
{
  %v = extractelement <1 x double> %0, i32 0
  %r = call double @__rcp_uniform_double(double %v)
  %rv = insertelement <1 x double> undef, double %r, i32 0 
  ret <WIDTH x double> %rv
}
define <WIDTH x float> @__rsqrt_varying_float(<WIDTH x float>) nounwind readnone alwaysinline
{
  %v = extractelement <1 x float> %0, i32 0
  %r = call float @__rsqrt_uniform_float(float %v)
  %rv = insertelement <1 x float> undef, float %r, i32 0 
  ret <WIDTH x float> %rv
}
define <WIDTH x double> @__rsqrt_varying_double(<WIDTH x double>) nounwind readnone alwaysinline
{
  %v = extractelement <1 x double> %0, i32 0
  %r = call double @__rsqrt_uniform_double(double %v)
  %rv = insertelement <1 x double> undef, double %r, i32 0 
  ret <WIDTH x double> %rv
}
define <WIDTH x float> @__sqrt_varying_float(<WIDTH x float>) nounwind readnone alwaysinline
{
  %v = extractelement <1 x float> %0, i32 0
  %r = call float @__sqrt_uniform_float(float %v)
  %rv = insertelement <1 x float> undef, float %r, i32 0 
  ret <WIDTH x float> %rv
}
define <WIDTH x double> @__sqrt_varying_double(<WIDTH x double>) nounwind readnone alwaysinline
{
  %v = extractelement <1 x double> %0, i32 0
  %r = call double @__sqrt_uniform_double(double %v)
  %rv = insertelement <1 x double> undef, double %r, i32 0 
  ret <WIDTH x double> %rv
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; population count

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
define  i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
 %call = call i32 @llvm.ctpop.i32(i32 %0)
 ret i32 %call
;;  %res = tail call i32 asm sideeffect "popc.b32 $0, $1;", "=r,r"(i32 %0)
 ;; ret i32 %res
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone
define  i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %call = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; binary prefix sum

define internal i64 @__warpBinExclusiveScan(i1 %p) nounwind readonly alwaysinline 
{
entry:
  %call  = call i32 @__ballot_nvptx(i1 zeroext %p)
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

;; svml is not support in PTX, will generate linking error

include(`svml.m4')
svml_stubs(float,f,WIDTH)
svml_stubs(double,d,WIDTH)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reductions

define  i64 @__movmsk(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
  %v64 = zext i1 %v to i64
  ret i64 %v64
}
define  i64 @__movmsk_ptx(<1 x i1>) nounwind readnone alwaysinline {
  %v = extractelement <1 x i1> %0, i32 0
   %v0  = call i32 @__ballot_nvptx(i1 %v)
   %v64 = zext i32 %v0 to i64
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
  %res0 = call i32 @__ballot_nvptx(i1 %v)
  %cmp = icmp eq i32 %res0, -1
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

;;;; reduce equal, must be tested and may fail if data has -1
define internal i32 @__shfl_reduce_and_step_i32_nvptx(i32, i32) nounwind readnone alwaysinline
{
  %shfl = tail call i32 asm sideeffect
      "{.reg .u32 r0; 
        .reg .pred p;
        shfl.bfly.b32  r0|p, $1, $2, 0;
        @p and.b32 r0, r0, $3;
        mov.u32 $0, r0;
      }", "=r,r,r,r"(i32 %0, i32 %1, i32 %0)
  ret i32 %shfl
}
shfl64(__shfl_reduce_and_step, i64)

define internal i32 @__reduce_and_i32(i32 %v0, i1 %mask) nounwind readnone alwaysinline
{
  %v  = select i1 %mask, i32 %v0, i32 -1
  %s1 = tail call i32 @__shfl_reduce_and_step_i32_nvptx(i32 %v,  i32 16);
  %s2 = tail call i32 @__shfl_reduce_and_step_i32_nvptx(i32 %s1, i32  8);
  %s3 = tail call i32 @__shfl_reduce_and_step_i32_nvptx(i32 %s2, i32  4);
  %s4 = tail call i32 @__shfl_reduce_and_step_i32_nvptx(i32 %s3, i32  2);
  %s5 = tail call i32 @__shfl_reduce_and_step_i32_nvptx(i32 %s4, i32  1);
  ret i32 %s5
}
define internal i64 @__reduce_and_i64(i64, i1) nounwind readnone alwaysinline
{
  %v   = bitcast i64 %0 to <2 x i32>
  %v0  = extractelement <2 x i32> %v, i32 0
  %v1  = extractelement <2 x i32> %v, i32 1
  %s0  = call i32 @__reduce_and_i32(i32 %v0, i1 %1)
  %s1  = call i32 @__reduce_and_i32(i32 %v1, i1 %1)
  %tmp = insertelement <2 x i32> undef, i32 %s0, i32 0
  %res = insertelement <2 x i32> %tmp,  i32 %s1, i32 1
  %ret = bitcast <2 x i32> %res to i64
  ret i64 %ret;
}

define(`reduce_equal',`
define i1 @__reduce_equal_$2(<1 x $1> %v0, $1 * %samevalue, <1 x i1> %maskv) nounwind alwaysinline
{
entry:
  %vv = bitcast <1 x $1> %v0 to <1 x $3>
  %sv = extractelement <1 x $3> %vv, i32 0
  %mask = extractelement <1 x i1> %maskv, i32 0

  %s = call $3 @__reduce_and_$3($3 %sv, i1 %mask);

  ;; find last active lane 
  %nact  = call i32 @__ballot_nvptx(i1 %mask)
  %lane1 = call i32 @__count_leading_zeros_i32(i32 %nact)
  %lane  = sub i32 31, %lane1

  ;; broadcast result from this lane
  %r = tail call $3 @__shfl_$3_nvptx($3 %s, i32 %lane)

  ;; compare result to the original value
  %c0  = icmp eq $3 %r, %sv
  %c1  = and i1 %c0, %mask
  %neq = call i32 @__ballot_nvptx(i1 %c1)
  %cmp = icmp eq i32 %neq, %nact

  br i1 %cmp, label %all_equal, label %all_not_equal
  
all_equal:
  %vstore = bitcast $3 %r to $1 
  store $1 %vstore, $1* %samevalue;
  ret i1 true

all_not_equal:
  ret i1 false

}
')
reduce_equal(i32,    int32, i32);
reduce_equal(i64,    int64, i64);
reduce_equal(float,  float, i32);
reduce_equal(double, double, i64);

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
  %val1 = extractelement <1 x  $1> %0, i32 0
  %val2 = extractelement <1 x  $1> %1, i32 0

  ;; fetch both values
  %lane = extractelement <1 x i32> %2, i32 0
  %lane_mask = and i32 %lane, 31
  %ret1 = tail call $1 @__shfl_$1_nvptx($1 %val1, i32 %lane_mask);
  %ret2 = tail call $1 @__shfl_$1_nvptx($1 %val2, i32 %lane_mask);

  ;; select the correct one
  %c    = icmp slt i32 %lane, 32              
  %rets = select i1 %c, $1 %ret1, $1 %ret2
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
  %lane = call i32 @__program_index()
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
  %tid  = call i32 @__program_index()
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefix sum stuff

define internal i32 @__shfl_scan_add_step_i32(i32 %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call i32 asm sideeffect  
      "{.reg .u32 r0;
       .reg .pred p;
       shfl.up.b32 r0|p, $1, $2, 0;
       @p add.u32 r0, r0, $3;
       mov.u32 $0, r0;
       }", "=r,r,r,r"(i32 %partial, i32 %up_offset, i32 %partial)
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
define internal i32 @__shfl_scan_or_step_i32(i32 %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call i32 asm sideeffect  
      "{.reg .u32 r0;
       .reg .pred p;
       shfl.up.b32 r0|p, $1, $2, 0;
       @p or.b32 r0, r0, $3;
       mov.u32 $0, r0;
       }", "=r,r,r,r"(i32 %partial, i32 %up_offset, i32 %partial)
  ret i32 %result;
}
define <1 x i32> @__exclusive_scan_or_i32(<1 x i32>, <1 x i1>) nounwind readnone alwaysinline
{
  %v0   = extractelement <1 x i32> %0, i32 0
  %mask = extractelement <1 x i1 > %1, i32 0
  %v1   = select i1 %mask, i32 %v0, i32 0

  ;; shfl-up by one for exclusive scan
  %v = tail call i32 asm sideeffect
      "{.reg .u32 r0;
        .reg .pred p;
        shfl.up.b32 r0|p, $1, 1, 0;
        @!p mov.u32 r0, 0;
        mov.u32 $0, r0;
      }","=r,r"(i32 %v1)
  
  %s1 = tail call i32 @__shfl_scan_or_step_i32(i32 %v,  i32  1);
  %s2 = tail call i32 @__shfl_scan_or_step_i32(i32 %s1, i32  2);
  %s3 = tail call i32 @__shfl_scan_or_step_i32(i32 %s2, i32  4);
  %s4 = tail call i32 @__shfl_scan_or_step_i32(i32 %s3, i32  8);
  %s5 = tail call i32 @__shfl_scan_or_step_i32(i32 %s4, i32 16);
  %retv = insertelement <1 x i32> undef, i32 %s5, i32 0
  ret <1 x i32> %retv
}
;;
define internal i32 @__shfl_scan_and_step_i32(i32 %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = call i32 asm 
      "{.reg .u32 r0;
       .reg .pred p;
       shfl.up.b32 r0|p, $1, $2, 0;
       @p and.b32 r0, r0, $3;
       mov.u32 $0, r0;
       }", "=r,r,r,r"(i32 %partial, i32 %up_offset, i32 %partial)
  ret i32 %result;
}
define <1 x i32> @__exclusive_scan_and_i32(<1 x i32>, <1 x i1>) nounwind readnone alwaysinline
{
  %v0   = extractelement <1 x i32> %0, i32 0
  %mask = extractelement <1 x i1 > %1, i32 0
  %v1   = select i1 %mask, i32 %v0, i32 -1

  ;; shfl-up by one for exclusive scan
  %v = call i32 asm
      "{.reg .u32 r0;
        .reg .pred p;
        shfl.up.b32 r0|p, $1, 1, 0;
        @!p mov.u32 r0, -1;
        mov.u32 $0, r0;
      }","=r,r"(i32 %v1)

  %s1 = call i32 @__shfl_scan_and_step_i32(i32 %v,  i32  1);
  %s2 = call i32 @__shfl_scan_and_step_i32(i32 %s1, i32  2);
  %s3 = call i32 @__shfl_scan_and_step_i32(i32 %s2, i32  4);
  %s4 = call i32 @__shfl_scan_and_step_i32(i32 %s3, i32  8);
  %s5 = call i32 @__shfl_scan_and_step_i32(i32 %s4, i32 16);
  %retv = insertelement <1 x i32> undef, i32 %s5, i32 0
  ret <1 x i32> %retv
}

define internal float @__shfl_scan_add_step_float(float %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call float asm sideeffect  
      "{.reg .f32 f0;
       .reg .pred p;
       shfl.up.b32 f0|p, $1, $2, 0;
       @p add.f32 f0, f0, $3;
       mov.f32 $0, f0;
       }", "=f,f,r,f"(float %partial, i32 %up_offset, float %partial)
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
define internal double @__shfl_scan_add_step_double(double %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call double asm sideeffect  
      "{.reg .s32 r<10>;
        .reg .f64 fd0;
       .reg .pred p;
       .reg .b32 temp;
       mov.b64 {r1,temp}, $1;
       mov.b64 {temp,r2}, $1;
       shfl.up.b32 r3,   r1, $2, 0;
       shfl.up.b32 r4|p, r2, $2, 0;
       mov.b64 fd0, {r3,r4};
       @p add.f64 fd0, fd0, $3;
       mov.f64 $0, fd0;
       }", "=d,d,r,d"(double %partial, i32 %up_offset, double %partial)
  ret double %result;
}
define <1 x double> @__exclusive_scan_add_double(<1 x double>, <1 x i1>) nounwind readnone alwaysinline
{
  %v0   = extractelement <1 x double> %0, i32 0
  %mask = extractelement <1 x i1 > %1, i32 0
  %v    = select i1 %mask, double %v0, double zeroinitializer

  %s1 = tail call double @__shfl_scan_add_step_double(double %v,  i32  1);
  %s2 = tail call double @__shfl_scan_add_step_double(double %s1, i32  2);
  %s3 = tail call double @__shfl_scan_add_step_double(double %s2, i32  4);
  %s4 = tail call double @__shfl_scan_add_step_double(double %s3, i32  8);
  %s5 = tail call double @__shfl_scan_add_step_double(double %s4, i32 16);
  %rets = fsub double %s5, %v
  %retv = bitcast double %rets to <1 x double>
  ret <1 x double> %retv
}

define internal i64 @__shfl_scan_add_step_i64(i64 %partial, i32 %up_offset) nounwind readnone alwaysinline
{
  %result = tail call i64 asm sideeffect  
      "{.reg .s32 r<10>;
        .reg .s64 rl0;
       .reg .pred p;
       .reg .b32 temp;
       mov.b64 {r1,temp}, $1;
       mov.b64 {temp,r2}, $1;
       shfl.up.b32 r3,   r1, $2, 0;
       shfl.up.b32 r4|p, r2, $2, 0;
       mov.b64 rl0, {r3,r4};
       @p add.s64 rl0, rl0, $3;
       mov.s64 $0, rl0;
       }", "=l,l,r,l"(i64 %partial, i32 %up_offset, i64 %partial) 
  ret i64 %result;
}
define <1 x i64> @__exclusive_scan_add_i64(<1 x i64>, <1 x i1>) nounwind readnone alwaysinline
{
  %v0   = extractelement <1 x i64> %0, i32 0
  %mask = extractelement <1 x i1 > %1, i32 0
  %v    = select i1 %mask, i64 %v0, i64 zeroinitializer

  %s1 = tail call i64 @__shfl_scan_add_step_i64(i64 %v,  i32  1);
  %s2 = tail call i64 @__shfl_scan_add_step_i64(i64 %s1, i32  2);
  %s3 = tail call i64 @__shfl_scan_add_step_i64(i64 %s2, i32  4);
  %s4 = tail call i64 @__shfl_scan_add_step_i64(i64 %s3, i32  8);
  %s5 = tail call i64 @__shfl_scan_add_step_i64(i64 %s4, i32 16);
  %rets = sub i64 %s5, %v
  %retv = bitcast i64 %rets to <1 x i64>
  ret <1 x i64> %retv
}

define(`exclusive_scan_i64',`
define <1 x i64> @__exclusive_scan_$1_i64(<1 x i64>, <1 x i1>) nounwind readnone alwaysinline
{
  %v = bitcast <1 x i64> %0 to <2 x i32>
  %v0 = extractelement <2 x i32> %v, i32 0
  %v1 = extractelement <2 x i32> %v, i32 1
  %inp0 = bitcast i32 %v0 to <1 x i32>
  %inp1 = bitcast i32 %v1 to <1 x i32>
  %res0 = call <1 x i32> @__exclusive_scan_$1_i32(<1 x i32> %inp0, <1 x i1> %1);
  %res1 = call <1 x i32> @__exclusive_scan_$1_i32(<1 x i32> %inp1, <1 x i1> %1);
  %r0   = bitcast <1 x i32> %res0 to i32
  %r1   = bitcast <1 x i32> %res1 to i32
  %ret0 = insertelement <2 x i32> undef, i32 %r0, i32 0
  %ret1 = insertelement <2 x i32> %ret0, i32 %r1, i32 1
  %ret  = bitcast <2 x i32> %ret1 to <1 x i64>
  ret <1 x i64> %ret
}
')
exclusive_scan_i64(or)
exclusive_scan_i64(and)

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
  %v = load PTR_OP_ARGS(`<WIDTH x i8> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i8> %1, <WIDTH x i8> %v
  store <WIDTH x i8> %v1, <WIDTH x i8> * %0
  ret void
}

define void @__masked_store_blend_i16(<WIDTH x i16>* nocapture, <WIDTH x i16>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i16> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i16> %1, <WIDTH x i16> %v
  store <WIDTH x i16> %v1, <WIDTH x i16> * %0
  ret void
}

define void @__masked_store_blend_i32(<WIDTH x i32>* nocapture, <WIDTH x i32>, 
                                      <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i32> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i32> %1, <WIDTH x i32> %v
  store <WIDTH x i32> %v1, <WIDTH x i32> * %0
  ret void
}

define void @__masked_store_blend_float(<WIDTH x float>* nocapture, <WIDTH x float>, 
                                        <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x float> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x float> %1, <WIDTH x float> %v
  store <WIDTH x float> %v1, <WIDTH x float> * %0
  ret void
}

define void @__masked_store_blend_i64(<WIDTH x i64>* nocapture,
                            <WIDTH x i64>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x i64> ')  %0
  %v1 = select <WIDTH x i1> %2, <WIDTH x i64> %1, <WIDTH x i64> %v
  store <WIDTH x i64> %v1, <WIDTH x i64> * %0
  ret void
}

define void @__masked_store_blend_double(<WIDTH x double>* nocapture,
                            <WIDTH x double>, <WIDTH x i1>) nounwind alwaysinline {
  %v = load PTR_OP_ARGS(`<WIDTH x double> ')  %0
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
define_prefetches()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; vector ops

define(`extract_insert',`
define $1 @__extract_$2(<1 x $1>, i32) nounwind readnone alwaysinline {
  %val = extractelement <1 x $1> %0, i32 0
  %extract = tail call $1 @__shfl_$1_nvptx($1 %val, i32 %1)
  ret $1 %extract
}

define <1 x $1> @__insert_$2(<1 x $1>, i32, 
                                   $1) nounwind readnone alwaysinline {
  %orig = extractelement <1 x $1> %0, i32 0
  %lane = call i32 @__program_index() 
  %c    = icmp eq i32 %lane, %1
  %val  = select i1 %c, $1 %2, $1 %orig
  %insert = insertelement <1 x $1> %0, $1 %val, i32 0
  ret <1 x $1> %insert
}
')

extract_insert(i8, int8)
extract_insert(i16, int16)
extract_insert(i32, int32)
extract_insert(i64, int64)
extract_insert(float, float)
extract_insert(double, double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; assert

declare void @__assertfail(i64,i64,i32,i64,i64) noreturn;
declare i32 @vprintf(i64,i64)
define i32 @__puts_nvptx(i8*) alwaysinline
{
  %str   = ptrtoint i8* %0 to i64
  %parm  = or i64 0, 0
  %call  = call i32 @vprintf(i64 %str, i64 %parm)
;;  %cr    = alloca <3 x i8>
;;  store <3 x i8> <i8 13, i8 10, i8 0>, <3 x i8>* %cr
;;  %cr1   = ptrtoint <3 x i8>* %cr to i64
;;  %call1 = call i32 @vprintf(i64 %cr1, i64 %parm)
  ret i32 %call;
}
define internal void @__abort_nvptx(i8* %str) noreturn
{
  %tmp1 = alloca <3 x i8>
  store <3 x i8> <i8 58, i8 58, i8 0>, <3 x i8>* %tmp1
  %tmp2 = alloca <2 x i8>
  store <2 x i8> <i8 0, i8 0>, <2 x i8>* %tmp2

  %param1 = ptrtoint <2 x i8>* %tmp2 to i64
  %param3 = or i32 0, 0
  %string = ptrtoint i8* %str to i64
  %param4 = ptrtoint <3 x i8>* %tmp1 to i64
  %param5 = or i64 1, 1
  call void @__assertfail(i64 %param1, i64 %string, i32 %param3, i64 %param4, i64 %param5);
  ret void
}

define void @__do_assert_uniform(i8 *%str, i1 %test, <WIDTH x MASK> %mask) {
  br i1 %test, label %ok, label %fail

fail:
  %lane = call i32 @__program_index()
  %cmp  = icmp eq i32 %lane, 0
  br i1 %cmp, label %fail_print, label %fail_void;
  


fail_print:
  call void @__abort_nvptx(i8* %str) noreturn
  unreachable

fail_void:
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
  call void @__abort_nvptx(i8* %str) noreturn
  unreachable

ok:
  ret void
}

define i64 @__clock() nounwind alwaysinline {
  %r = call i64 asm sideeffect "mov.b64 $0, %clock64;", "=l"();
  ret i64 %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; atomics and memory barriers

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; global_atomic_associative
;; More efficient implementation for atomics that are associative (e.g.,
;; add, and, ...).  If a basic implementation would do sometihng like:
;; result0 = atomic_op(ptr, val0)
;; result1 = atomic_op(ptr, val1)
;; ..
;; Then instead we can do:
;; tmp = (val0 op val1 op ...)
;; result0 = atomic_op(ptr, tmp)
;; result1 = (result0 op val0)
;; ..
;; And more efficiently compute the same result
;;
;; Takes five parameters:
;; $1: vector width of the target
;; $2: operation being performed (w.r.t. LLVM atomic intrinsic names)
;;     (add, sub...)
;; $3: return type of the LLVM atomic (e.g. i32)
;; $4: return type of the LLVM atomic type, in ispc naming paralance (e.g. int32)
;; $5: identity value for the operator (e.g. 0 for add, -1 for AND, ...)
;; add
define <1 x i32> @__atomic_add_int32_global(i32* %ptr, <1 x i32> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x  i1> %maskv to  i1
  %val  = bitcast <1 x i32> %valv  to i32
  br i1 %mask, label %exec, label %pass
exec:
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.add.u32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  %oldv = bitcast i32 %old to <1 x i32>
  ret <1 x i32> %oldv
pass:
  ret <1 x i32> %valv
}
;; sub
define <1 x i32> @__atomic_sub_int32_global(i32* %ptr, <1 x i32> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %nvalv = sub <1 x i32> <i32 0>, %valv
  %ret = call <1 x i32> @__atomic_add_int32_global(i32* %ptr, <1 x i32> %nvalv, <1 x i1> %maskv);
  ret <1 x i32> %ret;
}
;; and
define <1 x i32> @__atomic_and_int32_global(i32* %ptr, <1 x i32> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x  i1> %maskv to  i1
  %val  = bitcast <1 x i32> %valv  to i32
  br i1 %mask, label %exec, label %pass
exec:
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.and.b32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  %oldv = bitcast i32 %old to <1 x i32>
  ret <1 x i32> %oldv
pass:
  ret <1 x i32> %valv
}
;; or
define <1 x i32> @__atomic_or_int32_global(i32* %ptr, <1 x i32> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x  i1> %maskv to  i1
  %val  = bitcast <1 x i32> %valv  to i32
  br i1 %mask, label %exec, label %pass
exec:
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.or.b32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  %oldv = bitcast i32 %old to <1 x i32>
  ret <1 x i32> %oldv
pass:
  ret <1 x i32> %valv
}
;; xor
define <1 x i32> @__atomic_xor_int32_global(i32* %ptr, <1 x i32> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x  i1> %maskv to  i1
  %val  = bitcast <1 x i32> %valv  to i32
  br i1 %mask, label %exec, label %pass
exec:
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.xor.b32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  %oldv = bitcast i32 %old to <1 x i32>
  ret <1 x i32> %oldv
pass:
  ret <1 x i32> %valv
}

;;;;;;;;; int64
define <1 x i64> @__atomic_add_int64_global(i64* %ptr, <1 x i64> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x  i1> %maskv to  i1
  %val  = bitcast <1 x i64> %valv  to i64
  br i1 %mask, label %exec, label %pass
exec:
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.add.u64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  %oldv = bitcast i64 %old to <1 x i64>
  ret <1 x i64> %oldv
pass:
  ret <1 x i64> %valv
}
define <1 x i64> @__atomic_sub_int64_global(i64* %ptr, <1 x i64> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %nvalv = sub <1 x i64> <i64 0>, %valv
  %ret = call <1 x i64> @__atomic_add_int64_global(i64* %ptr, <1 x i64> %nvalv, <1 x i1> %maskv);
  ret <1 x i64> %ret;
}

;; and
define <1 x i64> @__atomic_and_int64_global(i64* %ptr, <1 x i64> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x  i1> %maskv to  i1
  %val  = bitcast <1 x i64> %valv  to i64
  br i1 %mask, label %exec, label %pass
exec:
  %andr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.and.b64 $0, [$1], $2;", "=l,l,l"(i64 %andr, i64 %val);
  %oldv = bitcast i64 %old to <1 x i64>
  ret <1 x i64> %oldv
pass:
  ret <1 x i64> %valv
}

;; or 
define <1 x i64> @__atomic_or_int64_global(i64* %ptr, <1 x i64> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x  i1> %maskv to  i1
  %val  = bitcast <1 x i64> %valv  to i64
  br i1 %mask, label %exec, label %pass
exec:
  %orr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.or.b64 $0, [$1], $2;", "=l,l,l"(i64 %orr, i64 %val);
  %oldv = bitcast i64 %old to <1 x i64>
  ret <1 x i64> %oldv
pass:
  ret <1 x i64> %valv
}

;; xor
define <1 x i64> @__atomic_xor_int64_global(i64* %ptr, <1 x i64> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x  i1> %maskv to  i1
  %val  = bitcast <1 x i64> %valv  to i64
  br i1 %mask, label %exec, label %pass
exec:
  %xorr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.xor.b64 $0, [$1], $2;", "=l,l,l"(i64 %xorr, i64 %val);
  %oldv = bitcast i64 %old to <1 x i64>
  ret <1 x i64> %oldv
pass:
  ret <1 x i64> %valv
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; global_atomic_uniform
;; Defines the implementation of a function that handles the mapping from
;; an ispc atomic function to the underlying LLVM intrinsics.  This variant
;; just calls the atomic once, for the given uniform value
;;
;; Takes four parameters:
;; $1: vector width of the target
;; $2: operation being performed (w.r.t. LLVM atomic intrinsic names)
;;     (add, sub...)
;; $3: return type of the LLVM atomic (e.g. i32)
;; $4: return type of the LLVM atomic type, in ispc naming paralance (e.g. int32)

define internal i32 @__get_first_active_lane()
{
  %nact  = call i32 @__ballot_nvptx(i1 true);
  %lane1 = call i32 @__count_leading_zeros_i32(i32 %nact)
  %lane  = sub i32 31, %lane1
  ret i32 %lane
}

define internal i32 @__atomic_add_uniform_int32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.add.u32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}
define internal i32 @__atomic_sub_uniform_int32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %nval = sub i32 0, %val;
  %old = tail call i32 @__atomic_add_uniform_int32_global_nvptx(i32* %ptr, i32 %nval);
  ret i32 %old;
}
define internal i32 @__atomic_and_uniform_int32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.and.b32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}
define internal i32 @__atomic_or_uniform_int32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.or.b32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}
define internal i32 @__atomic_xor_uniform_int32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.xor.b32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}
define internal i32 @__atomic_min_uniform_int32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.min.s32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}
define internal i32 @__atomic_max_uniform_int32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.max.s32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}
define internal i32 @__atomic_umin_uniform_uint32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.min.u32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}
define internal i32 @__atomic_umax_uniform_uint32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.max.u32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}


define internal i64 @__atomic_add_uniform_int64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.add.u64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}
define internal i64 @__atomic_sub_uniform_int64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %nval = sub i64 0, %val;
  %old = tail call i64 @__atomic_add_uniform_int64_global_nvptx(i64* %ptr, i64 %nval);
  ret i64 %old;
}
define internal i64 @__atomic_and_uniform_int64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.and.b64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}
define internal i64 @__atomic_or_uniform_int64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.or.b64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}
define internal i64 @__atomic_xor_uniform_int64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.xor.b64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}
define internal i64 @__atomic_min_uniform_int64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.min.s64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}
define internal i64 @__atomic_max_uniform_int64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.max.s64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}
define internal i64 @__atomic_umin_uniform_uint64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.min.u64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}
define internal i64 @__atomic_umax_uniform_uint64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.max.u64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}

define(`global_atomic',`
define <1 x $3> @__atomic_$2_$4_global($3* %ptr,  <1 x $3> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x i1> %maskv to i1
  %val  = bitcast <1 x $3> %valv  to $3
  br i1 %mask, label %exec, label %pass
exec:
  %old = call $3 @__atomic_$2_uniform_$4_global_nvptx($3 * %ptr, $3 %val);
  %oldv = bitcast $3 %old to <1 x $3>
  ret <1 x $3> %oldv
pass:
  ret <1 x $3> %valv
}
')
define(`global_atomic_uniform',`
define $3 @__atomic_$2_uniform_$4_global($3 * %ptr, $3 %val) nounwind alwaysinline
{
entry:
  %addr   = ptrtoint $3 * %ptr to i64
  %active = call i32 @__get_first_active_lane();
  %lane   = call i32 @__program_index();
  %c      = icmp eq i32 %lane, %active
  br i1 %c, label %p1, label %p2

p1:
  %t0 = call $3 @__atomic_$2_uniform_$4_global_nvptx($3 * %ptr, $3 %val);
  br label %p2;

p2: 
  %t1 = phi $3 [%t0, %p1], [zeroinitializer, %entry]
  %old = call $3 @__shfl_$3_nvptx($3 %t1, i32 %active)
  ret $3 %old;
}
')
define(`global_atomic_varying',`
define <1 x $3> @__atomic_$2_varying_$4_global(<1 x i64> %ptr, <1 x $3> %val, <1 x i1> %maskv) nounwind alwaysinline
{
entry:
  %addr  = bitcast <1 x i64> %ptr   to i64
  %c     = bitcast <1 x  i1> %maskv to  i1
  br i1 %c, label %p1, label %p2

p1:
  %sv = bitcast <1 x $3> %val to $3
  %sptr = inttoptr i64 %addr to $3*
  %t0 = call $3 @__atomic_$2_uniform_$4_global_nvptx($3 * %sptr, $3 %sv);
  %t0v = bitcast $3 %t0 to <1 x $3>
  ret < 1x $3> %t0v

p2: 
  ret <1 x $3> %val
}
')


global_atomic_uniform(1, add, i32, int32)
global_atomic_uniform(1, sub, i32, int32)
global_atomic_uniform(1, and, i32, int32)
global_atomic_uniform(1, or, i32, int32)
global_atomic_uniform(1, xor, i32, int32)
global_atomic_uniform(1, min, i32, int32)
global_atomic_uniform(1, max, i32, int32)
global_atomic_uniform(1, umin, i32, uint32)
global_atomic_uniform(1, umax, i32, uint32)

global_atomic_uniform(1, add, i64, int64)
global_atomic_uniform(1, sub, i64, int64)
global_atomic_uniform(1, and, i64, int64)
global_atomic_uniform(1, or, i64, int64)
global_atomic_uniform(1, xor, i64, int64)
global_atomic_uniform(1, min, i64, int64)
global_atomic_uniform(1, max, i64, int64)
global_atomic_uniform(1, umin, i64, uint64)
global_atomic_uniform(1, umax, i64, uint64)

global_atomic_varying(1, add, i32, int32)
global_atomic_varying(1, sub, i32, int32)
global_atomic_varying(1, and, i32, int32)
global_atomic_varying(1, or, i32, int32)
global_atomic_varying(1, xor, i32, int32)
global_atomic_varying(1, min, i32, int32)
global_atomic_varying(1, max, i32, int32)
global_atomic_varying(1, umin, i32, uint32)
global_atomic_varying(1, umax, i32, uint32)

global_atomic_varying(1, add, i64, int64)
global_atomic_varying(1, sub, i64, int64)
global_atomic_varying(1, and, i64, int64)
global_atomic_varying(1, or, i64, int64)
global_atomic_varying(1, xor, i64, int64)
global_atomic_varying(1, min, i64, int64)
global_atomic_varying(1, max, i64, int64)
global_atomic_varying(1, umin, i64, uint64)
global_atomic_varying(1, umax, i64, uint64)

;; Macro to declare the function that implements the swap atomic.  
;; Takes three parameters:
;; $1: vector width of the target
;; $2: llvm type of the vector elements (e.g. i32)
;; $3: ispc type of the elements (e.g. int32)

define internal i32 @__atomic_swap_uniform_int32_global_nvptx(i32* %ptr, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.exch.b32 $0, [$1], $2;", "=r,l,r"(i64 %addr, i32 %val);
  ret i32 %old;
}
define internal i64 @__atomic_swap_uniform_int64_global_nvptx(i64* %ptr, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.exch.b64 $0, [$1], $2;", "=l,l,l"(i64 %addr, i64 %val);
  ret i64 %old;
}
define internal float @__atomic_swap_uniform_float_global_nvptx(float* %ptr, float %val) nounwind alwaysinline
{
   %ptrI = bitcast float* %ptr to i32*
   %valI = bitcast float  %val to i32
   %retI = call i32 @__atomic_swap_uniform_int32_global_nvptx(i32* %ptrI, i32 %valI)
   %ret  = bitcast i32 %retI to float
   ret float %ret
}
define internal double @__atomic_swap_uniform_double_global_nvptx(double* %ptr, double %val) nounwind alwaysinline
{
   %ptrI = bitcast double* %ptr to i64*
   %valI = bitcast double  %val to i64
   %retI = call i64 @__atomic_swap_uniform_int64_global_nvptx(i64* %ptrI, i64 %valI)
   %ret  = bitcast i64 %retI to double
   ret double %ret
}
global_atomic_uniform(1, swap, i32, int32)
global_atomic_uniform(1, swap, i64, int64)
global_atomic_uniform(1, swap, float, float)
global_atomic_uniform(1, swap, double, double)
global_atomic_varying(1, swap, i32, int32)
global_atomic_varying(1, swap, i64, int64)
global_atomic_varying(1, swap, float, float)
global_atomic_varying(1, swap, double, double)


;; Similarly, macro to declare the function that implements the compare/exchange
;; atomic.  Takes three parameters:
;; $1: vector width of the target
;; $2: llvm type of the vector elements (e.g. i32)
;; $3: ispc type of the elements (e.g. int32)

define internal i32 @__atomic_compare_exchange_uniform_int32_global_nvptx(i32* %ptr, i32 %cmp, i32 %val) nounwind alwaysinline
{
  %addr = ptrtoint i32* %ptr to i64
  %old = tail call i32 asm sideeffect "atom.cas.b32 $0, [$1], $2, $3;", "=r,l,r,r"(i64 %addr, i32 %cmp, i32 %val);
  ret i32 %old;
}
define internal i64 @__atomic_compare_exchange_uniform_int64_global_nvptx(i64* %ptr, i64 %cmp, i64 %val) nounwind alwaysinline
{
  %addr = ptrtoint i64* %ptr to i64
  %old = tail call i64 asm sideeffect "atom.cas.b64 $0, [$1], $2, $3;", "=l,l,l,l"(i64 %addr, i64 %cmp, i64 %val);
  ret i64 %old;
}
define internal float @__atomic_compare_exchange_uniform_float_global_nvptx(float* %ptr, float %cmp, float %val) nounwind alwaysinline
{
   %ptrI = bitcast float* %ptr to i32*
   %cmpI = bitcast float  %cmp to i32
   %valI = bitcast float  %val to i32
   %retI = call i32 @__atomic_compare_exchange_uniform_int32_global_nvptx(i32* %ptrI, i32 %cmpI, i32 %valI)
   %ret  = bitcast i32 %retI to float
   ret float %ret
}
define internal double @__atomic_compare_exchange_uniform_double_global_nvptx(double* %ptr, double %cmp, double %val) nounwind alwaysinline
{
   %ptrI = bitcast double* %ptr to i64*
   %cmpI = bitcast double  %cmp to i64
   %valI = bitcast double  %val to i64
   %retI = call i64 @__atomic_compare_exchange_uniform_int64_global_nvptx(i64* %ptrI, i64 %cmpI, i64 %valI)
   %ret  = bitcast i64 %retI to double
   ret double %ret
}

;;;;;;;;;;;;
define(`global_atomic_cas',`
define <1 x $3> @__atomic_$2_$4_global($3* %ptr, <1 x $3> %cmpv, <1 x $3> %valv, <1 x i1> %maskv) nounwind alwaysinline
{
  %mask = bitcast <1 x i1> %maskv to i1
  %cmp  = bitcast <1 x $3> %cmpv  to $3
  %val  = bitcast <1 x $3> %valv  to $3
  br i1 %mask, label %exec, label %pass
exec:
  %old = call $3 @__atomic_$2_uniform_$4_global_nvptx($3 * %ptr, $3 %cmp, $3 %val);
  %oldv = bitcast $3 %old to <1 x $3>
  ret <1 x $3> %oldv
pass:
  ret <1 x $3> %valv
}
')
define(`global_atomic_cas_uniform',`
define $3 @__atomic_$2_uniform_$4_global($3 * %ptr, $3 %cmp, $3 %val) nounwind alwaysinline
{
entry:
  %addr   = ptrtoint $3 * %ptr to i64
  %active = call i32 @__get_first_active_lane();
  %lane   = call i32 @__program_index();
  %c      = icmp eq i32 %lane, %active
  br i1 %c, label %p1, label %p2

p1:
  %t0 = call $3 @__atomic_$2_uniform_$4_global_nvptx($3 * %ptr, $3 %cmp, $3 %val);
  br label %p2;

p2: 
  %t1 = phi $3 [%t0, %p1], [zeroinitializer, %entry]
  %old = call $3 @__shfl_$3_nvptx($3 %t1, i32 %active)
  ret $3 %old;
}
')
define(`global_atomic_cas_varying',`
define <1 x $3> @__atomic_$2_varying_$4_global(<1 x i64> %ptr, <1 x $3> %cmp, <1 x $3> %val, <1 x i1> %maskv) nounwind alwaysinline
{
entry:
  %addr  = bitcast <1 x i64> %ptr   to i64
  %c     = bitcast <1 x  i1> %maskv to  i1
  br i1 %c, label %p1, label %p2

p1:
  %sv = bitcast <1 x $3> %val to $3
  %sc = bitcast <1 x $3> %cmp to $3
  %sptr = inttoptr i64 %addr to $3*
  %t0 = call $3 @__atomic_$2_uniform_$4_global_nvptx($3 * %sptr, $3 %sc, $3 %sv);
  %t0v = bitcast $3 %t0 to <1 x $3>
  ret < 1x $3> %t0v

p2: 
  ret <1 x $3> %val
}
')

global_atomic_cas_uniform(1, compare_exchange, i32, int32)
global_atomic_cas_uniform(1, compare_exchange, i64, int64)
global_atomic_cas_uniform(1, compare_exchange, float, float)
global_atomic_cas_uniform(1, compare_exchange, double, double)
global_atomic_cas_varying(1, compare_exchange, i32, int32)
global_atomic_cas_varying(1, compare_exchange, i64, int64)
global_atomic_cas_varying(1, compare_exchange, float, float)
global_atomic_cas_varying(1, compare_exchange, double, double)
global_atomic_cas(1, compare_exchange, i32, int32)
global_atomic_cas(1, compare_exchange, i64, int64)
global_atomic_cas(1, compare_exchange, float, float)
global_atomic_cas(1, compare_exchange, double, double)




declare void @llvm.nvvm.membar.gl()
declare void @llvm.nvvm.membar.sys()
declare void @llvm.nvvm.membar.cta()

define void @__memory_barrier() nounwind readnone alwaysinline {
  ;; see http://llvm.org/bugs/show_bug.cgi?id=2829.  It seems like we
  ;; only get an MFENCE on x86 if "device" is true, but IMHO we should
  ;; in the case where the first 4 args are true but it is false.
  ;;  So we just always set that to true...
  call void @llvm.nvvm.membar.gl()
  ret void
}

saturation_arithmetic_novec();

;;;;;;;;;;;;;;;;;;;;
;; trigonometry


define(`transcendetals_decl',`
    declare float @__log_uniform_float(float) nounwind readnone
    declare <WIDTH x float> @__log_varying_float(<WIDTH x float>) nounwind readnone
    declare float @__exp_uniform_float(float) nounwind readnone
    declare <WIDTH x float> @__exp_varying_float(<WIDTH x float>) nounwind readnone
    declare float @__pow_uniform_float(float, float) nounwind readnone
    declare <WIDTH x float> @__pow_varying_float(<WIDTH x float>, <WIDTH x float>) nounwind readnone

    declare double @__log_uniform_double(double) nounwind readnone
    declare <WIDTH x double> @__log_varying_double(<WIDTH x double>) nounwind readnone
    declare double @__exp_uniform_double(double) nounwind readnone
    declare <WIDTH x double> @__exp_varying_double(<WIDTH x double>) nounwind readnone
    declare double @__pow_uniform_double(double, double) nounwind readnone
    declare <WIDTH x double> @__pow_varying_double(<WIDTH x double>, <WIDTH x double>) nounwind readnone
')

;; 1 - function call, e.g. __nv_fast_logf
;; 2 - data-type, float/double
;; 3 - local function name, e.g. __log, __exp, ..
define(`transcendentals1',`
declare $2 @$1($2)
define $2 @$3_uniform_$2($2) nounwind readnone alwaysinline
{
  %ret = call $2 @$1($2 %0)
  ret $2 %ret
}
define <1 x $2> @$3_varying_$2(<1 x $2>) nounwind readnone alwaysinline
{
  %v = bitcast <1 x $2> %0 to $2
  %r = call $2 @$3_uniform_$2($2 %v);
  %ret = bitcast $2 %r to <1 x $2>
  ret <1 x $2> %ret
}
')


define(`transcendentals2',`
declare $2 @$1($2, $2)
define $2 @$3_uniform_$2($2, $2) nounwind readnone alwaysinline
{
  %ret = call $2 @$1($2 %0, $2 %1)
  ret $2 %ret
}
define <1 x $2> @$3_varying_$2(<1 x $2>, <1x $2>) nounwind readnone alwaysinline
{
  %v0 = bitcast <1 x $2> %0 to $2
  %v1 = bitcast <1 x $2> %1 to $2
  %r = call $2 @$3_uniform_$2($2 %v0, $2 %v1);
  %ret = bitcast $2 %r to <1 x $2>
  ret <1 x $2> %ret
}
')
transcendentals1(__nv_fast_logf, float, __log)
transcendentals1(__nv_fast_expf, float, __exp)
transcendentals2(__nv_fast_powf, float, __pow)

transcendentals1(__nv_log, double, __log)
transcendentals1(__nv_exp, double, __exp)
transcendentals2(__nv_pow, double, __pow)


transcendentals1(__nv_fast_sinf, float, __sin)
transcendentals1(__nv_fast_cosf, float, __cos)
transcendentals1(__nv_fast_tanf, float, __tan)
transcendentals1(__nv_asinf,     float, __asin)
transcendentals1(__nv_acosf,     float, __acos)
transcendentals1(__nv_atanf,     float, __atan)
transcendentals2(__nv_atan2f,    float, __atan2)

transcendentals1(__nv_sin,   double, __sin)
transcendentals1(__nv_cos,   double, __cos)
transcendentals1(__nv_tan,   double, __tan)
transcendentals1(__nv_asin,  double, __asin)
transcendentals1(__nv_acos,  double, __acos)
transcendentals1(__nv_atan,  double, __atan)
transcendentals2(__nv_atan2, double, __atan2)

declare void @__sincos_uniform_float(float, float*, float*) nounwind readnone
declare void @__sincos_varying_float(<WIDTH x float>, <WIDTH x float>*, <WIDTH x float>*) nounwind readnone
declare void @__sincos_uniform_double(double, double*, double*) nounwind readnone
declare void @__sincos_varying_double(<WIDTH x double>, <WIDTH x double>*, <WIDTH x double>*) nounwind readnone

