; ModuleID = 'stencil_ispc_nvptx64.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64"

module asm ""
module asm ".extern .func  (.param .b32 func_retval0) cudaLaunchDevice"
module asm "("
module asm "  .param .b64 cudaLaunchDevice_param_0,"
module asm "  .param .b64 cudaLaunchDevice_param_1,"
module asm "  .param .align 4 .b8 cudaLaunchDevice_param_2[12],"
module asm "  .param .align 4 .b8 cudaLaunchDevice_param_3[12],"
module asm "  .param .b32 cudaLaunchDevice_param_4,"
module asm "  .param .b64 cudaLaunchDevice_param_5"
module asm ");"

@constDeltaForeach1 = private unnamed_addr constant [32 x i8] zeroinitializer
@constDeltaForeach4 = private unnamed_addr constant [32 x i8] c"\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F\10\11\12\13\14\15\16\17\18\19\1A\1B\1C\1D\1E\1F"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() nounwind readnone

declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() nounwind readnone

define i32 @__shfl_i32(i32, i32) {
  %shfl = tail call i32 asm sideeffect "shfl.idx.b32  $0, $1, $2, 0x1f;", "=r,r,r"(i32 %0, i32 %1)
  ret i32 %shfl
}

define float @__shfl_xor_float(float, i32) {
  %shfl = tail call float asm sideeffect "shfl.bfly.b32  $0, $1, $2, 0x1f;", "=f,f,r"(float %0, i32 %1)
  ret float %shfl
}

define i32 @__shfl_xor_i32(i32, i32) {
  %shfl = tail call i32 asm sideeffect "shfl.bfly.b32  $0, $1, $2, 0x1f;", "=r,r,r"(i32 %0, i32 %1)
  ret i32 %shfl
}

define float @__fminf(float, float) {
  %min = tail call float asm sideeffect "min.f32 $0, $1, $2;", "=f,f,f"(float %0, float %1)
  ret float %min
}

define float @__fmaxf(float, float) {
  %max = tail call float asm sideeffect "max.f32 $0, $1, $2;", "=f,f,f"(float %0, float %1)
  ret float %max
}

define i32 @__ballot(i1) {
  %conv = zext i1 %0 to i32
  %res = tail call i32 asm sideeffect "{ .reg .pred %p1; \0A         setp.ne.u32 %p1, $1, 0; \0A         vote.ballot.b32  $0, %p1; \0A      }", "=r,r"(i32 %conv)
  ret i32 %res
}

define i32 @__lanemask_lt() {
  %mask = tail call i32 asm sideeffect "mov.u32 $0, %lanemask_lt;", "=r"()
  ret i32 %mask
}

define i8* @ISPCAlloc(i8**, i64, i32) {
  ret i8* inttoptr (i64 1 to i8*)
}

declare i64 @cudaGetParameterBuffer(i64, i64)

define i8* @ISPCGetParamBuffer(i8**, i64 %align, i64 %size) {
entry:
  %tid.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %and = and i32 %tid.i, 31
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %ptri64tmp = tail call i64 @cudaGetParameterBuffer(i64 %align, i64 %size)
  %phitmp = inttoptr i64 %ptri64tmp to i8*
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %ptri64 = phi i8* [ %phitmp, %if.then ], [ null, %entry ]
  ret i8* %ptri64
}

define void @ISPCLaunch(i8**, i8* %func_ptr, i8* %func_args, i32 %ntx, i32 %nty, i32 %ntz) {
entry:
  %tid.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %and = and i32 %tid.i, 31
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %ntxm1 = add nsw i32 %ntx, -1
  %ntxm1d4 = ashr i32 %ntxm1, 2
  %nbx = add nsw i32 %ntxm1d4, 1
  %args_i64 = ptrtoint i8* %func_args to i64
  %func_i64 = ptrtoint i8* %func_ptr to i64
  %res_tmp = tail call i32 asm sideeffect "{\0A     .param .b64 param0;\0A     st.param.b64\09[param0+0], $1;\0A     .param .b64 param1;\0A     st.param.b64\09[param1+0], $2;\0A     .param .align 4 .b8 param2[12];\0A     st.param.b32\09[param2+0], $3; \0A     st.param.b32\09[param2+4], $4; \0A     st.param.b32\09[param2+8], $5; \0A     .param .align 4 .b8 param3[12];\0A     st.param.b32\09[param3+0], $6; \0A     st.param.b32\09[param3+4], $7; \0A     st.param.b32\09[param3+8], $8; \0A     .param .b32 param4;\0A     st.param.b32\09[param4+0], $9; \0A     .param .b64 param5;\0A     st.param.b64\09[param5+0], $10; \0A\0A     .param .b32 retval0;\0A     call.uni (retval0), \0A       cudaLaunchDevice,\0A       (\0A        param0, \0A        param1, \0A        param2, \0A        param3, \0A        param4, \0A        param5\0A       );\0A     ld.param.b32\09$0, [retval0+0];\0A  }\0A  ", "=r, l,l, r,r,r, r,r,r, r,l"(i64 %func_i64, i64 %args_i64, i32 %nbx, i32 %nty, i32 %ntz, i32 128, i32 1, i32 1, i32 0, i64 0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare i32 @cudaDeviceSynchronize()

define void @ISPCSync(i8*) {
  %2 = tail call i32 @cudaDeviceSynchronize()
  ret void
}

define i64 @__warpBinExclusiveScan(i1 %p) {
entry:
  %conv.i = zext i1 %p to i32
  %res.i = tail call i32 asm sideeffect "{ .reg .pred %p1; \0A         setp.ne.u32 %p1, $1, 0; \0A         vote.ballot.b32  $0, %p1; \0A      }", "=r,r"(i32 %conv.i)
  %res.i1 = tail call i32 asm sideeffect "popc.b32 $0, $1;", "=r,r"(i32 %res.i)
  %mask.i = tail call i32 asm sideeffect "mov.u32 $0, %lanemask_lt;", "=r"()
  %and = and i32 %mask.i, %res.i
  %res.i2 = tail call i32 asm sideeffect "popc.b32 $0, $1;", "=r,r"(i32 %and)
  %retval.sroa.1.4.insert.ext.i = zext i32 %res.i2 to i64
  %retval.sroa.1.4.insert.shift.i = shl nuw i64 %retval.sroa.1.4.insert.ext.i, 32
  %retval.sroa.0.0.insert.ext.i = zext i32 %res.i1 to i64
  %retval.sroa.0.0.insert.insert.i = or i64 %retval.sroa.1.4.insert.shift.i, %retval.sroa.0.0.insert.ext.i
  ret i64 %retval.sroa.0.0.insert.insert.i
}

define internal void @stencil_step_task___UM_uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_(i32 %x0, i32 %x1, i32 %y0, i32 %y1, i32 %z0, i32 %z1, i32 %Nx, i32 %Ny, i32 %Nz, double* %coef, double* %vsq, double* %Ain, double* %Aout) {
allocas:
  %bid.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul_calltmp_.i = shl i32 %bid.i.i, 2
  %tid.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bitop.i = ashr i32 %tid.i.i, 5
  %add_mul_calltmp__bitop.i = add i32 %bitop.i, %mul_calltmp_.i
  %nb.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %mul_calltmp_.i57 = shl i32 %nb.i.i, 2
  %greaterequal_calltmp_calltmp18 = icmp sge i32 %add_mul_calltmp__bitop.i, %mul_calltmp_.i57
  %bid.i.i58 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %nb.i.i59 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %greaterequal_calltmp21_calltmp24 = icmp sge i32 %bid.i.i58, %nb.i.i59
  %logical_or = or i1 %greaterequal_calltmp_calltmp18, %greaterequal_calltmp21_calltmp24
  %bid.i.i60 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %nb.i.i61 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %greaterequal_calltmp27_calltmp30 = icmp sge i32 %bid.i.i60, %nb.i.i61
  %logical_or31 = or i1 %logical_or, %greaterequal_calltmp27_calltmp30
  br i1 %logical_or31, label %if_then, label %if_exit

if_then:                                          ; preds = %foreach_reset19.i, %if_exit, %allocas
  ret void

if_exit:                                          ; preds = %allocas
  %bid.i.i62 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul_calltmp_.i63 = shl i32 %bid.i.i62, 7
  %tid.i.i64 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bitop.i657375 = add i32 %tid.i.i64, %mul_calltmp_.i63
  %mul_calltmp35_ = and i32 %bitop.i657375, -32
  %add_x0_load_mul_calltmp35_ = add i32 %mul_calltmp35_, %x0
  %add_xfirst_load_ = add i32 %add_x0_load_mul_calltmp35_, 32
  %c.i.i = icmp sgt i32 %add_xfirst_load_, %x1
  %r.i.i = select i1 %c.i.i, i32 %x1, i32 %add_xfirst_load_
  %bid.i.i67 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %mul_calltmp41_ = shl i32 %bid.i.i67, 3
  %add_y0_load_mul_calltmp41_ = add i32 %mul_calltmp41_, %y0
  %add_yfirst_load_ = add i32 %add_y0_load_mul_calltmp41_, 8
  %bid.i.i70 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %mul_calltmp47_ = shl i32 %bid.i.i70, 3
  %add_z0_load_mul_calltmp47_ = add i32 %mul_calltmp47_, %z0
  %add_zfirst_load_ = add i32 %add_z0_load_mul_calltmp47_, 8
  %c.i.i71 = icmp sgt i32 %add_zfirst_load_, %z1
  %r.i.i72 = select i1 %c.i.i71, i32 %z1, i32 %add_zfirst_load_
  %mul_Nx_load_Ny_load.i = mul i32 %Ny, %Nx
  %nitems29.i = sub i32 %r.i.i, %add_x0_load_mul_calltmp35_
  %nextras30.i = srem i32 %nitems29.i, 32
  %aligned_end31.i = sub i32 %r.i.i, %nextras30.i
  %tid.i4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %__laneidx.i = and i32 %tid.i4.i, 31
  %0 = zext i32 %__laneidx.i to i64
  %arrayidx.i = getelementptr [32 x i8]* @constDeltaForeach1, i64 0, i64 %0
  %cmp38.i396 = icmp slt i32 %add_z0_load_mul_calltmp47_, %r.i.i72
  br i1 %cmp38.i396, label %foreach_test21.i.preheader.lr.ph, label %if_then

foreach_test21.i.preheader.lr.ph:                 ; preds = %if_exit
  %c.i.i68 = icmp sgt i32 %add_yfirst_load_, %y1
  %r.i.i69 = select i1 %c.i.i68, i32 %y1, i32 %add_yfirst_load_
  %1 = load i8* %arrayidx.i, align 1
  %_zext.i394 = zext i8 %1 to i32
  %2 = insertelement <1 x i32> undef, i32 %_zext.i394, i32 0
  %smear_counter_init.i393 = insertelement <1 x i32> undef, i32 %add_z0_load_mul_calltmp47_, i32 0
  %iter_val.i395 = add <1 x i32> %smear_counter_init.i393, %2
  %smear_counter_init44.i387 = insertelement <1 x i32> undef, i32 %add_y0_load_mul_calltmp41_, i32 0
  %cmp54.i390 = icmp slt i32 %add_y0_load_mul_calltmp41_, %r.i.i69
  %before_aligned_end73.i385 = icmp slt i32 %add_x0_load_mul_calltmp35_, %aligned_end31.i
  %smear_end_init289.i = insertelement <1 x i32> undef, i32 %r.i.i, i32 0
  %Nxy_load298_broadcast_init.i = insertelement <1 x i32> undef, i32 %mul_Nx_load_Ny_load.i, i32 0
  %Nx_load300_broadcast_init.i = insertelement <1 x i32> undef, i32 %Nx, i32 0
  %Ain_load309_ptr2int.i = ptrtoint double* %Ain to i64
  %coef_load314_offset.i = getelementptr double* %coef, i64 1
  %coef_load365_offset.i = getelementptr double* %coef, i64 2
  %mul__Nx_load385.i = shl i32 %Nx, 1
  %mul__Nx_load393.i = mul i32 %Nx, -2
  %mul__Nxy_load402.i = shl i32 %mul_Nx_load_Ny_load.i, 1
  %mul__Nxy_load410.i = mul i32 %mul_Nx_load_Ny_load.i, -2
  %coef_load416_offset.i = getelementptr double* %coef, i64 3
  %mul__Nx_load436.i = mul i32 %Nx, 3
  %mul__Nx_load444.i = mul i32 %Nx, -3
  %mul__Nxy_load453.i = mul i32 %mul_Nx_load_Ny_load.i, 3
  %mul__Nxy_load461.i = mul i32 %mul_Nx_load_Ny_load.i, -3
  %Aout_load470_ptr2int.i = ptrtoint double* %Aout to i64
  %vsq_load488_ptr2int.i = ptrtoint double* %vsq to i64
  %3 = sub i32 -9, %y0
  %4 = shl i32 %bid.i.i67, 3
  %5 = sub i32 %3, %4
  %6 = xor i32 %y1, -1
  %7 = icmp sgt i32 %5, %6
  %smax = select i1 %7, i32 %5, i32 %6
  %8 = xor i32 %smax, -1
  %9 = sub i32 -9, %z0
  %10 = shl i32 %bid.i.i70, 3
  %11 = sub i32 %9, %10
  %12 = xor i32 %z1, -1
  %13 = icmp sgt i32 %11, %12
  %smax399 = select i1 %13, i32 %11, i32 %12
  %14 = xor i32 %smax399, -1
  br label %foreach_test21.i.preheader

foreach_full_body.i:                              ; preds = %outer_not_in_extras.i.preheader, %foreach_full_body.i
  %counter32.4.i386 = phi i32 [ %new_counter279.i, %foreach_full_body.i ], [ %add_x0_load_mul_calltmp35_, %outer_not_in_extras.i.preheader ]
  %tid.i.i56 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %__laneidx80.i = and i32 %tid.i.i56, 31
  %15 = zext i32 %__laneidx80.i to i64
  %arrayidx81.i = getelementptr [32 x i8]* @constDeltaForeach4, i64 0, i64 %15
  %16 = load i8* %arrayidx81.i, align 1
  %_zext82.i = zext i8 %16 to i32
  %coef_load_offset_load.i = load double* %coef, align 8
  %.lhs362.lhs.lhs = extractelement <1 x i32> %mul_z_load297_Nxy_load298_broadcast.i, i32 0
  %.lhs362.lhs.rhs.lhs = extractelement <1 x i32> %iter_val50.i392, i32 0
  %.lhs362.lhs.rhs = mul i32 %.lhs362.lhs.rhs.lhs, %Nx
  %.lhs362.lhs = add i32 %.lhs362.lhs.lhs, %.lhs362.lhs.rhs
  %.lhs362.rhs = add i32 %counter32.4.i386, %_zext82.i
  %.lhs362 = add i32 %.lhs362.lhs, %.lhs362.rhs
  %17 = shl i32 %.lhs362, 3
  %iptr__id.i.rhs = sext i32 %17 to i64
  %iptr__id.i = add i64 %iptr__id.i.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i = inttoptr i64 %iptr__id.i to double*
  %val__id.i = load double* %ptr__id.i, align 8
  %coef_load94_offset_load.i = load double* %coef_load314_offset.i, align 8
  %18 = add i32 %17, 8
  %iptr__id.i335.rhs = sext i32 %18 to i64
  %iptr__id.i335 = add i64 %iptr__id.i335.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i336 = inttoptr i64 %iptr__id.i335 to double*
  %val__id.i337 = load double* %ptr__id.i336, align 8
  %19 = add i32 %17, -8
  %iptr__id.i330.rhs = sext i32 %19 to i64
  %iptr__id.i330 = add i64 %iptr__id.i330.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i331 = inttoptr i64 %iptr__id.i330 to double*
  %val__id.i332 = load double* %ptr__id.i331, align 8
  %.lhs365 = add i32 %.lhs362, %Nx
  %20 = shl i32 %.lhs365, 3
  %iptr__id.i325.rhs = sext i32 %20 to i64
  %iptr__id.i325 = add i64 %iptr__id.i325.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i326 = inttoptr i64 %iptr__id.i325 to double*
  %val__id.i327 = load double* %ptr__id.i326, align 8
  %.lhs366 = sub i32 %.lhs362, %Nx
  %21 = shl i32 %.lhs366, 3
  %iptr__id.i320.rhs = sext i32 %21 to i64
  %iptr__id.i320 = add i64 %iptr__id.i320.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i321 = inttoptr i64 %iptr__id.i320 to double*
  %val__id.i322 = load double* %ptr__id.i321, align 8
  %.lhs367 = add i32 %.lhs362, %mul_Nx_load_Ny_load.i
  %22 = shl i32 %.lhs367, 3
  %iptr__id.i315.rhs = sext i32 %22 to i64
  %iptr__id.i315 = add i64 %iptr__id.i315.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i316 = inttoptr i64 %iptr__id.i315 to double*
  %val__id.i317 = load double* %ptr__id.i316, align 8
  %.lhs368 = sub i32 %.lhs362, %mul_Nx_load_Ny_load.i
  %23 = shl i32 %.lhs368, 3
  %iptr__id.i310.rhs = sext i32 %23 to i64
  %iptr__id.i310 = add i64 %iptr__id.i310.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i311 = inttoptr i64 %iptr__id.i310 to double*
  %val__id.i312 = load double* %ptr__id.i311, align 8
  %coef_load145_offset_load.i = load double* %coef_load365_offset.i, align 8
  %24 = add i32 %17, 16
  %iptr__id.i305.rhs = sext i32 %24 to i64
  %iptr__id.i305 = add i64 %iptr__id.i305.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i306 = inttoptr i64 %iptr__id.i305 to double*
  %val__id.i307 = load double* %ptr__id.i306, align 8
  %25 = add i32 %17, -16
  %iptr__id.i300.rhs = sext i32 %25 to i64
  %iptr__id.i300 = add i64 %iptr__id.i300.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i301 = inttoptr i64 %iptr__id.i300 to double*
  %val__id.i302 = load double* %ptr__id.i301, align 8
  %.lhs371 = add i32 %.lhs362, %mul__Nx_load385.i
  %26 = shl i32 %.lhs371, 3
  %iptr__id.i295.rhs = sext i32 %26 to i64
  %iptr__id.i295 = add i64 %iptr__id.i295.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i296 = inttoptr i64 %iptr__id.i295 to double*
  %val__id.i297 = load double* %ptr__id.i296, align 8
  %.lhs372 = add i32 %.lhs362, %mul__Nx_load393.i
  %27 = shl i32 %.lhs372, 3
  %iptr__id.i290.rhs = sext i32 %27 to i64
  %iptr__id.i290 = add i64 %iptr__id.i290.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i291 = inttoptr i64 %iptr__id.i290 to double*
  %val__id.i292 = load double* %ptr__id.i291, align 8
  %.lhs373 = add i32 %.lhs362, %mul__Nxy_load402.i
  %28 = shl i32 %.lhs373, 3
  %iptr__id.i285.rhs = sext i32 %28 to i64
  %iptr__id.i285 = add i64 %iptr__id.i285.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i286 = inttoptr i64 %iptr__id.i285 to double*
  %val__id.i287 = load double* %ptr__id.i286, align 8
  %.lhs374 = add i32 %.lhs362, %mul__Nxy_load410.i
  %29 = shl i32 %.lhs374, 3
  %iptr__id.i280.rhs = sext i32 %29 to i64
  %iptr__id.i280 = add i64 %iptr__id.i280.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i281 = inttoptr i64 %iptr__id.i280 to double*
  %val__id.i282 = load double* %ptr__id.i281, align 8
  %coef_load196_offset_load.i = load double* %coef_load416_offset.i, align 8
  %30 = add i32 %17, 24
  %iptr__id.i275.rhs = sext i32 %30 to i64
  %iptr__id.i275 = add i64 %iptr__id.i275.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i276 = inttoptr i64 %iptr__id.i275 to double*
  %val__id.i277 = load double* %ptr__id.i276, align 8
  %31 = add i32 %17, -24
  %iptr__id.i270.rhs = sext i32 %31 to i64
  %iptr__id.i270 = add i64 %iptr__id.i270.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i271 = inttoptr i64 %iptr__id.i270 to double*
  %val__id.i272 = load double* %ptr__id.i271, align 8
  %.lhs377 = add i32 %.lhs362, %mul__Nx_load436.i
  %32 = shl i32 %.lhs377, 3
  %iptr__id.i265.rhs = sext i32 %32 to i64
  %iptr__id.i265 = add i64 %iptr__id.i265.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i266 = inttoptr i64 %iptr__id.i265 to double*
  %val__id.i267 = load double* %ptr__id.i266, align 8
  %.lhs378 = add i32 %.lhs362, %mul__Nx_load444.i
  %33 = shl i32 %.lhs378, 3
  %iptr__id.i260.rhs = sext i32 %33 to i64
  %iptr__id.i260 = add i64 %iptr__id.i260.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i261 = inttoptr i64 %iptr__id.i260 to double*
  %val__id.i262 = load double* %ptr__id.i261, align 8
  %.lhs379 = add i32 %.lhs362, %mul__Nxy_load453.i
  %34 = shl i32 %.lhs379, 3
  %iptr__id.i255.rhs = sext i32 %34 to i64
  %iptr__id.i255 = add i64 %iptr__id.i255.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i256 = inttoptr i64 %iptr__id.i255 to double*
  %val__id.i257 = load double* %ptr__id.i256, align 8
  %.lhs380 = add i32 %.lhs362, %mul__Nxy_load461.i
  %35 = shl i32 %.lhs380, 3
  %iptr__id.i250.rhs = sext i32 %35 to i64
  %iptr__id.i250 = add i64 %iptr__id.i250.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i251 = inttoptr i64 %iptr__id.i250 to double*
  %val__id.i252 = load double* %ptr__id.i251, align 8
  %val__id.i247 = load double* %ptr__id.i, align 8
  %iptr__id.i240 = add i64 %iptr__id.i.rhs, %Aout_load470_ptr2int.i
  %ptr__id.i241 = inttoptr i64 %iptr__id.i240 to double*
  %val__id.i242 = load double* %ptr__id.i241, align 8
  %iptr__id.i235 = add i64 %iptr__id.i.rhs, %vsq_load488_ptr2int.i
  %ptr__id.i236 = inttoptr i64 %iptr__id.i235 to double*
  %val__id.i237 = load double* %ptr__id.i236, align 8
  %val__id.i233.lhs.lhs = fmul double %val__id.i247, 2.000000e+00
  %val__id.i233.lhs = fsub double %val__id.i233.lhs.lhs, %val__id.i242
  %val__id.i233.rhs.rhs.lhs.lhs.lhs = fmul double %coef_load_offset_load.i, %val__id.i
  %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs.lhs.lhs = fadd double %val__id.i337, %val__id.i332
  %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs.lhs = fadd double %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs.lhs.lhs, %val__id.i327
  %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs = fadd double %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs.lhs, %val__id.i322
  %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs.lhs = fadd double %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs, %val__id.i317
  %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs = fadd double %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs.lhs, %val__id.i312
  %val__id.i233.rhs.rhs.lhs.lhs.rhs = fmul double %coef_load94_offset_load.i, %val__id.i233.rhs.rhs.lhs.lhs.rhs.rhs
  %val__id.i233.rhs.rhs.lhs.lhs = fadd double %val__id.i233.rhs.rhs.lhs.lhs.lhs, %val__id.i233.rhs.rhs.lhs.lhs.rhs
  %val__id.i233.rhs.rhs.lhs.rhs.rhs.lhs.lhs.lhs.lhs = fadd double %val__id.i307, %val__id.i302
  %val__id.i233.rhs.rhs.lhs.rhs.rhs.lhs.lhs.lhs = fadd double %val__id.i233.rhs.rhs.lhs.rhs.rhs.lhs.lhs.lhs.lhs, %val__id.i297
  %val__id.i233.rhs.rhs.lhs.rhs.rhs.lhs.lhs = fadd double %val__id.i233.rhs.rhs.lhs.rhs.rhs.lhs.lhs.lhs, %val__id.i292
  %val__id.i233.rhs.rhs.lhs.rhs.rhs.lhs = fadd double %val__id.i233.rhs.rhs.lhs.rhs.rhs.lhs.lhs, %val__id.i287
  %val__id.i233.rhs.rhs.lhs.rhs.rhs = fadd double %val__id.i233.rhs.rhs.lhs.rhs.rhs.lhs, %val__id.i282
  %val__id.i233.rhs.rhs.lhs.rhs = fmul double %coef_load145_offset_load.i, %val__id.i233.rhs.rhs.lhs.rhs.rhs
  %val__id.i233.rhs.rhs.lhs = fadd double %val__id.i233.rhs.rhs.lhs.lhs, %val__id.i233.rhs.rhs.lhs.rhs
  %val__id.i233.rhs.rhs.rhs.rhs.lhs.lhs.lhs.lhs = fadd double %val__id.i277, %val__id.i272
  %val__id.i233.rhs.rhs.rhs.rhs.lhs.lhs.lhs = fadd double %val__id.i233.rhs.rhs.rhs.rhs.lhs.lhs.lhs.lhs, %val__id.i267
  %val__id.i233.rhs.rhs.rhs.rhs.lhs.lhs = fadd double %val__id.i233.rhs.rhs.rhs.rhs.lhs.lhs.lhs, %val__id.i262
  %val__id.i233.rhs.rhs.rhs.rhs.lhs = fadd double %val__id.i233.rhs.rhs.rhs.rhs.lhs.lhs, %val__id.i257
  %val__id.i233.rhs.rhs.rhs.rhs = fadd double %val__id.i233.rhs.rhs.rhs.rhs.lhs, %val__id.i252
  %val__id.i233.rhs.rhs.rhs = fmul double %coef_load196_offset_load.i, %val__id.i233.rhs.rhs.rhs.rhs
  %val__id.i233.rhs.rhs = fadd double %val__id.i233.rhs.rhs.lhs, %val__id.i233.rhs.rhs.rhs
  %val__id.i233.rhs = fmul double %val__id.i237, %val__id.i233.rhs.rhs
  %val__id.i233 = fadd double %val__id.i233.lhs, %val__id.i233.rhs
  store double %val__id.i233, double* %ptr__id.i241, align 8
  %new_counter279.i = add i32 %counter32.4.i386, 32
  %before_aligned_end73.i = icmp slt i32 %new_counter279.i, %aligned_end31.i
  br i1 %before_aligned_end73.i, label %foreach_full_body.i, label %partial_inner_all_outer.i

foreach_test21.i.preheader:                       ; preds = %foreach_reset19.i, %foreach_test21.i.preheader.lr.ph
  %iter_val.i398 = phi <1 x i32> [ %iter_val.i395, %foreach_test21.i.preheader.lr.ph ], [ %iter_val.i, %foreach_reset19.i ]
  %counter.0.i397 = phi i32 [ %add_z0_load_mul_calltmp47_, %foreach_test21.i.preheader.lr.ph ], [ %new_counter.i, %foreach_reset19.i ]
  %tid.i3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %__laneidx47.i = and i32 %tid.i3.i, 31
  %36 = zext i32 %__laneidx47.i to i64
  %arrayidx48.i = getelementptr [32 x i8]* @constDeltaForeach1, i64 0, i64 %36
  br i1 %cmp54.i390, label %outer_not_in_extras.i.preheader.lr.ph, label %foreach_reset19.i

outer_not_in_extras.i.preheader.lr.ph:            ; preds = %foreach_test21.i.preheader
  %37 = load i8* %arrayidx48.i, align 1
  %_zext49.i388 = zext i8 %37 to i32
  %38 = insertelement <1 x i32> undef, i32 %_zext49.i388, i32 0
  %iter_val50.i389 = add <1 x i32> %smear_counter_init44.i387, %38
  %mul_z_load297_Nxy_load298_broadcast.i = mul <1 x i32> %iter_val.i398, %Nxy_load298_broadcast_init.i
  br label %outer_not_in_extras.i.preheader

foreach_reset19.i:                                ; preds = %foreach_reset27.i, %foreach_test21.i.preheader
  %new_counter.i = add i32 %counter.0.i397, 1
  %smear_counter_init.i = insertelement <1 x i32> undef, i32 %new_counter.i, i32 0
  %39 = load i8* %arrayidx.i, align 1
  %_zext.i = zext i8 %39 to i32
  %40 = insertelement <1 x i32> undef, i32 %_zext.i, i32 0
  %iter_val.i = add <1 x i32> %smear_counter_init.i, %40
  %exitcond400 = icmp eq i32 %new_counter.i, %14
  br i1 %exitcond400, label %if_then, label %foreach_test21.i.preheader

outer_not_in_extras.i.preheader:                  ; preds = %foreach_reset27.i, %outer_not_in_extras.i.preheader.lr.ph
  %iter_val50.i392 = phi <1 x i32> [ %iter_val50.i389, %outer_not_in_extras.i.preheader.lr.ph ], [ %iter_val50.i, %foreach_reset27.i ]
  %counter25.1.i391 = phi i32 [ %add_y0_load_mul_calltmp41_, %outer_not_in_extras.i.preheader.lr.ph ], [ %new_counter35.i, %foreach_reset27.i ]
  br i1 %before_aligned_end73.i385, label %foreach_full_body.i, label %partial_inner_all_outer.i

foreach_reset27.i:                                ; preds = %pl_dolane.i, %partial_inner_only.i, %partial_inner_all_outer.i
  %new_counter35.i = add i32 %counter25.1.i391, 1
  %smear_counter_init44.i = insertelement <1 x i32> undef, i32 %new_counter35.i, i32 0
  %41 = load i8* %arrayidx48.i, align 1
  %_zext49.i = zext i8 %41 to i32
  %42 = insertelement <1 x i32> undef, i32 %_zext49.i, i32 0
  %iter_val50.i = add <1 x i32> %smear_counter_init44.i, %42
  %exitcond = icmp eq i32 %new_counter35.i, %8
  br i1 %exitcond, label %foreach_reset19.i, label %outer_not_in_extras.i.preheader

partial_inner_all_outer.i:                        ; preds = %outer_not_in_extras.i.preheader, %foreach_full_body.i
  %counter32.4.i.lcssa = phi i32 [ %add_x0_load_mul_calltmp35_, %outer_not_in_extras.i.preheader ], [ %new_counter279.i, %foreach_full_body.i ]
  %before_full_end.i = icmp slt i32 %counter32.4.i.lcssa, %r.i.i
  br i1 %before_full_end.i, label %partial_inner_only.i, label %foreach_reset27.i

partial_inner_only.i:                             ; preds = %partial_inner_all_outer.i
  %smear_counter_init282.i = insertelement <1 x i32> undef, i32 %counter32.4.i.lcssa, i32 0
  %tid.i2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %__laneidx285.i = and i32 %tid.i2.i, 31
  %43 = zext i32 %__laneidx285.i to i64
  %arrayidx286.i = getelementptr [32 x i8]* @constDeltaForeach4, i64 0, i64 %43
  %44 = load i8* %arrayidx286.i, align 1
  %_zext287.i = zext i8 %44 to i32
  %45 = insertelement <1 x i32> undef, i32 %_zext287.i, i32 0
  %iter_val288.i = add <1 x i32> %smear_counter_init282.i, %45
  %cmp291.i = icmp slt <1 x i32> %iter_val288.i, %smear_end_init289.i
  %mul_y_load299_Nx_load300_broadcast.i = mul <1 x i32> %iter_val50.i392, %Nx_load300_broadcast_init.i
  %add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast.i = add <1 x i32> %mul_z_load297_Nxy_load298_broadcast.i, %mul_y_load299_Nx_load300_broadcast.i
  %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i = add <1 x i32> %add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast.i, %iter_val288.i
  %v.i.i224 = extractelement <1 x i1> %cmp291.i, i32 0
  br i1 %v.i.i224, label %pl_dolane.i, label %foreach_reset27.i

pl_dolane.i:                                      ; preds = %partial_inner_only.i
  %coef_load303_offset_load.i = load double* %coef, align 8
  %.lhs361 = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %46 = shl i32 %.lhs361, 3
  %iptr__id.i225.rhs = sext i32 %46 to i64
  %iptr__id.i225 = add i64 %iptr__id.i225.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i226 = inttoptr i64 %iptr__id.i225 to double*
  %val__id.i227 = load double* %ptr__id.i226, align 8
  %coef_load314_offset_load.i401 = load double* %coef_load314_offset.i, align 8
  %.lhs360.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs360 = shl i32 %.lhs360.lhs, 3
  %47 = add i32 %.lhs360, 8
  %iptr__id.i218.rhs = sext i32 %47 to i64
  %iptr__id.i218 = add i64 %iptr__id.i218.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i219 = inttoptr i64 %iptr__id.i218 to double*
  %val__id.i220 = load double* %ptr__id.i219, align 8
  %.lhs359.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs359 = shl i32 %.lhs359.lhs, 3
  %48 = add i32 %.lhs359, -8
  %iptr__id.i211.rhs = sext i32 %48 to i64
  %iptr__id.i211 = add i64 %iptr__id.i211.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i212 = inttoptr i64 %iptr__id.i211 to double*
  %val__id.i213 = load double* %ptr__id.i212, align 8
  %.lhs358.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs358 = add i32 %.lhs358.lhs, %Nx
  %49 = shl i32 %.lhs358, 3
  %iptr__id.i204.rhs = sext i32 %49 to i64
  %iptr__id.i204 = add i64 %iptr__id.i204.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i205 = inttoptr i64 %iptr__id.i204 to double*
  %val__id.i206 = load double* %ptr__id.i205, align 8
  %.lhs357.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs357 = sub i32 %.lhs357.lhs, %Nx
  %50 = shl i32 %.lhs357, 3
  %iptr__id.i197.rhs = sext i32 %50 to i64
  %iptr__id.i197 = add i64 %iptr__id.i197.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i198 = inttoptr i64 %iptr__id.i197 to double*
  %val__id.i199 = load double* %ptr__id.i198, align 8
  %.lhs356.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs356 = add i32 %.lhs356.lhs, %mul_Nx_load_Ny_load.i
  %51 = shl i32 %.lhs356, 3
  %iptr__id.i190.rhs = sext i32 %51 to i64
  %iptr__id.i190 = add i64 %iptr__id.i190.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i191 = inttoptr i64 %iptr__id.i190 to double*
  %val__id.i192 = load double* %ptr__id.i191, align 8
  %.lhs355.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs355 = sub i32 %.lhs355.lhs, %mul_Nx_load_Ny_load.i
  %52 = shl i32 %.lhs355, 3
  %iptr__id.i183.rhs = sext i32 %52 to i64
  %iptr__id.i183 = add i64 %iptr__id.i183.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i184 = inttoptr i64 %iptr__id.i183 to double*
  %val__id.i185 = load double* %ptr__id.i184, align 8
  %coef_load365_offset_load.i457 = load double* %coef_load365_offset.i, align 8
  %.lhs354.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs354 = shl i32 %.lhs354.lhs, 3
  %53 = add i32 %.lhs354, 16
  %iptr__id.i176.rhs = sext i32 %53 to i64
  %iptr__id.i176 = add i64 %iptr__id.i176.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i177 = inttoptr i64 %iptr__id.i176 to double*
  %val__id.i178 = load double* %ptr__id.i177, align 8
  %.lhs353.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs353 = shl i32 %.lhs353.lhs, 3
  %54 = add i32 %.lhs353, -16
  %iptr__id.i169.rhs = sext i32 %54 to i64
  %iptr__id.i169 = add i64 %iptr__id.i169.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i170 = inttoptr i64 %iptr__id.i169 to double*
  %val__id.i171 = load double* %ptr__id.i170, align 8
  %.lhs352.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs352 = add i32 %.lhs352.lhs, %mul__Nx_load385.i
  %55 = shl i32 %.lhs352, 3
  %iptr__id.i162.rhs = sext i32 %55 to i64
  %iptr__id.i162 = add i64 %iptr__id.i162.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i163 = inttoptr i64 %iptr__id.i162 to double*
  %val__id.i164 = load double* %ptr__id.i163, align 8
  %.lhs351.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs351 = add i32 %.lhs351.lhs, %mul__Nx_load393.i
  %56 = shl i32 %.lhs351, 3
  %iptr__id.i155.rhs = sext i32 %56 to i64
  %iptr__id.i155 = add i64 %iptr__id.i155.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i156 = inttoptr i64 %iptr__id.i155 to double*
  %val__id.i157 = load double* %ptr__id.i156, align 8
  %.lhs350.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs350 = add i32 %.lhs350.lhs, %mul__Nxy_load402.i
  %57 = shl i32 %.lhs350, 3
  %iptr__id.i148.rhs = sext i32 %57 to i64
  %iptr__id.i148 = add i64 %iptr__id.i148.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i149 = inttoptr i64 %iptr__id.i148 to double*
  %val__id.i150 = load double* %ptr__id.i149, align 8
  %.lhs349.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs349 = add i32 %.lhs349.lhs, %mul__Nxy_load410.i
  %58 = shl i32 %.lhs349, 3
  %iptr__id.i141.rhs = sext i32 %58 to i64
  %iptr__id.i141 = add i64 %iptr__id.i141.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i142 = inttoptr i64 %iptr__id.i141 to double*
  %val__id.i143 = load double* %ptr__id.i142, align 8
  %coef_load416_offset_load.i544 = load double* %coef_load416_offset.i, align 8
  %.lhs348.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs348 = shl i32 %.lhs348.lhs, 3
  %59 = add i32 %.lhs348, 24
  %iptr__id.i134.rhs = sext i32 %59 to i64
  %iptr__id.i134 = add i64 %iptr__id.i134.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i135 = inttoptr i64 %iptr__id.i134 to double*
  %val__id.i136 = load double* %ptr__id.i135, align 8
  %.lhs347.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs347 = shl i32 %.lhs347.lhs, 3
  %60 = add i32 %.lhs347, -24
  %iptr__id.i127.rhs = sext i32 %60 to i64
  %iptr__id.i127 = add i64 %iptr__id.i127.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i128 = inttoptr i64 %iptr__id.i127 to double*
  %val__id.i129 = load double* %ptr__id.i128, align 8
  %.lhs346.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs346 = add i32 %.lhs346.lhs, %mul__Nx_load436.i
  %61 = shl i32 %.lhs346, 3
  %iptr__id.i120.rhs = sext i32 %61 to i64
  %iptr__id.i120 = add i64 %iptr__id.i120.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i121 = inttoptr i64 %iptr__id.i120 to double*
  %val__id.i122 = load double* %ptr__id.i121, align 8
  %.lhs345.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs345 = add i32 %.lhs345.lhs, %mul__Nx_load444.i
  %62 = shl i32 %.lhs345, 3
  %iptr__id.i113.rhs = sext i32 %62 to i64
  %iptr__id.i113 = add i64 %iptr__id.i113.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i114 = inttoptr i64 %iptr__id.i113 to double*
  %val__id.i115 = load double* %ptr__id.i114, align 8
  %.lhs344.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs344 = add i32 %.lhs344.lhs, %mul__Nxy_load453.i
  %63 = shl i32 %.lhs344, 3
  %iptr__id.i106.rhs = sext i32 %63 to i64
  %iptr__id.i106 = add i64 %iptr__id.i106.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i107 = inttoptr i64 %iptr__id.i106 to double*
  %val__id.i108 = load double* %ptr__id.i107, align 8
  %.lhs343.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %.lhs343 = add i32 %.lhs343.lhs, %mul__Nxy_load461.i
  %64 = shl i32 %.lhs343, 3
  %iptr__id.i99.rhs = sext i32 %64 to i64
  %iptr__id.i99 = add i64 %iptr__id.i99.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i100 = inttoptr i64 %iptr__id.i99 to double*
  %val__id.i101 = load double* %ptr__id.i100, align 8
  %.lhs342 = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %65 = shl i32 %.lhs342, 3
  %iptr__id.i92.rhs = sext i32 %65 to i64
  %iptr__id.i92 = add i64 %iptr__id.i92.rhs, %Ain_load309_ptr2int.i
  %ptr__id.i93 = inttoptr i64 %iptr__id.i92 to double*
  %val__id.i94 = load double* %ptr__id.i93, align 8
  %.lhs341 = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %66 = shl i32 %.lhs341, 3
  %iptr__id.i85.rhs = sext i32 %66 to i64
  %iptr__id.i85 = add i64 %iptr__id.i85.rhs, %Aout_load470_ptr2int.i
  %ptr__id.i86 = inttoptr i64 %iptr__id.i85 to double*
  %val__id.i87 = load double* %ptr__id.i86, align 8
  %.lhs340 = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %67 = shl i32 %.lhs340, 3
  %iptr__id.i80.rhs = sext i32 %67 to i64
  %iptr__id.i80 = add i64 %iptr__id.i80.rhs, %vsq_load488_ptr2int.i
  %ptr__id.i81 = inttoptr i64 %iptr__id.i80 to double*
  %val__id.i82 = load double* %ptr__id.i81, align 8
  %.lhs = extractelement <1 x i32> %add_add_mul_z_load297_Nxy_load298_broadcast_mul_y_load299_Nx_load300_broadcast_x_load301.i, i32 0
  %68 = shl i32 %.lhs, 3
  %iptr__id.i76.rhs = sext i32 %68 to i64
  %iptr__id.i76 = add i64 %iptr__id.i76.rhs, %Aout_load470_ptr2int.i
  %ptr__id.i77 = inttoptr i64 %iptr__id.i76 to double*
  %val__id.i78.lhs.lhs = fmul double %val__id.i94, 2.000000e+00
  %val__id.i78.lhs = fsub double %val__id.i78.lhs.lhs, %val__id.i87
  %val__id.i78.rhs.rhs.lhs.lhs.lhs = fmul double %coef_load303_offset_load.i, %val__id.i227
  %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs.lhs.lhs = fadd double %val__id.i220, %val__id.i213
  %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs.lhs = fadd double %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs.lhs.lhs, %val__id.i206
  %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs = fadd double %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs.lhs, %val__id.i199
  %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs.lhs = fadd double %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs.lhs.lhs, %val__id.i192
  %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs = fadd double %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs.lhs, %val__id.i185
  %val__id.i78.rhs.rhs.lhs.lhs.rhs = fmul double %coef_load314_offset_load.i401, %val__id.i78.rhs.rhs.lhs.lhs.rhs.rhs
  %val__id.i78.rhs.rhs.lhs.lhs = fadd double %val__id.i78.rhs.rhs.lhs.lhs.lhs, %val__id.i78.rhs.rhs.lhs.lhs.rhs
  %val__id.i78.rhs.rhs.lhs.rhs.rhs.lhs.lhs.lhs.lhs = fadd double %val__id.i178, %val__id.i171
  %val__id.i78.rhs.rhs.lhs.rhs.rhs.lhs.lhs.lhs = fadd double %val__id.i78.rhs.rhs.lhs.rhs.rhs.lhs.lhs.lhs.lhs, %val__id.i164
  %val__id.i78.rhs.rhs.lhs.rhs.rhs.lhs.lhs = fadd double %val__id.i78.rhs.rhs.lhs.rhs.rhs.lhs.lhs.lhs, %val__id.i157
  %val__id.i78.rhs.rhs.lhs.rhs.rhs.lhs = fadd double %val__id.i78.rhs.rhs.lhs.rhs.rhs.lhs.lhs, %val__id.i150
  %val__id.i78.rhs.rhs.lhs.rhs.rhs = fadd double %val__id.i78.rhs.rhs.lhs.rhs.rhs.lhs, %val__id.i143
  %val__id.i78.rhs.rhs.lhs.rhs = fmul double %coef_load365_offset_load.i457, %val__id.i78.rhs.rhs.lhs.rhs.rhs
  %val__id.i78.rhs.rhs.lhs = fadd double %val__id.i78.rhs.rhs.lhs.lhs, %val__id.i78.rhs.rhs.lhs.rhs
  %val__id.i78.rhs.rhs.rhs.rhs.lhs.lhs.lhs.lhs = fadd double %val__id.i136, %val__id.i129
  %val__id.i78.rhs.rhs.rhs.rhs.lhs.lhs.lhs = fadd double %val__id.i78.rhs.rhs.rhs.rhs.lhs.lhs.lhs.lhs, %val__id.i122
  %val__id.i78.rhs.rhs.rhs.rhs.lhs.lhs = fadd double %val__id.i78.rhs.rhs.rhs.rhs.lhs.lhs.lhs, %val__id.i115
  %val__id.i78.rhs.rhs.rhs.rhs.lhs = fadd double %val__id.i78.rhs.rhs.rhs.rhs.lhs.lhs, %val__id.i108
  %val__id.i78.rhs.rhs.rhs.rhs = fadd double %val__id.i78.rhs.rhs.rhs.rhs.lhs, %val__id.i101
  %val__id.i78.rhs.rhs.rhs = fmul double %coef_load416_offset_load.i544, %val__id.i78.rhs.rhs.rhs.rhs
  %val__id.i78.rhs.rhs = fadd double %val__id.i78.rhs.rhs.lhs, %val__id.i78.rhs.rhs.rhs
  %val__id.i78.rhs = fmul double %val__id.i78.rhs.rhs, %val__id.i82
  %val__id.i78 = fadd double %val__id.i78.lhs, %val__id.i78.rhs
  store double %val__id.i78, double* %ptr__id.i77, align 8
  br label %foreach_reset27.i
}

define void @loop_stencil_ispc_tasks___uniuniuniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_un_3C_und_3E_(i32 %t0, i32 %t1, i32 %x0, i32 %x1, i32 %y0, i32 %y1, i32 %z0, i32 %z1, i32 %Nx, i32 %Ny, i32 %Nz, double* %coef, double* %vsq, double* %Aeven, double* %Aodd, <1 x i1> %__mask) {
allocas:
  %less_t_load_t1_load94 = icmp slt i32 %t0, %t1
  br i1 %less_t_load_t1_load94, label %for_loop.lr.ph, label %for_exit

for_loop.lr.ph:                                   ; preds = %allocas
  %add_sub_x1_load21_x0_load22_ = sub i32 31, %x0
  %sub_add_sub_x1_load21_x0_load22__ = add i32 %add_sub_x1_load21_x0_load22_, %x1
  %div_sub_add_sub_x1_load21_x0_load22___ = sdiv i32 %sub_add_sub_x1_load21_x0_load22__, 32
  %add_sub_y1_load23_y0_load24_ = sub i32 7, %y0
  %sub_add_sub_y1_load23_y0_load24__ = add i32 %add_sub_y1_load23_y0_load24_, %y1
  %div_sub_add_sub_y1_load23_y0_load24___ = sdiv i32 %sub_add_sub_y1_load23_y0_load24__, 8
  %add_sub_z1_load25_z0_load26_ = sub i32 7, %z0
  %sub_add_sub_z1_load25_z0_load26__ = add i32 %add_sub_z1_load25_z0_load26_, %z1
  %div_sub_add_sub_z1_load25_z0_load26___ = sdiv i32 %sub_add_sub_z1_load25_z0_load26__, 8
  %ntxm1.i = add nsw i32 %div_sub_add_sub_x1_load21_x0_load22___, -1
  %ntxm1d4.i = ashr i32 %ntxm1.i, 2
  %nbx.i = add nsw i32 %ntxm1d4.i, 1
  br label %for_loop

for_loop:                                         ; preds = %if_exit, %for_loop.lr.ph
  %t.095 = phi i32 [ %t0, %for_loop.lr.ph ], [ %t_load78_plus1, %if_exit ]
  %bitop = and i32 %t.095, 1
  %equal_bitop_ = icmp eq i32 %bitop, 0
  %tid.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %and.i = and i32 %tid.i.i, 31
  %cmp.i = icmp eq i32 %and.i, 0
  br i1 %cmp.i, label %if.then.i, label %ISPCGetParamBuffer.exit

if.then.i:                                        ; preds = %for_loop
  %ptri64tmp.i = tail call i64 @cudaGetParameterBuffer(i64 8, i64 72)
  %phitmp.i = inttoptr i64 %ptri64tmp.i to i8*
  br label %ISPCGetParamBuffer.exit

ISPCGetParamBuffer.exit:                          ; preds = %if.then.i, %for_loop
  %ptri64.i = phi i8* [ %phitmp.i, %if.then.i ], [ null, %for_loop ]
  %cmp1 = icmp eq i8* %ptri64.i, null
  br i1 %equal_bitop_, label %if_then, label %if_else

for_exit:                                         ; preds = %if_exit, %allocas
  %0 = tail call i32 @cudaDeviceSynchronize()
  ret void

if_then:                                          ; preds = %ISPCGetParamBuffer.exit
  br i1 %cmp1, label %if_false, label %if_true

if_else:                                          ; preds = %ISPCGetParamBuffer.exit
  br i1 %cmp1, label %if_false62, label %if_true61

if_exit:                                          ; preds = %if.then.i92, %if_false62, %if.then.i83, %if_false
  %1 = tail call i32 @cudaDeviceSynchronize()
  %t_load78_plus1 = add i32 %t.095, 1
  %exitcond = icmp eq i32 %t_load78_plus1, %t1
  br i1 %exitcond, label %for_exit, label %for_loop

if_true:                                          ; preds = %if_then
  %funarg = bitcast i8* %ptri64.i to i32*
  store i32 %x0, i32* %funarg, align 4
  %funarg27 = getelementptr i8* %ptri64.i, i64 4
  %2 = bitcast i8* %funarg27 to i32*
  store i32 %x1, i32* %2, align 4
  %funarg28 = getelementptr i8* %ptri64.i, i64 8
  %3 = bitcast i8* %funarg28 to i32*
  store i32 %y0, i32* %3, align 4
  %funarg29 = getelementptr i8* %ptri64.i, i64 12
  %4 = bitcast i8* %funarg29 to i32*
  store i32 %y1, i32* %4, align 4
  %funarg30 = getelementptr i8* %ptri64.i, i64 16
  %5 = bitcast i8* %funarg30 to i32*
  store i32 %z0, i32* %5, align 4
  %funarg31 = getelementptr i8* %ptri64.i, i64 20
  %6 = bitcast i8* %funarg31 to i32*
  store i32 %z1, i32* %6, align 4
  %funarg32 = getelementptr i8* %ptri64.i, i64 24
  %7 = bitcast i8* %funarg32 to i32*
  store i32 %Nx, i32* %7, align 4
  %funarg33 = getelementptr i8* %ptri64.i, i64 28
  %8 = bitcast i8* %funarg33 to i32*
  store i32 %Ny, i32* %8, align 4
  %funarg34 = getelementptr i8* %ptri64.i, i64 32
  %9 = bitcast i8* %funarg34 to i32*
  store i32 %Nz, i32* %9, align 4
  %funarg35 = getelementptr i8* %ptri64.i, i64 40
  %10 = bitcast i8* %funarg35 to double**
  store double* %coef, double** %10, align 8
  %funarg36 = getelementptr i8* %ptri64.i, i64 48
  %11 = bitcast i8* %funarg36 to double**
  store double* %vsq, double** %11, align 8
  %funarg37 = getelementptr i8* %ptri64.i, i64 56
  %12 = bitcast i8* %funarg37 to double**
  store double* %Aeven, double** %12, align 8
  %funarg38 = getelementptr i8* %ptri64.i, i64 64
  %13 = bitcast i8* %funarg38 to double**
  store double* %Aodd, double** %13, align 8
  br label %if_false

if_false:                                         ; preds = %if_true, %if_then
  %tid.i.i80 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %and.i81 = and i32 %tid.i.i80, 31
  %cmp.i82 = icmp eq i32 %and.i81, 0
  br i1 %cmp.i82, label %if.then.i83, label %if_exit

if.then.i83:                                      ; preds = %if_false
  %args_i64.i = ptrtoint i8* %ptri64.i to i64
  %res_tmp.i = tail call i32 asm sideeffect "{\0A     .param .b64 param0;\0A     st.param.b64\09[param0+0], $1;\0A     .param .b64 param1;\0A     st.param.b64\09[param1+0], $2;\0A     .param .align 4 .b8 param2[12];\0A     st.param.b32\09[param2+0], $3; \0A     st.param.b32\09[param2+4], $4; \0A     st.param.b32\09[param2+8], $5; \0A     .param .align 4 .b8 param3[12];\0A     st.param.b32\09[param3+0], $6; \0A     st.param.b32\09[param3+4], $7; \0A     st.param.b32\09[param3+8], $8; \0A     .param .b32 param4;\0A     st.param.b32\09[param4+0], $9; \0A     .param .b64 param5;\0A     st.param.b64\09[param5+0], $10; \0A\0A     .param .b32 retval0;\0A     call.uni (retval0), \0A       cudaLaunchDevice,\0A       (\0A        param0, \0A        param1, \0A        param2, \0A        param3, \0A        param4, \0A        param5\0A       );\0A     ld.param.b32\09$0, [retval0+0];\0A  }\0A  ", "=r, l,l, r,r,r, r,r,r, r,l"(i64 ptrtoint (void (i32, i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*)* @stencil_step_task___UM_uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_ to i64), i64 %args_i64.i, i32 %nbx.i, i32 %div_sub_add_sub_y1_load23_y0_load24___, i32 %div_sub_add_sub_z1_load25_z0_load26___, i32 128, i32 1, i32 1, i32 0, i64 0)
  br label %if_exit

if_true61:                                        ; preds = %if_else
  %funarg64 = bitcast i8* %ptri64.i to i32*
  store i32 %x0, i32* %funarg64, align 4
  %funarg65 = getelementptr i8* %ptri64.i, i64 4
  %14 = bitcast i8* %funarg65 to i32*
  store i32 %x1, i32* %14, align 4
  %funarg66 = getelementptr i8* %ptri64.i, i64 8
  %15 = bitcast i8* %funarg66 to i32*
  store i32 %y0, i32* %15, align 4
  %funarg67 = getelementptr i8* %ptri64.i, i64 12
  %16 = bitcast i8* %funarg67 to i32*
  store i32 %y1, i32* %16, align 4
  %funarg68 = getelementptr i8* %ptri64.i, i64 16
  %17 = bitcast i8* %funarg68 to i32*
  store i32 %z0, i32* %17, align 4
  %funarg69 = getelementptr i8* %ptri64.i, i64 20
  %18 = bitcast i8* %funarg69 to i32*
  store i32 %z1, i32* %18, align 4
  %funarg70 = getelementptr i8* %ptri64.i, i64 24
  %19 = bitcast i8* %funarg70 to i32*
  store i32 %Nx, i32* %19, align 4
  %funarg71 = getelementptr i8* %ptri64.i, i64 28
  %20 = bitcast i8* %funarg71 to i32*
  store i32 %Ny, i32* %20, align 4
  %funarg72 = getelementptr i8* %ptri64.i, i64 32
  %21 = bitcast i8* %funarg72 to i32*
  store i32 %Nz, i32* %21, align 4
  %funarg73 = getelementptr i8* %ptri64.i, i64 40
  %22 = bitcast i8* %funarg73 to double**
  store double* %coef, double** %22, align 8
  %funarg74 = getelementptr i8* %ptri64.i, i64 48
  %23 = bitcast i8* %funarg74 to double**
  store double* %vsq, double** %23, align 8
  %funarg75 = getelementptr i8* %ptri64.i, i64 56
  %24 = bitcast i8* %funarg75 to double**
  store double* %Aodd, double** %24, align 8
  %funarg76 = getelementptr i8* %ptri64.i, i64 64
  %25 = bitcast i8* %funarg76 to double**
  store double* %Aeven, double** %25, align 8
  br label %if_false62

if_false62:                                       ; preds = %if_true61, %if_else
  %tid.i.i84 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %and.i85 = and i32 %tid.i.i84, 31
  %cmp.i86 = icmp eq i32 %and.i85, 0
  br i1 %cmp.i86, label %if.then.i92, label %if_exit

if.then.i92:                                      ; preds = %if_false62
  %args_i64.i90 = ptrtoint i8* %ptri64.i to i64
  %res_tmp.i91 = tail call i32 asm sideeffect "{\0A     .param .b64 param0;\0A     st.param.b64\09[param0+0], $1;\0A     .param .b64 param1;\0A     st.param.b64\09[param1+0], $2;\0A     .param .align 4 .b8 param2[12];\0A     st.param.b32\09[param2+0], $3; \0A     st.param.b32\09[param2+4], $4; \0A     st.param.b32\09[param2+8], $5; \0A     .param .align 4 .b8 param3[12];\0A     st.param.b32\09[param3+0], $6; \0A     st.param.b32\09[param3+4], $7; \0A     st.param.b32\09[param3+8], $8; \0A     .param .b32 param4;\0A     st.param.b32\09[param4+0], $9; \0A     .param .b64 param5;\0A     st.param.b64\09[param5+0], $10; \0A\0A     .param .b32 retval0;\0A     call.uni (retval0), \0A       cudaLaunchDevice,\0A       (\0A        param0, \0A        param1, \0A        param2, \0A        param3, \0A        param4, \0A        param5\0A       );\0A     ld.param.b32\09$0, [retval0+0];\0A  }\0A  ", "=r, l,l, r,r,r, r,r,r, r,l"(i64 ptrtoint (void (i32, i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*)* @stencil_step_task___UM_uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_ to i64), i64 %args_i64.i90, i32 %nbx.i, i32 %div_sub_add_sub_y1_load23_y0_load24___, i32 %div_sub_add_sub_z1_load25_z0_load26___, i32 128, i32 1, i32 1, i32 0, i64 0)
  br label %if_exit
}

define void @loop_stencil_ispc_tasks(i32 %t0, i32 %t1, i32 %x0, i32 %x1, i32 %y0, i32 %y1, i32 %z0, i32 %z1, i32 %Nx, i32 %Ny, i32 %Nz, double* %coef, double* %vsq, double* %Aeven, double* %Aodd) {
allocas:
  %less_t_load_t1_load94 = icmp slt i32 %t0, %t1
  br i1 %less_t_load_t1_load94, label %for_loop.lr.ph, label %for_exit

for_loop.lr.ph:                                   ; preds = %allocas
  %add_sub_x1_load21_x0_load22_ = sub i32 31, %x0
  %sub_add_sub_x1_load21_x0_load22__ = add i32 %add_sub_x1_load21_x0_load22_, %x1
  %div_sub_add_sub_x1_load21_x0_load22___ = sdiv i32 %sub_add_sub_x1_load21_x0_load22__, 32
  %add_sub_y1_load23_y0_load24_ = sub i32 7, %y0
  %sub_add_sub_y1_load23_y0_load24__ = add i32 %add_sub_y1_load23_y0_load24_, %y1
  %div_sub_add_sub_y1_load23_y0_load24___ = sdiv i32 %sub_add_sub_y1_load23_y0_load24__, 8
  %add_sub_z1_load25_z0_load26_ = sub i32 7, %z0
  %sub_add_sub_z1_load25_z0_load26__ = add i32 %add_sub_z1_load25_z0_load26_, %z1
  %div_sub_add_sub_z1_load25_z0_load26___ = sdiv i32 %sub_add_sub_z1_load25_z0_load26__, 8
  %ntxm1.i = add nsw i32 %div_sub_add_sub_x1_load21_x0_load22___, -1
  %ntxm1d4.i = ashr i32 %ntxm1.i, 2
  %nbx.i = add nsw i32 %ntxm1d4.i, 1
  br label %for_loop

for_loop:                                         ; preds = %if_exit, %for_loop.lr.ph
  %t.095 = phi i32 [ %t0, %for_loop.lr.ph ], [ %t_load78_plus1, %if_exit ]
  %bitop = and i32 %t.095, 1
  %equal_bitop_ = icmp eq i32 %bitop, 0
  %tid.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %and.i = and i32 %tid.i.i, 31
  %cmp.i = icmp eq i32 %and.i, 0
  br i1 %cmp.i, label %if.then.i, label %ISPCGetParamBuffer.exit

if.then.i:                                        ; preds = %for_loop
  %ptri64tmp.i = tail call i64 @cudaGetParameterBuffer(i64 8, i64 72)
  %phitmp.i = inttoptr i64 %ptri64tmp.i to i8*
  br label %ISPCGetParamBuffer.exit

ISPCGetParamBuffer.exit:                          ; preds = %if.then.i, %for_loop
  %ptri64.i = phi i8* [ %phitmp.i, %if.then.i ], [ null, %for_loop ]
  %cmp1 = icmp eq i8* %ptri64.i, null
  br i1 %equal_bitop_, label %if_then, label %if_else

for_exit:                                         ; preds = %if_exit, %allocas
  %0 = tail call i32 @cudaDeviceSynchronize()
  ret void

if_then:                                          ; preds = %ISPCGetParamBuffer.exit
  br i1 %cmp1, label %if_false, label %if_true

if_else:                                          ; preds = %ISPCGetParamBuffer.exit
  br i1 %cmp1, label %if_false62, label %if_true61

if_exit:                                          ; preds = %if.then.i92, %if_false62, %if.then.i83, %if_false
  %1 = tail call i32 @cudaDeviceSynchronize()
  %t_load78_plus1 = add i32 %t.095, 1
  %exitcond = icmp eq i32 %t_load78_plus1, %t1
  br i1 %exitcond, label %for_exit, label %for_loop

if_true:                                          ; preds = %if_then
  %funarg = bitcast i8* %ptri64.i to i32*
  store i32 %x0, i32* %funarg, align 4
  %funarg27 = getelementptr i8* %ptri64.i, i64 4
  %2 = bitcast i8* %funarg27 to i32*
  store i32 %x1, i32* %2, align 4
  %funarg28 = getelementptr i8* %ptri64.i, i64 8
  %3 = bitcast i8* %funarg28 to i32*
  store i32 %y0, i32* %3, align 4
  %funarg29 = getelementptr i8* %ptri64.i, i64 12
  %4 = bitcast i8* %funarg29 to i32*
  store i32 %y1, i32* %4, align 4
  %funarg30 = getelementptr i8* %ptri64.i, i64 16
  %5 = bitcast i8* %funarg30 to i32*
  store i32 %z0, i32* %5, align 4
  %funarg31 = getelementptr i8* %ptri64.i, i64 20
  %6 = bitcast i8* %funarg31 to i32*
  store i32 %z1, i32* %6, align 4
  %funarg32 = getelementptr i8* %ptri64.i, i64 24
  %7 = bitcast i8* %funarg32 to i32*
  store i32 %Nx, i32* %7, align 4
  %funarg33 = getelementptr i8* %ptri64.i, i64 28
  %8 = bitcast i8* %funarg33 to i32*
  store i32 %Ny, i32* %8, align 4
  %funarg34 = getelementptr i8* %ptri64.i, i64 32
  %9 = bitcast i8* %funarg34 to i32*
  store i32 %Nz, i32* %9, align 4
  %funarg35 = getelementptr i8* %ptri64.i, i64 40
  %10 = bitcast i8* %funarg35 to double**
  store double* %coef, double** %10, align 8
  %funarg36 = getelementptr i8* %ptri64.i, i64 48
  %11 = bitcast i8* %funarg36 to double**
  store double* %vsq, double** %11, align 8
  %funarg37 = getelementptr i8* %ptri64.i, i64 56
  %12 = bitcast i8* %funarg37 to double**
  store double* %Aeven, double** %12, align 8
  %funarg38 = getelementptr i8* %ptri64.i, i64 64
  %13 = bitcast i8* %funarg38 to double**
  store double* %Aodd, double** %13, align 8
  br label %if_false

if_false:                                         ; preds = %if_true, %if_then
  %tid.i.i80 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %and.i81 = and i32 %tid.i.i80, 31
  %cmp.i82 = icmp eq i32 %and.i81, 0
  br i1 %cmp.i82, label %if.then.i83, label %if_exit

if.then.i83:                                      ; preds = %if_false
  %args_i64.i = ptrtoint i8* %ptri64.i to i64
  %res_tmp.i = tail call i32 asm sideeffect "{\0A     .param .b64 param0;\0A     st.param.b64\09[param0+0], $1;\0A     .param .b64 param1;\0A     st.param.b64\09[param1+0], $2;\0A     .param .align 4 .b8 param2[12];\0A     st.param.b32\09[param2+0], $3; \0A     st.param.b32\09[param2+4], $4; \0A     st.param.b32\09[param2+8], $5; \0A     .param .align 4 .b8 param3[12];\0A     st.param.b32\09[param3+0], $6; \0A     st.param.b32\09[param3+4], $7; \0A     st.param.b32\09[param3+8], $8; \0A     .param .b32 param4;\0A     st.param.b32\09[param4+0], $9; \0A     .param .b64 param5;\0A     st.param.b64\09[param5+0], $10; \0A\0A     .param .b32 retval0;\0A     call.uni (retval0), \0A       cudaLaunchDevice,\0A       (\0A        param0, \0A        param1, \0A        param2, \0A        param3, \0A        param4, \0A        param5\0A       );\0A     ld.param.b32\09$0, [retval0+0];\0A  }\0A  ", "=r, l,l, r,r,r, r,r,r, r,l"(i64 ptrtoint (void (i32, i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*)* @stencil_step_task___UM_uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_ to i64), i64 %args_i64.i, i32 %nbx.i, i32 %div_sub_add_sub_y1_load23_y0_load24___, i32 %div_sub_add_sub_z1_load25_z0_load26___, i32 128, i32 1, i32 1, i32 0, i64 0)
  br label %if_exit

if_true61:                                        ; preds = %if_else
  %funarg64 = bitcast i8* %ptri64.i to i32*
  store i32 %x0, i32* %funarg64, align 4
  %funarg65 = getelementptr i8* %ptri64.i, i64 4
  %14 = bitcast i8* %funarg65 to i32*
  store i32 %x1, i32* %14, align 4
  %funarg66 = getelementptr i8* %ptri64.i, i64 8
  %15 = bitcast i8* %funarg66 to i32*
  store i32 %y0, i32* %15, align 4
  %funarg67 = getelementptr i8* %ptri64.i, i64 12
  %16 = bitcast i8* %funarg67 to i32*
  store i32 %y1, i32* %16, align 4
  %funarg68 = getelementptr i8* %ptri64.i, i64 16
  %17 = bitcast i8* %funarg68 to i32*
  store i32 %z0, i32* %17, align 4
  %funarg69 = getelementptr i8* %ptri64.i, i64 20
  %18 = bitcast i8* %funarg69 to i32*
  store i32 %z1, i32* %18, align 4
  %funarg70 = getelementptr i8* %ptri64.i, i64 24
  %19 = bitcast i8* %funarg70 to i32*
  store i32 %Nx, i32* %19, align 4
  %funarg71 = getelementptr i8* %ptri64.i, i64 28
  %20 = bitcast i8* %funarg71 to i32*
  store i32 %Ny, i32* %20, align 4
  %funarg72 = getelementptr i8* %ptri64.i, i64 32
  %21 = bitcast i8* %funarg72 to i32*
  store i32 %Nz, i32* %21, align 4
  %funarg73 = getelementptr i8* %ptri64.i, i64 40
  %22 = bitcast i8* %funarg73 to double**
  store double* %coef, double** %22, align 8
  %funarg74 = getelementptr i8* %ptri64.i, i64 48
  %23 = bitcast i8* %funarg74 to double**
  store double* %vsq, double** %23, align 8
  %funarg75 = getelementptr i8* %ptri64.i, i64 56
  %24 = bitcast i8* %funarg75 to double**
  store double* %Aodd, double** %24, align 8
  %funarg76 = getelementptr i8* %ptri64.i, i64 64
  %25 = bitcast i8* %funarg76 to double**
  store double* %Aeven, double** %25, align 8
  br label %if_false62

if_false62:                                       ; preds = %if_true61, %if_else
  %tid.i.i84 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %and.i85 = and i32 %tid.i.i84, 31
  %cmp.i86 = icmp eq i32 %and.i85, 0
  br i1 %cmp.i86, label %if.then.i92, label %if_exit

if.then.i92:                                      ; preds = %if_false62
  %args_i64.i90 = ptrtoint i8* %ptri64.i to i64
  %res_tmp.i91 = tail call i32 asm sideeffect "{\0A     .param .b64 param0;\0A     st.param.b64\09[param0+0], $1;\0A     .param .b64 param1;\0A     st.param.b64\09[param1+0], $2;\0A     .param .align 4 .b8 param2[12];\0A     st.param.b32\09[param2+0], $3; \0A     st.param.b32\09[param2+4], $4; \0A     st.param.b32\09[param2+8], $5; \0A     .param .align 4 .b8 param3[12];\0A     st.param.b32\09[param3+0], $6; \0A     st.param.b32\09[param3+4], $7; \0A     st.param.b32\09[param3+8], $8; \0A     .param .b32 param4;\0A     st.param.b32\09[param4+0], $9; \0A     .param .b64 param5;\0A     st.param.b64\09[param5+0], $10; \0A\0A     .param .b32 retval0;\0A     call.uni (retval0), \0A       cudaLaunchDevice,\0A       (\0A        param0, \0A        param1, \0A        param2, \0A        param3, \0A        param4, \0A        param5\0A       );\0A     ld.param.b32\09$0, [retval0+0];\0A  }\0A  ", "=r, l,l, r,r,r, r,r,r, r,l"(i64 ptrtoint (void (i32, i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*)* @stencil_step_task___UM_uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_ to i64), i64 %args_i64.i90, i32 %nbx.i, i32 %div_sub_add_sub_y1_load23_y0_load24___, i32 %div_sub_add_sub_z1_load25_z0_load26___, i32 128, i32 1, i32 1, i32 0, i64 0)
  br label %if_exit
}

!llvm.ident = !{!0}
!nvvm.annotations = !{!1, !2}

!0 = metadata !{metadata !"clang version 3.4 (trunk 194723)"}
!1 = metadata !{void (i32, i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*)* @stencil_step_task___UM_uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_, metadata !"kernel", i32 1}
!2 = metadata !{void (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*)* @loop_stencil_ispc_tasks, metadata !"kernel", i32 1}
