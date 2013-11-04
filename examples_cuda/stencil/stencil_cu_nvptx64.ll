; ModuleID = 'stencil_cu_nvptx64.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64"

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() #0

; Function Attrs: nounwind
define void @stencil_step_task(i32 %x0, i32 %x1, i32 %y0, i32 %y1, i32 %z0, i32 %Nx, i32 %Ny, i32 %Nz, double* nocapture %coef, double* %vsq, double* %Ain, double* %Aout) #1 {
allocas:
  %bid.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2
  %add_z0_load_calltmp = add i32 %bid.i.i, %z0
  %bid.i.i21 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2
  %add_z0_load15_calltmp18 = add i32 %z0, 1
  %add_add_z0_load15_calltmp18_ = add i32 %add_z0_load15_calltmp18, %bid.i.i21
  %mul_Nx_load_Ny_load.i = mul i32 %Ny, %Nx
  %coef_load_offset_load.i = load double* %coef, align 8
  %coef_load16_offset.i = getelementptr double* %coef, i64 1
  %coef_load16_offset_load.i = load double* %coef_load16_offset.i, align 8
  %coef_load19_offset.i = getelementptr double* %coef, i64 2
  %coef_load19_offset_load.i = load double* %coef_load19_offset.i, align 8
  %coef_load22_offset.i = getelementptr double* %coef, i64 3
  %coef_load22_offset_load.i = load double* %coef_load22_offset.i, align 8
  %less_z_load_z1_load.i161 = icmp slt i32 %add_z0_load_calltmp, %add_add_z0_load15_calltmp18_
  br i1 %less_z_load_z1_load.i161, label %for_test28.i.preheader.lr.ph, label %stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_.exit

for_test28.i.preheader.lr.ph:                     ; preds = %allocas
  %less_y_load_y1_load.i159 = icmp slt i32 %y0, %y1
  %less_xb_load_x1_load.i157 = icmp slt i32 %x0, %x1
  %x1_load199_broadcast_init.i = insertelement <1 x i32> undef, i32 %x1, i32 0
  %mul__Nx_load119.i = shl i32 %Nx, 1
  %mul__Nx_load167.i = mul i32 %Nx, 3
  %mul__Nx_load127.i = mul i32 %Nx, -2
  %Ain_load65_ptr2int.i = ptrtoint double* %Ain to i64
  %mul__Nx_load175.i = mul i32 %Nx, -3
  %mul__Nxy_load136.i = shl i32 %mul_Nx_load_Ny_load.i, 1
  %mul__Nxy_load184.i = mul i32 %mul_Nx_load_Ny_load.i, 3
  %mul__Nxy_load144.i = mul i32 %mul_Nx_load_Ny_load.i, -2
  %mul__Nxy_load192.i = mul i32 %mul_Nx_load_Ny_load.i, -3
  %Aout_load_ptr2int.i = ptrtoint double* %Aout to i64
  %vsq_load_ptr2int.i = ptrtoint double* %vsq to i64
  %0 = add i32 %bid.i.i21, %z0
  br label %for_test28.i.preheader

for_test28.i.preheader:                           ; preds = %for_exit31.i, %for_test28.i.preheader.lr.ph
  %z.0.i162 = phi i32 [ %add_z0_load_calltmp, %for_test28.i.preheader.lr.ph ], [ %z_load245_plus1.i, %for_exit31.i ]
  br i1 %less_y_load_y1_load.i159, label %for_test35.i.preheader.lr.ph, label %for_exit31.i

for_test35.i.preheader.lr.ph:                     ; preds = %for_test28.i.preheader
  %mul_z_load45_Nxy_load.i = mul i32 %z.0.i162, %mul_Nx_load_Ny_load.i
  br i1 %less_xb_load_x1_load.i157, label %for_loop37.i.lr.ph.us, label %for_exit31.i

for_exit38.i.us:                                  ; preds = %safe_if_after_true.i.us
  %y_load244_plus1.i.us = add i32 %y.0.i160.us, 1
  %exitcond = icmp eq i32 %y_load244_plus1.i.us, %y1
  br i1 %exitcond, label %for_exit31.i, label %for_loop37.i.lr.ph.us

for_loop37.i.us:                                  ; preds = %for_loop37.i.lr.ph.us, %safe_if_after_true.i.us
  %xb.0.i158.us = phi i32 [ %x0, %for_loop37.i.lr.ph.us ], [ %add_xb_load243_calltmp241.i.us, %safe_if_after_true.i.us ]
  %tid.i.i.i.us = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
  %tid.i.i.i.i.us = tail call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #2
  %sub_calltmp3_.i.i.us = add i32 %tid.i.i.i.i.us, -1
  %bitop.i.i.us = and i32 %sub_calltmp3_.i.i.us, %tid.i.i.i.us
  %add_xb_load42_calltmp.i.us = add i32 %bitop.i.i.us, %xb.0.i158.us
  %add_xb_load42_calltmp_broadcast_init.i.us = insertelement <1 x i32> undef, i32 %add_xb_load42_calltmp.i.us, i32 0
  %less_x_load198_x1_load199_broadcast.i.us = icmp slt <1 x i32> %add_xb_load42_calltmp_broadcast_init.i.us, %x1_load199_broadcast_init.i
  %v.i.i.us = extractelement <1 x i1> %less_x_load198_x1_load199_broadcast.i.us, i32 0
  br i1 %v.i.i.us, label %pl_dolane.i.us, label %safe_if_after_true.i.us

pl_dolane.i.us:                                   ; preds = %for_loop37.i.us
  %.lhs.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %add_xb_load42_calltmp.i.us
  %.lhs.us = shl i32 %.lhs.lhs.us, 3
  %1 = add i32 %.lhs.us, -8
  %iptr__id.i.rhs.us = sext i32 %1 to i64
  %iptr__id.i.us = add i64 %iptr__id.i.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i.us = inttoptr i64 %iptr__id.i.us to double*
  %val__id.i.us = load double* %ptr__id.i.us, align 8
  %2 = add i32 %.lhs.us, 8
  %iptr__id.i130.rhs.us = sext i32 %2 to i64
  %iptr__id.i130.us = add i64 %iptr__id.i130.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i131.us = inttoptr i64 %iptr__id.i130.us to double*
  %val__id.i132.us = load double* %ptr__id.i131.us, align 8
  %3 = add i32 %.lhs.us, -16
  %iptr__id.i125.rhs.us = sext i32 %3 to i64
  %iptr__id.i125.us = add i64 %iptr__id.i125.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i126.us = inttoptr i64 %iptr__id.i125.us to double*
  %val__id.i127.us = load double* %ptr__id.i126.us, align 8
  %4 = add i32 %.lhs.us, 16
  %iptr__id.i120.rhs.us = sext i32 %4 to i64
  %iptr__id.i120.us = add i64 %iptr__id.i120.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i121.us = inttoptr i64 %iptr__id.i120.us to double*
  %val__id.i122.us = load double* %ptr__id.i121.us, align 8
  %.lhs138.us = add i32 %.lhs138.lhs.us, %add_xb_load42_calltmp.i.us
  %5 = shl i32 %.lhs138.us, 3
  %iptr__id.i115.rhs.us = sext i32 %5 to i64
  %iptr__id.i115.us = add i64 %iptr__id.i115.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i116.us = inttoptr i64 %iptr__id.i115.us to double*
  %val__id.i117.us = load double* %ptr__id.i116.us, align 8
  %6 = add i32 %.lhs.us, -24
  %iptr__id.i110.rhs.us = sext i32 %6 to i64
  %iptr__id.i110.us = add i64 %iptr__id.i110.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i111.us = inttoptr i64 %iptr__id.i110.us to double*
  %val__id.i112.us = load double* %ptr__id.i111.us, align 8
  %7 = add i32 %.lhs.us, 24
  %iptr__id.i105.rhs.us = sext i32 %7 to i64
  %iptr__id.i105.us = add i64 %iptr__id.i105.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i106.us = inttoptr i64 %iptr__id.i105.us to double*
  %val__id.i107.us = load double* %ptr__id.i106.us, align 8
  %.lhs141.us = add i32 %.lhs141.lhs.us, %add_xb_load42_calltmp.i.us
  %8 = shl i32 %.lhs141.us, 3
  %iptr__id.i100.rhs.us = sext i32 %8 to i64
  %iptr__id.i100.us = add i64 %iptr__id.i100.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i101.us = inttoptr i64 %iptr__id.i100.us to double*
  %val__id.i102.us = load double* %ptr__id.i101.us, align 8
  %.lhs142.us = add i32 %.lhs142.lhs.us, %add_xb_load42_calltmp.i.us
  %9 = shl i32 %.lhs142.us, 3
  %iptr__id.i95.rhs.us = sext i32 %9 to i64
  %iptr__id.i95.us = add i64 %iptr__id.i95.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i96.us = inttoptr i64 %iptr__id.i95.us to double*
  %val__id.i97.us = load double* %ptr__id.i96.us, align 8
  %.lhs143.us = add i32 %.lhs143.lhs.us, %add_xb_load42_calltmp.i.us
  %10 = shl i32 %.lhs143.us, 3
  %iptr__id.i90.rhs.us = sext i32 %10 to i64
  %iptr__id.i90.us = add i64 %iptr__id.i90.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i91.us = inttoptr i64 %iptr__id.i90.us to double*
  %val__id.i92.us = load double* %ptr__id.i91.us, align 8
  %.lhs144.us = add i32 %.lhs144.lhs.us, %add_xb_load42_calltmp.i.us
  %11 = shl i32 %.lhs144.us, 3
  %iptr__id.i85.rhs.us = sext i32 %11 to i64
  %iptr__id.i85.us = add i64 %iptr__id.i85.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i86.us = inttoptr i64 %iptr__id.i85.us to double*
  %val__id.i87.us = load double* %ptr__id.i86.us, align 8
  %.lhs145.us = add i32 %.lhs145.lhs.us, %add_xb_load42_calltmp.i.us
  %12 = shl i32 %.lhs145.us, 3
  %iptr__id.i80.rhs.us = sext i32 %12 to i64
  %iptr__id.i80.us = add i64 %iptr__id.i80.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i81.us = inttoptr i64 %iptr__id.i80.us to double*
  %val__id.i82.us = load double* %ptr__id.i81.us, align 8
  %.lhs146.us = add i32 %.lhs146.lhs.us, %add_xb_load42_calltmp.i.us
  %13 = shl i32 %.lhs146.us, 3
  %iptr__id.i75.rhs.us = sext i32 %13 to i64
  %iptr__id.i75.us = add i64 %iptr__id.i75.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i76.us = inttoptr i64 %iptr__id.i75.us to double*
  %val__id.i77.us = load double* %ptr__id.i76.us, align 8
  %.lhs147.us = add i32 %.lhs147.lhs.us, %add_xb_load42_calltmp.i.us
  %14 = shl i32 %.lhs147.us, 3
  %iptr__id.i70.rhs.us = sext i32 %14 to i64
  %iptr__id.i70.us = add i64 %iptr__id.i70.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i71.us = inttoptr i64 %iptr__id.i70.us to double*
  %val__id.i72.us = load double* %ptr__id.i71.us, align 8
  %.lhs148.us = add i32 %.lhs148.lhs.us, %add_xb_load42_calltmp.i.us
  %15 = shl i32 %.lhs148.us, 3
  %iptr__id.i65.rhs.us = sext i32 %15 to i64
  %iptr__id.i65.us = add i64 %iptr__id.i65.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i66.us = inttoptr i64 %iptr__id.i65.us to double*
  %val__id.i67.us = load double* %ptr__id.i66.us, align 8
  %.lhs149.us = add i32 %.lhs149.lhs.us, %add_xb_load42_calltmp.i.us
  %16 = shl i32 %.lhs149.us, 3
  %iptr__id.i60.rhs.us = sext i32 %16 to i64
  %iptr__id.i60.us = add i64 %iptr__id.i60.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i61.us = inttoptr i64 %iptr__id.i60.us to double*
  %val__id.i62.us = load double* %ptr__id.i61.us, align 8
  %.lhs150.us = add i32 %.lhs150.lhs.us, %add_xb_load42_calltmp.i.us
  %17 = shl i32 %.lhs150.us, 3
  %iptr__id.i55.rhs.us = sext i32 %17 to i64
  %iptr__id.i55.us = add i64 %iptr__id.i55.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i56.us = inttoptr i64 %iptr__id.i55.us to double*
  %val__id.i57.us = load double* %ptr__id.i56.us, align 8
  %.lhs151.us = add i32 %add_xb_load42_calltmp.i.us, %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us
  %18 = shl i32 %.lhs151.us, 3
  %iptr__id.i50.rhs.us = sext i32 %18 to i64
  %iptr__id.i50.us = add i64 %iptr__id.i50.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i51.us = inttoptr i64 %iptr__id.i50.us to double*
  %val__id.i52.us = load double* %ptr__id.i51.us, align 8
  %.lhs152.us = add i32 %.lhs152.lhs.us, %add_xb_load42_calltmp.i.us
  %19 = shl i32 %.lhs152.us, 3
  %iptr__id.i45.rhs.us = sext i32 %19 to i64
  %iptr__id.i45.us = add i64 %iptr__id.i45.rhs.us, %Ain_load65_ptr2int.i
  %ptr__id.i46.us = inttoptr i64 %iptr__id.i45.us to double*
  %val__id.i47.us = load double* %ptr__id.i46.us, align 8
  %val__id.i41.us = load double* %ptr__id.i51.us, align 8
  %iptr__id.i32.us = add i64 %iptr__id.i50.rhs.us, %Aout_load_ptr2int.i
  %ptr__id.i33.us = inttoptr i64 %iptr__id.i32.us to double*
  %val__id.i34.us = load double* %ptr__id.i33.us, align 8
  %iptr__id.i27.rhs.us = sext i32 %.lhs.us to i64
  %iptr__id.i27.us = add i64 %iptr__id.i27.rhs.us, %vsq_load_ptr2int.i
  %ptr__id.i28.us = inttoptr i64 %iptr__id.i27.us to double*
  %val__id.i29.us = load double* %ptr__id.i28.us, align 8
  %iptr__id.i23.us = add i64 %iptr__id.i50.rhs.us, %Aout_load_ptr2int.i
  %ptr__id.i24.us = inttoptr i64 %iptr__id.i23.us to double*
  %val__id.i25.lhs.us.lhs = fmul double %val__id.i41.us, 2.000000e+00
  %val__id.i25.lhs.us = fsub double %val__id.i25.lhs.us.lhs, %val__id.i34.us
  %val__id.i25.rhs.rhs.lhs.lhs.rhs.lhs.lhs.lhs.lhs.us = fadd double %val__id.i127.us, %val__id.i122.us
  %val__id.i25.rhs.rhs.lhs.lhs.rhs.lhs.lhs.lhs.us = fadd double %val__id.i25.rhs.rhs.lhs.lhs.rhs.lhs.lhs.lhs.lhs.us, %val__id.i102.us
  %val__id.i25.rhs.rhs.lhs.lhs.rhs.lhs.lhs.us = fadd double %val__id.i25.rhs.rhs.lhs.lhs.rhs.lhs.lhs.lhs.us, %val__id.i87.us
  %val__id.i25.rhs.rhs.lhs.lhs.rhs.lhs.us = fadd double %val__id.i25.rhs.rhs.lhs.lhs.rhs.lhs.lhs.us, %val__id.i72.us
  %val__id.i25.rhs.rhs.lhs.lhs.rhs.us = fadd double %val__id.i25.rhs.rhs.lhs.lhs.rhs.lhs.us, %val__id.i57.us
  %val__id.i25.rhs.rhs.lhs.lhs.us = fmul double %coef_load19_offset_load.i, %val__id.i25.rhs.rhs.lhs.lhs.rhs.us
  %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.lhs.lhs.lhs.lhs.us = fadd double %val__id.i.us, %val__id.i132.us
  %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.lhs.lhs.lhs.us = fadd double %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.lhs.lhs.lhs.lhs.us, %val__id.i117.us
  %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.lhs.lhs.us = fadd double %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.lhs.lhs.lhs.us, %val__id.i97.us
  %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.lhs.us = fadd double %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.lhs.lhs.us, %val__id.i82.us
  %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.us = fadd double %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.lhs.us, %val__id.i67.us
  %val__id.i25.rhs.rhs.lhs.rhs.lhs.us = fmul double %coef_load16_offset_load.i, %val__id.i25.rhs.rhs.lhs.rhs.lhs.rhs.us
  %val__id.i25.rhs.rhs.lhs.rhs.rhs.us = fmul double %coef_load_offset_load.i, %val__id.i52.us
  %val__id.i25.rhs.rhs.lhs.rhs.us = fadd double %val__id.i25.rhs.rhs.lhs.rhs.lhs.us, %val__id.i25.rhs.rhs.lhs.rhs.rhs.us
  %val__id.i25.rhs.rhs.lhs.us = fadd double %val__id.i25.rhs.rhs.lhs.lhs.us, %val__id.i25.rhs.rhs.lhs.rhs.us
  %val__id.i25.rhs.rhs.rhs.rhs.lhs.lhs.lhs.lhs.us = fadd double %val__id.i112.us, %val__id.i107.us
  %val__id.i25.rhs.rhs.rhs.rhs.lhs.lhs.lhs.us = fadd double %val__id.i25.rhs.rhs.rhs.rhs.lhs.lhs.lhs.lhs.us, %val__id.i92.us
  %val__id.i25.rhs.rhs.rhs.rhs.lhs.lhs.us = fadd double %val__id.i25.rhs.rhs.rhs.rhs.lhs.lhs.lhs.us, %val__id.i77.us
  %val__id.i25.rhs.rhs.rhs.rhs.lhs.us = fadd double %val__id.i25.rhs.rhs.rhs.rhs.lhs.lhs.us, %val__id.i62.us
  %val__id.i25.rhs.rhs.rhs.rhs.us = fadd double %val__id.i25.rhs.rhs.rhs.rhs.lhs.us, %val__id.i47.us
  %val__id.i25.rhs.rhs.rhs.us = fmul double %coef_load22_offset_load.i, %val__id.i25.rhs.rhs.rhs.rhs.us
  %val__id.i25.rhs.rhs.us = fadd double %val__id.i25.rhs.rhs.lhs.us, %val__id.i25.rhs.rhs.rhs.us
  %val__id.i25.rhs.us = fmul double %val__id.i25.rhs.rhs.us, %val__id.i29.us
  %val__id.i25.us = fadd double %val__id.i25.lhs.us, %val__id.i25.rhs.us
  store double %val__id.i25.us, double* %ptr__id.i24.us, align 8
  br label %safe_if_after_true.i.us

safe_if_after_true.i.us:                          ; preds = %pl_dolane.i.us, %for_loop37.i.us
  %tid.i.i1.i.us = tail call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #2
  %add_xb_load243_calltmp241.i.us = add i32 %tid.i.i1.i.us, %xb.0.i158.us
  %less_xb_load_x1_load.i.us = icmp slt i32 %add_xb_load243_calltmp241.i.us, %x1
  br i1 %less_xb_load_x1_load.i.us, label %for_loop37.i.us, label %for_exit38.i.us

for_loop37.i.lr.ph.us:                            ; preds = %for_exit38.i.us, %for_test35.i.preheader.lr.ph
  %y.0.i160.us = phi i32 [ %y_load244_plus1.i.us, %for_exit38.i.us ], [ %y0, %for_test35.i.preheader.lr.ph ]
  %mul_y_load46_Nx_load47.i.us = mul i32 %y.0.i160.us, %Nx
  %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us = add i32 %mul_y_load46_Nx_load47.i.us, %mul_z_load45_Nxy_load.i
  %.lhs138.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %Nx
  %.lhs141.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul__Nx_load119.i
  %.lhs142.lhs.us = sub i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %Nx
  %.lhs143.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul__Nx_load167.i
  %.lhs144.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul__Nx_load127.i
  %.lhs145.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul_Nx_load_Ny_load.i
  %.lhs146.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul__Nx_load175.i
  %.lhs147.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul__Nxy_load136.i
  %.lhs148.lhs.us = sub i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul_Nx_load_Ny_load.i
  %.lhs149.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul__Nxy_load184.i
  %.lhs150.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul__Nxy_load144.i
  %.lhs152.lhs.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.i.us, %mul__Nxy_load192.i
  br label %for_loop37.i.us

for_exit31.i:                                     ; preds = %for_exit38.i.us, %for_test35.i.preheader.lr.ph, %for_test28.i.preheader
  %z_load245_plus1.i = add i32 %z.0.i162, 1
  %exitcond163 = icmp eq i32 %z.0.i162, %0
  br i1 %exitcond163, label %stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_.exit, label %for_test28.i.preheader

stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_.exit: ; preds = %for_exit31.i, %allocas
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-features"="+sm_35" }
attributes #2 = { nounwind }

!nvvm.annotations = !{!0}

!0 = metadata !{void (i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*)* @stencil_step_task, metadata !"kernel", i32 1}
!1 = metadata !{ }
!2 = metadata !{ metadata !"output", metadata !0 }
!3 = metadata !{ metadata !"input1", metadata !0 }
!4 = metadata !{ metadata !"input2", metadata !0 }
