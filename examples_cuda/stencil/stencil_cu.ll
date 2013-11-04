; ModuleID = 'stencil_cu.bc'
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind
declare i8* @ISPCAlloc(i8**, i64, i32) #0

; Function Attrs: nounwind
declare void @ISPCLaunch(i8**, i8*, i8*, i32, i32, i32) #0

; Function Attrs: nounwind
declare void @ISPCSync(i8*) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) #1

; Function Attrs: nounwind readonly
declare <4 x double> @llvm.x86.avx.maskload.pd.256(i8*, <4 x double>) #2

; Function Attrs: nounwind
declare void @llvm.x86.avx.maskstore.pd.256(i8*, <4 x double>, <4 x double>) #0

; Function Attrs: nounwind
define internal fastcc void @stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_(i32 %x0, i32 %x1, i32 %y0, i32 %y1, i32 %z0, i32 %z1, i32 %Nx, i32 %Ny, double* noalias nocapture %coef, double* noalias %vsq, double* noalias %Ain, double* noalias %Aout, <8 x i32> %__mask) #3 {
allocas:
  %floatmask.i = bitcast <8 x i32> %__mask to <8 x float>
  %v.i = tail call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i) #1
  %cmp.i = icmp eq i32 %v.i, 255
  %mul_Nx_load_Ny_load = mul i32 %Ny, %Nx
  %coef_load_offset_load = load double* %coef, align 8
  %coef_load18_offset = getelementptr double* %coef, i64 1
  %coef_load18_offset_load = load double* %coef_load18_offset, align 8
  %coef_load21_offset = getelementptr double* %coef, i64 2
  %coef_load21_offset_load = load double* %coef_load21_offset, align 8
  %coef_load24_offset = getelementptr double* %coef, i64 3
  %coef_load24_offset_load = load double* %coef_load24_offset, align 8
  %less_z_load_z1_load260 = icmp slt i32 %z0, %z1
  br i1 %cmp.i, label %for_test.preheader, label %for_test264.preheader

for_test264.preheader:                            ; preds = %allocas
  br i1 %less_z_load_z1_load260, label %for_test275.preheader.lr.ph, label %for_exit

for_test275.preheader.lr.ph:                      ; preds = %for_test264.preheader
  %less_y_load282_y1_load283264 = icmp slt i32 %y0, %y1
  %less_xb_load293_x1_load294262 = icmp slt i32 %x0, %x1
  %x1_load463_broadcast_init = insertelement <8 x i32> undef, i32 %x1, i32 0
  %x1_load463_broadcast = shufflevector <8 x i32> %x1_load463_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  %mul__Nx_load382 = shl i32 %Nx, 1
  %mul__Nx_load431 = mul i32 %Nx, 3
  %mul__Nx_load390 = mul i32 %Nx, -2
  %mul__Nx_load439 = mul i32 %Nx, -3
  %mul__Nxy_load399 = shl i32 %mul_Nx_load_Ny_load, 1
  %mul__Nxy_load448 = mul i32 %mul_Nx_load_Ny_load, 3
  %mul__Nxy_load407 = mul i32 %mul_Nx_load_Ny_load, -2
  %mul__Nxy_load456 = mul i32 %mul_Nx_load_Ny_load, -3
  %Ain_load327_ptr2int_2void = bitcast double* %Ain to i8*
  %mask0.i.i201 = shufflevector <8 x i32> %__mask, <8 x i32> undef, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1.i.i202 = shufflevector <8 x i32> %__mask, <8 x i32> undef, <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %mask0d.i.i203 = bitcast <8 x i32> %mask0.i.i201 to <4 x double>
  %mask1d.i.i204 = bitcast <8 x i32> %mask1.i.i202 to <4 x double>
  %coef1_load315_broadcast_init = insertelement <8 x double> undef, double %coef_load18_offset_load, i32 0
  %coef0_load306_broadcast_init = insertelement <8 x double> undef, double %coef_load_offset_load, i32 0
  %coef2_load364_broadcast_init = insertelement <8 x double> undef, double %coef_load21_offset_load, i32 0
  %coef1_load315_broadcast = shufflevector <8 x double> %coef1_load315_broadcast_init, <8 x double> undef, <8 x i32> zeroinitializer
  %coef0_load306_broadcast = shufflevector <8 x double> %coef0_load306_broadcast_init, <8 x double> undef, <8 x i32> zeroinitializer
  %coef3_load413_broadcast_init = insertelement <8 x double> undef, double %coef_load24_offset_load, i32 0
  %coef2_load364_broadcast = shufflevector <8 x double> %coef2_load364_broadcast_init, <8 x double> undef, <8 x i32> zeroinitializer
  %coef3_load413_broadcast = shufflevector <8 x double> %coef3_load413_broadcast_init, <8 x double> undef, <8 x i32> zeroinitializer
  %Aout_load488_ptr2int_2void = bitcast double* %Aout to i8*
  %vsq_load494_ptr2int_2void = bitcast double* %vsq to i8*
  br label %for_test275.preheader

for_test.preheader:                               ; preds = %allocas
  br i1 %less_z_load_z1_load260, label %for_test30.preheader.lr.ph, label %for_exit

for_test30.preheader.lr.ph:                       ; preds = %for_test.preheader
  %less_y_load_y1_load258 = icmp slt i32 %y0, %y1
  %less_xb_load_x1_load256 = icmp slt i32 %x0, %x1
  %x1_load199_broadcast_init = insertelement <8 x i32> undef, i32 %x1, i32 0
  %x1_load199_broadcast = shufflevector <8 x i32> %x1_load199_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  %mul__Nx_load119 = shl i32 %Nx, 1
  %mul__Nx_load167 = mul i32 %Nx, 3
  %mul__Nx_load127 = mul i32 %Nx, -2
  %mul__Nx_load175 = mul i32 %Nx, -3
  %mul__Nxy_load136 = shl i32 %mul_Nx_load_Ny_load, 1
  %mul__Nxy_load184 = mul i32 %mul_Nx_load_Ny_load, 3
  %mul__Nxy_load144 = mul i32 %mul_Nx_load_Ny_load, -2
  %mul__Nxy_load192 = mul i32 %mul_Nx_load_Ny_load, -3
  %Ain_load65_ptr2int_2void = bitcast double* %Ain to i8*
  %coef1_load_broadcast_init = insertelement <8 x double> undef, double %coef_load18_offset_load, i32 0
  %coef0_load_broadcast_init = insertelement <8 x double> undef, double %coef_load_offset_load, i32 0
  %coef2_load_broadcast_init = insertelement <8 x double> undef, double %coef_load21_offset_load, i32 0
  %coef1_load_broadcast = shufflevector <8 x double> %coef1_load_broadcast_init, <8 x double> undef, <8 x i32> zeroinitializer
  %coef0_load_broadcast = shufflevector <8 x double> %coef0_load_broadcast_init, <8 x double> undef, <8 x i32> zeroinitializer
  %coef3_load_broadcast_init = insertelement <8 x double> undef, double %coef_load24_offset_load, i32 0
  %coef2_load_broadcast = shufflevector <8 x double> %coef2_load_broadcast_init, <8 x double> undef, <8 x i32> zeroinitializer
  %coef3_load_broadcast = shufflevector <8 x double> %coef3_load_broadcast_init, <8 x double> undef, <8 x i32> zeroinitializer
  %Aout_load219_ptr2int_2void = bitcast double* %Aout to i8*
  %vsq_load_ptr2int_2void = bitcast double* %vsq to i8*
  br label %for_test30.preheader

for_test30.preheader:                             ; preds = %for_exit33, %for_test30.preheader.lr.ph
  %z.0261 = phi i32 [ %z0, %for_test30.preheader.lr.ph ], [ %z_load242_plus1, %for_exit33 ]
  br i1 %less_y_load_y1_load258, label %for_test37.preheader.lr.ph, label %for_exit33

for_test37.preheader.lr.ph:                       ; preds = %for_test30.preheader
  %mul_z_load45_Nxy_load = mul i32 %z.0261, %mul_Nx_load_Ny_load
  br i1 %less_xb_load_x1_load256, label %for_loop39.lr.ph.us, label %for_exit33

for_exit40.us:                                    ; preds = %safe_if_after_true.us
  %y_load241_plus1.us = add i32 %y.0259.us, 1
  %exitcond = icmp eq i32 %y_load241_plus1.us, %y1
  br i1 %exitcond, label %for_exit33, label %for_loop39.lr.ph.us

for_loop39.us:                                    ; preds = %for_loop39.lr.ph.us, %safe_if_after_true.us
  %xb.0257.us = phi i32 [ %x0, %for_loop39.lr.ph.us ], [ %add_xb_load240_.us, %safe_if_after_true.us ]
  %xb_load44_broadcast_init.us = insertelement <8 x i32> undef, i32 %xb.0257.us, i32 0
  %xb_load44_broadcast.us = shufflevector <8 x i32> %xb_load44_broadcast_init.us, <8 x i32> undef, <8 x i32> zeroinitializer
  %add_xb_load44_broadcast_.us = add <8 x i32> %xb_load44_broadcast.us, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %less_x_load198_x1_load199_broadcast.us = icmp slt <8 x i32> %add_xb_load44_broadcast_.us, %x1_load199_broadcast
  %"oldMask&test.us" = select <8 x i1> %less_x_load198_x1_load199_broadcast.us, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer
  %floatmask.i244.us = bitcast <8 x i32> %"oldMask&test.us" to <8 x float>
  %v.i245.us = tail call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i244.us) #1
  %cmp.i246.us = icmp eq i32 %v.i245.us, 0
  br i1 %cmp.i246.us, label %safe_if_after_true.us, label %safe_if_run_true.us

safe_if_run_true.us:                              ; preds = %for_loop39.us
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast.elt0.us = add i32 %xb.0257.us, %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us
  %scaled_varying.elt0.us = shl i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast.elt0.us, 3
  %"varying+const_offsets.elt0.us" = add i32 %scaled_varying.elt0.us, -8
  %0 = sext i32 %"varying+const_offsets.elt0.us" to i64
  %ptr.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %0, !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
  %ptr_cast_for_load.us = bitcast i8* %ptr.us to <8 x double>*
  %ptr_masked_load521.us = load <8 x double>* %ptr_cast_for_load.us, align 8, !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
  %"varying+const_offsets529.elt0.us" = add i32 %scaled_varying.elt0.us, 8
  %1 = sext i32 %"varying+const_offsets529.elt0.us" to i64
  %ptr530.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %1, !filename !0, !first_line !1, !first_column !4, !last_line !1, !last_column !5
  %ptr_cast_for_load531.us = bitcast i8* %ptr530.us to <8 x double>*
  %ptr530_masked_load532.us = load <8 x double>* %ptr_cast_for_load531.us, align 8, !filename !0, !first_line !1, !first_column !4, !last_line !1, !last_column !5
  %"varying+const_offsets540.elt0.us" = add i32 %scaled_varying.elt0.us, -16
  %2 = sext i32 %"varying+const_offsets540.elt0.us" to i64
  %ptr541.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %2, !filename !0, !first_line !6, !first_column !2, !last_line !6, !last_column !3
  %ptr_cast_for_load542.us = bitcast i8* %ptr541.us to <8 x double>*
  %ptr541_masked_load543.us = load <8 x double>* %ptr_cast_for_load542.us, align 8, !filename !0, !first_line !6, !first_column !2, !last_line !6, !last_column !3
  %"varying+const_offsets551.elt0.us" = add i32 %scaled_varying.elt0.us, 16
  %3 = sext i32 %"varying+const_offsets551.elt0.us" to i64
  %ptr552.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %3, !filename !0, !first_line !6, !first_column !4, !last_line !6, !last_column !5
  %ptr_cast_for_load553.us = bitcast i8* %ptr552.us to <8 x double>*
  %ptr552_masked_load554.us = load <8 x double>* %ptr_cast_for_load553.us, align 8, !filename !0, !first_line !6, !first_column !4, !last_line !6, !last_column !5
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast556_mul__Nx_load71_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast556.elt0.us, %xb.0257.us
  %scaled_varying560.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast556_mul__Nx_load71_broadcast.elt0.us, 3
  %4 = sext i32 %scaled_varying560.elt0.us to i64
  %ptr562.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %4, !filename !0, !first_line !2, !first_column !7, !last_line !2, !last_column !8
  %ptr_cast_for_load563.us = bitcast i8* %ptr562.us to <8 x double>*
  %ptr562_masked_load564.us = load <8 x double>* %ptr_cast_for_load563.us, align 8, !filename !0, !first_line !2, !first_column !7, !last_line !2, !last_column !8
  %add_Ain_load57_offset_load_Ain_load65_offset_load.us = fadd <8 x double> %ptr_masked_load521.us, %ptr530_masked_load532.us
  %"varying+const_offsets572.elt0.us" = add i32 %scaled_varying.elt0.us, -24
  %5 = sext i32 %"varying+const_offsets572.elt0.us" to i64
  %ptr573.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %5, !filename !0, !first_line !9, !first_column !2, !last_line !9, !last_column !3
  %ptr_cast_for_load574.us = bitcast i8* %ptr573.us to <8 x double>*
  %ptr573_masked_load575.us = load <8 x double>* %ptr_cast_for_load574.us, align 8, !filename !0, !first_line !9, !first_column !2, !last_line !9, !last_column !3
  %"varying+const_offsets583.elt0.us" = add i32 %scaled_varying.elt0.us, 24
  %6 = sext i32 %"varying+const_offsets583.elt0.us" to i64
  %ptr584.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %6, !filename !0, !first_line !9, !first_column !4, !last_line !9, !last_column !5
  %ptr_cast_for_load585.us = bitcast i8* %ptr584.us to <8 x double>*
  %ptr584_masked_load586.us = load <8 x double>* %ptr_cast_for_load585.us, align 8, !filename !0, !first_line !9, !first_column !4, !last_line !9, !last_column !5
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast588_mul__Nx_load119_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast588.elt0.us, %xb.0257.us
  %scaled_varying593.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast588_mul__Nx_load119_broadcast.elt0.us, 3
  %7 = sext i32 %scaled_varying593.elt0.us to i64
  %ptr595.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %7, !filename !0, !first_line !10, !first_column !11, !last_line !10, !last_column !1
  %ptr_cast_for_load596.us = bitcast i8* %ptr595.us to <8 x double>*
  %ptr595_masked_load597.us = load <8 x double>* %ptr_cast_for_load596.us, align 8, !filename !0, !first_line !10, !first_column !11, !last_line !10, !last_column !1
  %add_Ain_load105_offset_load_Ain_load113_offset_load.us = fadd <8 x double> %ptr541_masked_load543.us, %ptr552_masked_load554.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast599_mul__Nx_load79_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast599.elt0.us, %xb.0257.us
  %scaled_varying604.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast599_mul__Nx_load79_broadcast.elt0.us, 3
  %8 = sext i32 %scaled_varying604.elt0.us to i64
  %ptr606.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %8, !filename !0, !first_line !2, !first_column !12, !last_line !2, !last_column !13
  %ptr_cast_for_load607.us = bitcast i8* %ptr606.us to <8 x double>*
  %ptr606_masked_load608.us = load <8 x double>* %ptr_cast_for_load607.us, align 8, !filename !0, !first_line !2, !first_column !12, !last_line !2, !last_column !13
  %add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load.us = fadd <8 x double> %add_Ain_load57_offset_load_Ain_load65_offset_load.us, %ptr562_masked_load564.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast610_mul__Nx_load167_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast610.elt0.us, %xb.0257.us
  %scaled_varying615.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast610_mul__Nx_load167_broadcast.elt0.us, 3
  %9 = sext i32 %scaled_varying615.elt0.us to i64
  %ptr617.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %9, !filename !0, !first_line !14, !first_column !11, !last_line !14, !last_column !1
  %ptr_cast_for_load618.us = bitcast i8* %ptr617.us to <8 x double>*
  %ptr617_masked_load619.us = load <8 x double>* %ptr_cast_for_load618.us, align 8, !filename !0, !first_line !14, !first_column !11, !last_line !14, !last_column !1
  %add_Ain_load153_offset_load_Ain_load161_offset_load.us = fadd <8 x double> %ptr573_masked_load575.us, %ptr584_masked_load586.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast621_mul__Nx_load127_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast621.elt0.us, %xb.0257.us
  %scaled_varying626.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast621_mul__Nx_load127_broadcast.elt0.us, 3
  %10 = sext i32 %scaled_varying626.elt0.us to i64
  %ptr628.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %10, !filename !0, !first_line !10, !first_column !6, !last_line !10, !last_column !15
  %ptr_cast_for_load629.us = bitcast i8* %ptr628.us to <8 x double>*
  %ptr628_masked_load630.us = load <8 x double>* %ptr_cast_for_load629.us, align 8, !filename !0, !first_line !10, !first_column !6, !last_line !10, !last_column !15
  %add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load.us = fadd <8 x double> %add_Ain_load105_offset_load_Ain_load113_offset_load.us, %ptr595_masked_load597.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast632_mul__Nxy_load88_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast632.elt0.us, %xb.0257.us
  %scaled_varying637.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast632_mul__Nxy_load88_broadcast.elt0.us, 3
  %11 = sext i32 %scaled_varying637.elt0.us to i64
  %ptr639.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %11, !filename !0, !first_line !12, !first_column !11, !last_line !12, !last_column !1
  %ptr_cast_for_load640.us = bitcast i8* %ptr639.us to <8 x double>*
  %ptr639_masked_load641.us = load <8 x double>* %ptr_cast_for_load640.us, align 8, !filename !0, !first_line !12, !first_column !11, !last_line !12, !last_column !1
  %add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load.us = fadd <8 x double> %add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load.us, %ptr606_masked_load608.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast643_mul__Nx_load175_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast643.elt0.us, %xb.0257.us
  %scaled_varying648.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast643_mul__Nx_load175_broadcast.elt0.us, 3
  %12 = sext i32 %scaled_varying648.elt0.us to i64
  %ptr650.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %12, !filename !0, !first_line !14, !first_column !6, !last_line !14, !last_column !15
  %ptr_cast_for_load651.us = bitcast i8* %ptr650.us to <8 x double>*
  %ptr650_masked_load652.us = load <8 x double>* %ptr_cast_for_load651.us, align 8, !filename !0, !first_line !14, !first_column !6, !last_line !14, !last_column !15
  %add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load.us = fadd <8 x double> %add_Ain_load153_offset_load_Ain_load161_offset_load.us, %ptr617_masked_load619.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast654_mul__Nxy_load136_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast654.elt0.us, %xb.0257.us
  %scaled_varying659.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast654_mul__Nxy_load136_broadcast.elt0.us, 3
  %13 = sext i32 %scaled_varying659.elt0.us to i64
  %ptr661.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %13, !filename !0, !first_line !16, !first_column !11, !last_line !16, !last_column !1
  %ptr_cast_for_load662.us = bitcast i8* %ptr661.us to <8 x double>*
  %ptr661_masked_load663.us = load <8 x double>* %ptr_cast_for_load662.us, align 8, !filename !0, !first_line !16, !first_column !11, !last_line !16, !last_column !1
  %add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load.us = fadd <8 x double> %add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load.us, %ptr628_masked_load630.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast665_mul__Nxy_load96_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast665.elt0.us, %xb.0257.us
  %scaled_varying670.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast665_mul__Nxy_load96_broadcast.elt0.us, 3
  %14 = sext i32 %scaled_varying670.elt0.us to i64
  %ptr672.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %14, !filename !0, !first_line !12, !first_column !6, !last_line !12, !last_column !15
  %ptr_cast_for_load673.us = bitcast i8* %ptr672.us to <8 x double>*
  %ptr672_masked_load674.us = load <8 x double>* %ptr_cast_for_load673.us, align 8, !filename !0, !first_line !12, !first_column !6, !last_line !12, !last_column !15
  %add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load.us = fadd <8 x double> %add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load.us, %ptr639_masked_load641.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast676_mul__Nxy_load184_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast676.elt0.us, %xb.0257.us
  %scaled_varying681.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast676_mul__Nxy_load184_broadcast.elt0.us, 3
  %15 = sext i32 %scaled_varying681.elt0.us to i64
  %ptr683.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %15, !filename !0, !first_line !17, !first_column !11, !last_line !17, !last_column !1
  %ptr_cast_for_load684.us = bitcast i8* %ptr683.us to <8 x double>*
  %ptr683_masked_load685.us = load <8 x double>* %ptr_cast_for_load684.us, align 8, !filename !0, !first_line !17, !first_column !11, !last_line !17, !last_column !1
  %add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load.us = fadd <8 x double> %add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load.us, %ptr650_masked_load652.us
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast687_mul__Nxy_load144_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast687.elt0.us, %xb.0257.us
  %scaled_varying692.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast687_mul__Nxy_load144_broadcast.elt0.us, 3
  %16 = sext i32 %scaled_varying692.elt0.us to i64
  %ptr694.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %16, !filename !0, !first_line !16, !first_column !6, !last_line !16, !last_column !15
  %ptr_cast_for_load695.us = bitcast i8* %ptr694.us to <8 x double>*
  %ptr694_masked_load696.us = load <8 x double>* %ptr_cast_for_load695.us, align 8, !filename !0, !first_line !16, !first_column !6, !last_line !16, !last_column !15
  %add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load.us = fadd <8 x double> %add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load.us, %ptr661_masked_load663.us
  %add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load.us = fadd <8 x double> %add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load.us, %ptr672_masked_load674.us
  %17 = sext i32 %scaled_varying.elt0.us to i64
  %ptr705.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %17, !filename !0, !first_line !8, !first_column !18, !last_line !8, !last_column !19
  %ptr_cast_for_load706.us = bitcast i8* %ptr705.us to <8 x double>*
  %ptr705_masked_load707.us = load <8 x double>* %ptr_cast_for_load706.us, align 8, !filename !0, !first_line !8, !first_column !18, !last_line !8, !last_column !19
  %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast709_mul__Nxy_load192_broadcast.elt0.us = add i32 %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast709.elt0.us, %xb.0257.us
  %scaled_varying714.elt0.us = shl i32 %add_add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast709_mul__Nxy_load192_broadcast.elt0.us, 3
  %18 = sext i32 %scaled_varying714.elt0.us to i64
  %ptr716.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %18, !filename !0, !first_line !17, !first_column !6, !last_line !17, !last_column !15
  %ptr_cast_for_load717.us = bitcast i8* %ptr716.us to <8 x double>*
  %ptr716_masked_load718.us = load <8 x double>* %ptr_cast_for_load717.us, align 8, !filename !0, !first_line !17, !first_column !6, !last_line !17, !last_column !15
  %add_add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load_Ain_load185_offset_load.us = fadd <8 x double> %add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load.us, %ptr683_masked_load685.us
  %add_add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load_Ain_load145_offset_load.us = fadd <8 x double> %add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load.us, %ptr694_masked_load696.us
  %mul_coef1_load_broadcast_add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load.us = fmul <8 x double> %coef1_load_broadcast, %add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load.us
  %mul_coef0_load_broadcast_Ain_load_offset_load.us = fmul <8 x double> %coef0_load_broadcast, %ptr705_masked_load707.us
  %add_add_add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load_Ain_load185_offset_load_Ain_load193_offset_load.us = fadd <8 x double> %add_add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load_Ain_load185_offset_load.us, %ptr716_masked_load718.us
  %mul_coef2_load_broadcast_add_add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load_Ain_load145_offset_load.us = fmul <8 x double> %coef2_load_broadcast, %add_add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load_Ain_load145_offset_load.us
  %add_mul_coef0_load_broadcast_Ain_load_offset_load_mul_coef1_load_broadcast_add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load.us = fadd <8 x double> %mul_coef1_load_broadcast_add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load.us, %mul_coef0_load_broadcast_Ain_load_offset_load.us
  %mul_coef3_load_broadcast_add_add_add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load_Ain_load185_offset_load_Ain_load193_offset_load.us = fmul <8 x double> %coef3_load_broadcast, %add_add_add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load_Ain_load185_offset_load_Ain_load193_offset_load.us
  %add_add_mul_coef0_load_broadcast_Ain_load_offset_load_mul_coef1_load_broadcast_add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load_mul_coef2_load_broadcast_add_add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load_Ain_load145_offset_load.us = fadd <8 x double> %mul_coef2_load_broadcast_add_add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load_Ain_load145_offset_load.us, %add_mul_coef0_load_broadcast_Ain_load_offset_load_mul_coef1_load_broadcast_add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load.us
  %add_add_add_mul_coef0_load_broadcast_Ain_load_offset_load_mul_coef1_load_broadcast_add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load_mul_coef2_load_broadcast_add_add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load_Ain_load145_offset_load_mul_coef3_load_broadcast_add_add_add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load_Ain_load185_offset_load_Ain_load193_offset_load.us = fadd <8 x double> %add_add_mul_coef0_load_broadcast_Ain_load_offset_load_mul_coef1_load_broadcast_add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load_mul_coef2_load_broadcast_add_add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load_Ain_load145_offset_load.us, %mul_coef3_load_broadcast_add_add_add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load_Ain_load185_offset_load_Ain_load193_offset_load.us
  %mask0.i.i234.us = shufflevector <8 x i32> %"oldMask&test.us", <8 x i32> undef, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1.i.i235.us = shufflevector <8 x i32> %"oldMask&test.us", <8 x i32> undef, <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %mask0d.i.i236.us = bitcast <8 x i32> %mask0.i.i234.us to <4 x double>
  %mask1d.i.i237.us = bitcast <8 x i32> %mask1.i.i235.us to <4 x double>
  %val0d.i.i238.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr705.us, <4 x double> %mask0d.i.i236.us) #0
  %ptr727.sum.us = add i64 %17, 32
  %ptr1.i.i239.us = getelementptr i8* %Ain_load65_ptr2int_2void, i64 %ptr727.sum.us
  %val1d.i.i240.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i239.us, <4 x double> %mask1d.i.i237.us) #0
  %vald.i.i241.us = shufflevector <4 x double> %val0d.i.i238.us, <4 x double> %val1d.i.i240.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mul__Ain_load211_offset_load.us = fmul <8 x double> %vald.i.i241.us, <double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00>
  %ptr736.us = getelementptr i8* %Aout_load219_ptr2int_2void, i64 %17, !filename !0, !first_line !20, !first_column !21, !last_line !20, !last_column !22
  %val0d.i.i228.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr736.us, <4 x double> %mask0d.i.i236.us) #0
  %ptr1.i.i229.us = getelementptr i8* %Aout_load219_ptr2int_2void, i64 %ptr727.sum.us
  %val1d.i.i230.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i229.us, <4 x double> %mask1d.i.i237.us) #0
  %vald.i.i231.us = shufflevector <4 x double> %val0d.i.i228.us, <4 x double> %val1d.i.i230.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %sub_mul__Ain_load211_offset_load_Aout_load219_offset_load.us = fsub <8 x double> %mul__Ain_load211_offset_load.us, %vald.i.i231.us
  %ptr745.us = getelementptr i8* %vsq_load_ptr2int_2void, i64 %17, !filename !0, !first_line !23, !first_column !24, !last_line !23, !last_column !7
  %val0d.i.i218.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr745.us, <4 x double> %mask0d.i.i236.us) #0
  %ptr1.i.i219.us = getelementptr i8* %vsq_load_ptr2int_2void, i64 %ptr727.sum.us
  %val1d.i.i220.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i219.us, <4 x double> %mask1d.i.i237.us) #0
  %vald.i.i221.us = shufflevector <4 x double> %val0d.i.i218.us, <4 x double> %val1d.i.i220.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mul_vsq_load_offset_load_div_load.us = fmul <8 x double> %add_add_add_mul_coef0_load_broadcast_Ain_load_offset_load_mul_coef1_load_broadcast_add_add_add_add_add_Ain_load57_offset_load_Ain_load65_offset_load_Ain_load73_offset_load_Ain_load81_offset_load_Ain_load89_offset_load_Ain_load97_offset_load_mul_coef2_load_broadcast_add_add_add_add_add_Ain_load105_offset_load_Ain_load113_offset_load_Ain_load121_offset_load_Ain_load129_offset_load_Ain_load137_offset_load_Ain_load145_offset_load_mul_coef3_load_broadcast_add_add_add_add_add_Ain_load153_offset_load_Ain_load161_offset_load_Ain_load169_offset_load_Ain_load177_offset_load_Ain_load185_offset_load_Ain_load193_offset_load.us, %vald.i.i221.us
  %add_sub_mul__Ain_load211_offset_load_Aout_load219_offset_load_mul_vsq_load_offset_load_div_load.us = fadd <8 x double> %sub_mul__Ain_load211_offset_load_Aout_load219_offset_load.us, %mul_vsq_load_offset_load_div_load.us
  %val0.i.i.us = shufflevector <8 x double> %add_sub_mul__Ain_load211_offset_load_Aout_load219_offset_load_mul_vsq_load_offset_load_div_load.us, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %val1.i.i.us = shufflevector <8 x double> %add_sub_mul__Ain_load211_offset_load_Aout_load219_offset_load_mul_vsq_load_offset_load_div_load.us, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  call void @llvm.x86.avx.maskstore.pd.256(i8* %ptr736.us, <4 x double> %mask0d.i.i236.us, <4 x double> %val0.i.i.us) #0
  call void @llvm.x86.avx.maskstore.pd.256(i8* %ptr1.i.i229.us, <4 x double> %mask1d.i.i237.us, <4 x double> %val1.i.i.us) #0
  br label %safe_if_after_true.us

safe_if_after_true.us:                            ; preds = %safe_if_run_true.us, %for_loop39.us
  %add_xb_load240_.us = add i32 %xb.0257.us, 8
  %less_xb_load_x1_load.us = icmp slt i32 %add_xb_load240_.us, %x1
  br i1 %less_xb_load_x1_load.us, label %for_loop39.us, label %for_exit40.us

for_loop39.lr.ph.us:                              ; preds = %for_exit40.us, %for_test37.preheader.lr.ph
  %y.0259.us = phi i32 [ %y_load241_plus1.us, %for_exit40.us ], [ %y0, %for_test37.preheader.lr.ph ]
  %mul_y_load46_Nx_load47.us = mul i32 %y.0259.us, %Nx
  %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us = add i32 %mul_y_load46_Nx_load47.us, %mul_z_load45_Nxy_load
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast556.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %Nx
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast588.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul__Nx_load119
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast599.elt0.us = sub i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %Nx
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast610.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul__Nx_load167
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast621.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul__Nx_load127
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast632.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul_Nx_load_Ny_load
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast643.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul__Nx_load175
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast654.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul__Nxy_load136
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast665.elt0.us = sub i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul_Nx_load_Ny_load
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast676.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul__Nxy_load184
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast687.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul__Nxy_load144
  %add_add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47_broadcast_xb_load44_broadcast709.elt0.us = add i32 %add_mul_z_load45_Nxy_load_mul_y_load46_Nx_load47.us, %mul__Nxy_load192
  br label %for_loop39.us

for_exit:                                         ; preds = %for_exit278, %for_exit33, %for_test.preheader, %for_test264.preheader
  ret void

for_exit33:                                       ; preds = %for_exit40.us, %for_test37.preheader.lr.ph, %for_test30.preheader
  %z_load242_plus1 = add i32 %z.0261, 1
  %exitcond269 = icmp eq i32 %z_load242_plus1, %z1
  br i1 %exitcond269, label %for_exit, label %for_test30.preheader

for_test275.preheader:                            ; preds = %for_exit278, %for_test275.preheader.lr.ph
  %z269.0268 = phi i32 [ %z0, %for_test275.preheader.lr.ph ], [ %z_load518_plus1, %for_exit278 ]
  br i1 %less_y_load282_y1_load283264, label %for_test286.preheader.lr.ph, label %for_exit278

for_test286.preheader.lr.ph:                      ; preds = %for_test275.preheader
  %mul_z_load300_Nxy_load301 = mul i32 %z269.0268, %mul_Nx_load_Ny_load
  br i1 %less_xb_load293_x1_load294262, label %for_loop288.lr.ph.us, label %for_exit278

for_exit289.us:                                   ; preds = %safe_if_after_true466.us
  %y_load517_plus1.us = add i32 %y280.0265.us, 1
  %exitcond271 = icmp eq i32 %y_load517_plus1.us, %y1
  br i1 %exitcond271, label %for_exit278, label %for_loop288.lr.ph.us

for_loop288.us:                                   ; preds = %for_loop288.lr.ph.us, %safe_if_after_true466.us
  %xb291.0263.us = phi i32 [ %x0, %for_loop288.lr.ph.us ], [ %add_xb291_load_.us, %safe_if_after_true466.us ]
  %xb_load298_broadcast_init.us = insertelement <8 x i32> undef, i32 %xb291.0263.us, i32 0
  %xb_load298_broadcast.us = shufflevector <8 x i32> %xb_load298_broadcast_init.us, <8 x i32> undef, <8 x i32> zeroinitializer
  %add_xb_load298_broadcast_.us = add <8 x i32> %xb_load298_broadcast.us, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %less_x_load462_x1_load463_broadcast.us = icmp slt <8 x i32> %add_xb_load298_broadcast_.us, %x1_load463_broadcast
  %"oldMask&test468.us" = select <8 x i1> %less_x_load462_x1_load463_broadcast.us, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer
  %"internal_mask&function_mask472.us" = and <8 x i32> %"oldMask&test468.us", %__mask
  %floatmask.i211.us = bitcast <8 x i32> %"internal_mask&function_mask472.us" to <8 x float>
  %v.i212.us = tail call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i211.us) #1
  %cmp.i213.us = icmp eq i32 %v.i212.us, 0
  br i1 %cmp.i213.us, label %safe_if_after_true466.us, label %safe_if_run_true467.us

safe_if_run_true467.us:                           ; preds = %for_loop288.us
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast.elt0.us = add i32 %xb291.0263.us, %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us
  %scaled_varying757.elt0.us = shl i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast.elt0.us, 3
  %"varying+const_offsets.elt0758.us" = add i32 %scaled_varying757.elt0.us, -8
  %19 = sext i32 %"varying+const_offsets.elt0758.us" to i64
  %ptr759.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %19, !filename !0, !first_line !1, !first_column !2, !last_line !1, !last_column !3
  %val0d.i.i205.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr759.us, <4 x double> %mask0d.i.i203) #0
  %ptr759.sum.us = add i64 %19, 32
  %ptr1.i.i206.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr759.sum.us
  %val1d.i.i207.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i206.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i208.us = shufflevector <4 x double> %val0d.i.i205.us, <4 x double> %val1d.i.i207.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %"varying+const_offsets767.elt0.us" = add i32 %scaled_varying757.elt0.us, 8
  %20 = sext i32 %"varying+const_offsets767.elt0.us" to i64
  %ptr768.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %20, !filename !0, !first_line !1, !first_column !4, !last_line !1, !last_column !5
  %val0d.i.i195.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr768.us, <4 x double> %mask0d.i.i203) #0
  %ptr768.sum.us = add i64 %20, 32
  %ptr1.i.i196.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr768.sum.us
  %val1d.i.i197.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i196.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i198.us = shufflevector <4 x double> %val0d.i.i195.us, <4 x double> %val1d.i.i197.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %"varying+const_offsets776.elt0.us" = add i32 %scaled_varying757.elt0.us, -16
  %21 = sext i32 %"varying+const_offsets776.elt0.us" to i64
  %ptr777.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %21, !filename !0, !first_line !6, !first_column !2, !last_line !6, !last_column !3
  %val0d.i.i185.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr777.us, <4 x double> %mask0d.i.i203) #0
  %ptr777.sum.us = add i64 %21, 32
  %ptr1.i.i186.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr777.sum.us
  %val1d.i.i187.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i186.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i188.us = shufflevector <4 x double> %val0d.i.i185.us, <4 x double> %val1d.i.i187.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %"varying+const_offsets785.elt0.us" = add i32 %scaled_varying757.elt0.us, 16
  %22 = sext i32 %"varying+const_offsets785.elt0.us" to i64
  %ptr786.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %22, !filename !0, !first_line !6, !first_column !4, !last_line !6, !last_column !5
  %val0d.i.i175.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr786.us, <4 x double> %mask0d.i.i203) #0
  %ptr786.sum.us = add i64 %22, 32
  %ptr1.i.i176.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr786.sum.us
  %val1d.i.i177.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i176.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i178.us = shufflevector <4 x double> %val0d.i.i175.us, <4 x double> %val1d.i.i177.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast788_mul__Nx_load333_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast788.elt0.us, %xb291.0263.us
  %scaled_varying793.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast788_mul__Nx_load333_broadcast.elt0.us, 3
  %23 = sext i32 %scaled_varying793.elt0.us to i64
  %ptr795.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %23, !filename !0, !first_line !2, !first_column !7, !last_line !2, !last_column !8
  %val0d.i.i165.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr795.us, <4 x double> %mask0d.i.i203) #0
  %ptr795.sum.us = add i64 %23, 32
  %ptr1.i.i166.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr795.sum.us
  %val1d.i.i167.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i166.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i168.us = shufflevector <4 x double> %val0d.i.i165.us, <4 x double> %val1d.i.i167.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_Ain_load319_offset_load_Ain_load327_offset_load.us = fadd <8 x double> %vald.i.i208.us, %vald.i.i198.us
  %"varying+const_offsets803.elt0.us" = add i32 %scaled_varying757.elt0.us, -24
  %24 = sext i32 %"varying+const_offsets803.elt0.us" to i64
  %ptr804.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %24, !filename !0, !first_line !9, !first_column !2, !last_line !9, !last_column !3
  %val0d.i.i155.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr804.us, <4 x double> %mask0d.i.i203) #0
  %ptr804.sum.us = add i64 %24, 32
  %ptr1.i.i156.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr804.sum.us
  %val1d.i.i157.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i156.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i158.us = shufflevector <4 x double> %val0d.i.i155.us, <4 x double> %val1d.i.i157.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %"varying+const_offsets812.elt0.us" = add i32 %scaled_varying757.elt0.us, 24
  %25 = sext i32 %"varying+const_offsets812.elt0.us" to i64
  %ptr813.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %25, !filename !0, !first_line !9, !first_column !4, !last_line !9, !last_column !5
  %val0d.i.i145.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr813.us, <4 x double> %mask0d.i.i203) #0
  %ptr813.sum.us = add i64 %25, 32
  %ptr1.i.i146.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr813.sum.us
  %val1d.i.i147.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i146.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i148.us = shufflevector <4 x double> %val0d.i.i145.us, <4 x double> %val1d.i.i147.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast815_mul__Nx_load382_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast815.elt0.us, %xb291.0263.us
  %scaled_varying820.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast815_mul__Nx_load382_broadcast.elt0.us, 3
  %26 = sext i32 %scaled_varying820.elt0.us to i64
  %ptr822.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %26, !filename !0, !first_line !10, !first_column !11, !last_line !10, !last_column !1
  %val0d.i.i135.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr822.us, <4 x double> %mask0d.i.i203) #0
  %ptr822.sum.us = add i64 %26, 32
  %ptr1.i.i136.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr822.sum.us
  %val1d.i.i137.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i136.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i138.us = shufflevector <4 x double> %val0d.i.i135.us, <4 x double> %val1d.i.i137.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_Ain_load368_offset_load_Ain_load376_offset_load.us = fadd <8 x double> %vald.i.i188.us, %vald.i.i178.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast824_mul__Nx_load341_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast824.elt0.us, %xb291.0263.us
  %scaled_varying829.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast824_mul__Nx_load341_broadcast.elt0.us, 3
  %27 = sext i32 %scaled_varying829.elt0.us to i64
  %ptr831.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %27, !filename !0, !first_line !2, !first_column !12, !last_line !2, !last_column !13
  %val0d.i.i125.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr831.us, <4 x double> %mask0d.i.i203) #0
  %ptr831.sum.us = add i64 %27, 32
  %ptr1.i.i126.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr831.sum.us
  %val1d.i.i127.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i126.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i128.us = shufflevector <4 x double> %val0d.i.i125.us, <4 x double> %val1d.i.i127.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load.us = fadd <8 x double> %add_Ain_load319_offset_load_Ain_load327_offset_load.us, %vald.i.i168.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast833_mul__Nx_load431_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast833.elt0.us, %xb291.0263.us
  %scaled_varying838.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast833_mul__Nx_load431_broadcast.elt0.us, 3
  %28 = sext i32 %scaled_varying838.elt0.us to i64
  %ptr840.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %28, !filename !0, !first_line !14, !first_column !11, !last_line !14, !last_column !1
  %val0d.i.i115.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr840.us, <4 x double> %mask0d.i.i203) #0
  %ptr840.sum.us = add i64 %28, 32
  %ptr1.i.i116.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr840.sum.us
  %val1d.i.i117.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i116.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i118.us = shufflevector <4 x double> %val0d.i.i115.us, <4 x double> %val1d.i.i117.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_Ain_load417_offset_load_Ain_load425_offset_load.us = fadd <8 x double> %vald.i.i158.us, %vald.i.i148.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast842_mul__Nx_load390_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast842.elt0.us, %xb291.0263.us
  %scaled_varying847.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast842_mul__Nx_load390_broadcast.elt0.us, 3
  %29 = sext i32 %scaled_varying847.elt0.us to i64
  %ptr849.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %29, !filename !0, !first_line !10, !first_column !6, !last_line !10, !last_column !15
  %val0d.i.i105.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr849.us, <4 x double> %mask0d.i.i203) #0
  %ptr849.sum.us = add i64 %29, 32
  %ptr1.i.i106.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr849.sum.us
  %val1d.i.i107.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i106.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i108.us = shufflevector <4 x double> %val0d.i.i105.us, <4 x double> %val1d.i.i107.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load.us = fadd <8 x double> %add_Ain_load368_offset_load_Ain_load376_offset_load.us, %vald.i.i138.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast851_mul__Nxy_load350_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast851.elt0.us, %xb291.0263.us
  %scaled_varying856.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast851_mul__Nxy_load350_broadcast.elt0.us, 3
  %30 = sext i32 %scaled_varying856.elt0.us to i64
  %ptr858.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %30, !filename !0, !first_line !12, !first_column !11, !last_line !12, !last_column !1
  %val0d.i.i95.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr858.us, <4 x double> %mask0d.i.i203) #0
  %ptr858.sum.us = add i64 %30, 32
  %ptr1.i.i96.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr858.sum.us
  %val1d.i.i97.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i96.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i98.us = shufflevector <4 x double> %val0d.i.i95.us, <4 x double> %val1d.i.i97.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load.us = fadd <8 x double> %add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load.us, %vald.i.i128.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast860_mul__Nx_load439_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast860.elt0.us, %xb291.0263.us
  %scaled_varying865.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast860_mul__Nx_load439_broadcast.elt0.us, 3
  %31 = sext i32 %scaled_varying865.elt0.us to i64
  %ptr867.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %31, !filename !0, !first_line !14, !first_column !6, !last_line !14, !last_column !15
  %val0d.i.i85.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr867.us, <4 x double> %mask0d.i.i203) #0
  %ptr867.sum.us = add i64 %31, 32
  %ptr1.i.i86.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr867.sum.us
  %val1d.i.i87.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i86.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i88.us = shufflevector <4 x double> %val0d.i.i85.us, <4 x double> %val1d.i.i87.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load.us = fadd <8 x double> %add_Ain_load417_offset_load_Ain_load425_offset_load.us, %vald.i.i118.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast869_mul__Nxy_load399_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast869.elt0.us, %xb291.0263.us
  %scaled_varying874.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast869_mul__Nxy_load399_broadcast.elt0.us, 3
  %32 = sext i32 %scaled_varying874.elt0.us to i64
  %ptr876.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %32, !filename !0, !first_line !16, !first_column !11, !last_line !16, !last_column !1
  %val0d.i.i75.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr876.us, <4 x double> %mask0d.i.i203) #0
  %ptr876.sum.us = add i64 %32, 32
  %ptr1.i.i76.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr876.sum.us
  %val1d.i.i77.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i76.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i78.us = shufflevector <4 x double> %val0d.i.i75.us, <4 x double> %val1d.i.i77.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load.us = fadd <8 x double> %add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load.us, %vald.i.i108.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast878_mul__Nxy_load358_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast878.elt0.us, %xb291.0263.us
  %scaled_varying883.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast878_mul__Nxy_load358_broadcast.elt0.us, 3
  %33 = sext i32 %scaled_varying883.elt0.us to i64
  %ptr885.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %33, !filename !0, !first_line !12, !first_column !6, !last_line !12, !last_column !15
  %val0d.i.i65.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr885.us, <4 x double> %mask0d.i.i203) #0
  %ptr885.sum.us = add i64 %33, 32
  %ptr1.i.i66.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr885.sum.us
  %val1d.i.i67.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i66.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i68.us = shufflevector <4 x double> %val0d.i.i65.us, <4 x double> %val1d.i.i67.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load.us = fadd <8 x double> %add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load.us, %vald.i.i98.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast887_mul__Nxy_load448_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast887.elt0.us, %xb291.0263.us
  %scaled_varying892.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast887_mul__Nxy_load448_broadcast.elt0.us, 3
  %34 = sext i32 %scaled_varying892.elt0.us to i64
  %ptr894.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %34, !filename !0, !first_line !17, !first_column !11, !last_line !17, !last_column !1
  %val0d.i.i55.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr894.us, <4 x double> %mask0d.i.i203) #0
  %ptr894.sum.us = add i64 %34, 32
  %ptr1.i.i56.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr894.sum.us
  %val1d.i.i57.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i56.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i58.us = shufflevector <4 x double> %val0d.i.i55.us, <4 x double> %val1d.i.i57.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load.us = fadd <8 x double> %add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load.us, %vald.i.i88.us
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast896_mul__Nxy_load407_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast896.elt0.us, %xb291.0263.us
  %scaled_varying901.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast896_mul__Nxy_load407_broadcast.elt0.us, 3
  %35 = sext i32 %scaled_varying901.elt0.us to i64
  %ptr903.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %35, !filename !0, !first_line !16, !first_column !6, !last_line !16, !last_column !15
  %val0d.i.i45.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr903.us, <4 x double> %mask0d.i.i203) #0
  %ptr903.sum.us = add i64 %35, 32
  %ptr1.i.i46.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr903.sum.us
  %val1d.i.i47.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i46.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i48.us = shufflevector <4 x double> %val0d.i.i45.us, <4 x double> %val1d.i.i47.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load.us = fadd <8 x double> %add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load.us, %vald.i.i78.us
  %add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load.us = fadd <8 x double> %add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load.us, %vald.i.i68.us
  %36 = sext i32 %scaled_varying757.elt0.us to i64
  %ptr912.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %36, !filename !0, !first_line !8, !first_column !18, !last_line !8, !last_column !19
  %val0d.i.i35.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr912.us, <4 x double> %mask0d.i.i203) #0
  %ptr912.sum.us = add i64 %36, 32
  %ptr1.i.i36.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr912.sum.us
  %val1d.i.i37.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i36.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i38.us = shufflevector <4 x double> %val0d.i.i35.us, <4 x double> %val1d.i.i37.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast914_mul__Nxy_load456_broadcast.elt0.us = add i32 %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast914.elt0.us, %xb291.0263.us
  %scaled_varying919.elt0.us = shl i32 %add_add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast914_mul__Nxy_load456_broadcast.elt0.us, 3
  %37 = sext i32 %scaled_varying919.elt0.us to i64
  %ptr921.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %37, !filename !0, !first_line !17, !first_column !6, !last_line !17, !last_column !15
  %val0d.i.i25.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr921.us, <4 x double> %mask0d.i.i203) #0
  %ptr921.sum.us = add i64 %37, 32
  %ptr1.i.i26.us = getelementptr i8* %Ain_load327_ptr2int_2void, i64 %ptr921.sum.us
  %val1d.i.i27.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i26.us, <4 x double> %mask1d.i.i204) #0
  %vald.i.i28.us = shufflevector <4 x double> %val0d.i.i25.us, <4 x double> %val1d.i.i27.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load_Ain_load449_offset_load.us = fadd <8 x double> %add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load.us, %vald.i.i58.us
  %add_add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load_Ain_load408_offset_load.us = fadd <8 x double> %add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load.us, %vald.i.i48.us
  %mul_coef1_load315_broadcast_add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load.us = fmul <8 x double> %coef1_load315_broadcast, %add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load.us
  %mul_coef0_load306_broadcast_Ain_load310_offset_load.us = fmul <8 x double> %coef0_load306_broadcast, %vald.i.i38.us
  %add_add_add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load_Ain_load449_offset_load_Ain_load457_offset_load.us = fadd <8 x double> %add_add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load_Ain_load449_offset_load.us, %vald.i.i28.us
  %mul_coef2_load364_broadcast_add_add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load_Ain_load408_offset_load.us = fmul <8 x double> %coef2_load364_broadcast, %add_add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load_Ain_load408_offset_load.us
  %add_mul_coef0_load306_broadcast_Ain_load310_offset_load_mul_coef1_load315_broadcast_add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load.us = fadd <8 x double> %mul_coef1_load315_broadcast_add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load.us, %mul_coef0_load306_broadcast_Ain_load310_offset_load.us
  %mul_coef3_load413_broadcast_add_add_add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load_Ain_load449_offset_load_Ain_load457_offset_load.us = fmul <8 x double> %coef3_load413_broadcast, %add_add_add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load_Ain_load449_offset_load_Ain_load457_offset_load.us
  %add_add_mul_coef0_load306_broadcast_Ain_load310_offset_load_mul_coef1_load315_broadcast_add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load_mul_coef2_load364_broadcast_add_add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load_Ain_load408_offset_load.us = fadd <8 x double> %mul_coef2_load364_broadcast_add_add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load_Ain_load408_offset_load.us, %add_mul_coef0_load306_broadcast_Ain_load310_offset_load_mul_coef1_load315_broadcast_add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load.us
  %add_add_add_mul_coef0_load306_broadcast_Ain_load310_offset_load_mul_coef1_load315_broadcast_add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load_mul_coef2_load364_broadcast_add_add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load_Ain_load408_offset_load_mul_coef3_load413_broadcast_add_add_add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load_Ain_load449_offset_load_Ain_load457_offset_load.us = fadd <8 x double> %add_add_mul_coef0_load306_broadcast_Ain_load310_offset_load_mul_coef1_load315_broadcast_add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load_mul_coef2_load364_broadcast_add_add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load_Ain_load408_offset_load.us, %mul_coef3_load413_broadcast_add_add_add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load_Ain_load449_offset_load_Ain_load457_offset_load.us
  %mask0.i.i11.us = shufflevector <8 x i32> %"internal_mask&function_mask472.us", <8 x i32> undef, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %mask1.i.i12.us = shufflevector <8 x i32> %"internal_mask&function_mask472.us", <8 x i32> undef, <8 x i32> <i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  %mask0d.i.i13.us = bitcast <8 x i32> %mask0.i.i11.us to <4 x double>
  %mask1d.i.i14.us = bitcast <8 x i32> %mask1.i.i12.us to <4 x double>
  %val0d.i.i15.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr912.us, <4 x double> %mask0d.i.i13.us) #0
  %val1d.i.i17.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i36.us, <4 x double> %mask1d.i.i14.us) #0
  %vald.i.i18.us = shufflevector <4 x double> %val0d.i.i15.us, <4 x double> %val1d.i.i17.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mul__Ain_load480_offset_load.us = fmul <8 x double> %vald.i.i18.us, <double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00>
  %ptr939.us = getelementptr i8* %Aout_load488_ptr2int_2void, i64 %36, !filename !0, !first_line !20, !first_column !21, !last_line !20, !last_column !22
  %val0d.i.i5.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr939.us, <4 x double> %mask0d.i.i13.us) #0
  %ptr1.i.i6.us = getelementptr i8* %Aout_load488_ptr2int_2void, i64 %ptr912.sum.us
  %val1d.i.i7.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i6.us, <4 x double> %mask1d.i.i14.us) #0
  %vald.i.i8.us = shufflevector <4 x double> %val0d.i.i5.us, <4 x double> %val1d.i.i7.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %sub_mul__Ain_load480_offset_load_Aout_load488_offset_load.us = fsub <8 x double> %mul__Ain_load480_offset_load.us, %vald.i.i8.us
  %ptr948.us = getelementptr i8* %vsq_load494_ptr2int_2void, i64 %36, !filename !0, !first_line !23, !first_column !24, !last_line !23, !last_column !7
  %val0d.i.i.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr948.us, <4 x double> %mask0d.i.i13.us) #0
  %ptr1.i.i.us = getelementptr i8* %vsq_load494_ptr2int_2void, i64 %ptr912.sum.us
  %val1d.i.i.us = tail call <4 x double> @llvm.x86.avx.maskload.pd.256(i8* %ptr1.i.i.us, <4 x double> %mask1d.i.i14.us) #0
  %vald.i.i.us = shufflevector <4 x double> %val0d.i.i.us, <4 x double> %val1d.i.i.us, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mul_vsq_load494_offset_load_div_load499.us = fmul <8 x double> %add_add_add_mul_coef0_load306_broadcast_Ain_load310_offset_load_mul_coef1_load315_broadcast_add_add_add_add_add_Ain_load319_offset_load_Ain_load327_offset_load_Ain_load335_offset_load_Ain_load343_offset_load_Ain_load351_offset_load_Ain_load359_offset_load_mul_coef2_load364_broadcast_add_add_add_add_add_Ain_load368_offset_load_Ain_load376_offset_load_Ain_load384_offset_load_Ain_load392_offset_load_Ain_load400_offset_load_Ain_load408_offset_load_mul_coef3_load413_broadcast_add_add_add_add_add_Ain_load417_offset_load_Ain_load425_offset_load_Ain_load433_offset_load_Ain_load441_offset_load_Ain_load449_offset_load_Ain_load457_offset_load.us, %vald.i.i.us
  %add_sub_mul__Ain_load480_offset_load_Aout_load488_offset_load_mul_vsq_load494_offset_load_div_load499.us = fadd <8 x double> %sub_mul__Ain_load480_offset_load_Aout_load488_offset_load.us, %mul_vsq_load494_offset_load_div_load499.us
  %val0.i.i253.us = shufflevector <8 x double> %add_sub_mul__Ain_load480_offset_load_Aout_load488_offset_load_mul_vsq_load494_offset_load_div_load499.us, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %val1.i.i254.us = shufflevector <8 x double> %add_sub_mul__Ain_load480_offset_load_Aout_load488_offset_load_mul_vsq_load494_offset_load_div_load499.us, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  call void @llvm.x86.avx.maskstore.pd.256(i8* %ptr939.us, <4 x double> %mask0d.i.i13.us, <4 x double> %val0.i.i253.us) #0
  call void @llvm.x86.avx.maskstore.pd.256(i8* %ptr1.i.i6.us, <4 x double> %mask1d.i.i14.us, <4 x double> %val1.i.i254.us) #0
  br label %safe_if_after_true466.us

safe_if_after_true466.us:                         ; preds = %safe_if_run_true467.us, %for_loop288.us
  %add_xb291_load_.us = add i32 %xb291.0263.us, 8
  %less_xb_load293_x1_load294.us = icmp slt i32 %add_xb291_load_.us, %x1
  br i1 %less_xb_load293_x1_load294.us, label %for_loop288.us, label %for_exit289.us

for_loop288.lr.ph.us:                             ; preds = %for_exit289.us, %for_test286.preheader.lr.ph
  %y280.0265.us = phi i32 [ %y_load517_plus1.us, %for_exit289.us ], [ %y0, %for_test286.preheader.lr.ph ]
  %mul_y_load302_Nx_load303.us = mul i32 %y280.0265.us, %Nx
  %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us = add i32 %mul_y_load302_Nx_load303.us, %mul_z_load300_Nxy_load301
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast788.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %Nx
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast815.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul__Nx_load382
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast824.elt0.us = sub i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %Nx
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast833.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul__Nx_load431
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast842.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul__Nx_load390
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast851.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul_Nx_load_Ny_load
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast860.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul__Nx_load439
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast869.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul__Nxy_load399
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast878.elt0.us = sub i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul_Nx_load_Ny_load
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast887.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul__Nxy_load448
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast896.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul__Nxy_load407
  %add_add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303_broadcast_xb_load298_broadcast914.elt0.us = add i32 %add_mul_z_load300_Nxy_load301_mul_y_load302_Nx_load303.us, %mul__Nxy_load456
  br label %for_loop288.us

for_exit278:                                      ; preds = %for_exit289.us, %for_test286.preheader.lr.ph, %for_test275.preheader
  %z_load518_plus1 = add i32 %z269.0268, 1
  %exitcond272 = icmp eq i32 %z_load518_plus1, %z1
  br i1 %exitcond272, label %for_exit, label %for_test275.preheader
}

; Function Attrs: nounwind
define internal void @stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_({ i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* noalias nocapture, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) #3 {
allocas:
  %x01 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 0
  %x02 = load i32* %x01, align 4
  %x13 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 1
  %x14 = load i32* %x13, align 4
  %y05 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 2
  %y06 = load i32* %y05, align 4
  %y17 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 3
  %y18 = load i32* %y17, align 4
  %z09 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 4
  %z010 = load i32* %z09, align 4
  %Nx11 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 5
  %Nx12 = load i32* %Nx11, align 4
  %Ny13 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 6
  %Ny14 = load i32* %Ny13, align 4
  %coef17 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 8
  %coef18 = load double** %coef17, align 8
  %vsq19 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 9
  %vsq20 = load double** %vsq19, align 8
  %Ain21 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 10
  %Ain22 = load double** %Ain21, align 8
  %Aout23 = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 11
  %Aout24 = load double** %Aout23, align 8
  %task_struct_mask = getelementptr { i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }* %0, i64 0, i32 12
  %mask = load <8 x i32>* %task_struct_mask, align 32
  %floatmask.i = bitcast <8 x i32> %mask to <8 x float>
  %v.i = tail call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i) #1
  %cmp.i = icmp eq i32 %v.i, 255
  %add_z0_load_taskIndex_load = add i32 %z010, %3
  %add_z0_load27_taskIndex_load28 = add i32 %3, 1
  %add_add_z0_load27_taskIndex_load28_ = add i32 %add_z0_load27_taskIndex_load28, %z010
  br i1 %cmp.i, label %all_on, label %some_on

all_on:                                           ; preds = %allocas
  tail call fastcc void @stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_(i32 %x02, i32 %x14, i32 %y06, i32 %y18, i32 %add_z0_load_taskIndex_load, i32 %add_add_z0_load27_taskIndex_load28_, i32 %Nx12, i32 %Ny14, double* %coef18, double* %vsq20, double* %Ain22, double* %Aout24, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>)
  ret void

some_on:                                          ; preds = %allocas
  tail call fastcc void @stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_(i32 %x02, i32 %x14, i32 %y06, i32 %y18, i32 %add_z0_load_taskIndex_load, i32 %add_add_z0_load27_taskIndex_load28_, i32 %Nx12, i32 %Ny14, double* %coef18, double* %vsq20, double* %Ain22, double* %Aout24, <8 x i32> %mask)
  ret void
}

; Function Attrs: nounwind
define void @loop_stencil_ispc_tasks(i32 %t0, i32 %t1, i32 %x0, i32 %x1, i32 %y0, i32 %y1, i32 %z0, i32 %z1, i32 %Nx, i32 %Ny, i32 %Nz, double* %coef, double* %vsq, double* %Aeven, double* %Aodd) #3 {
allocas:
  %launch_group_handle = alloca i8*, align 8
  store i8* null, i8** %launch_group_handle, align 8
  %less_t_load_t1_load166 = icmp slt i32 %t0, %t1
  br i1 %less_t_load_t1_load166, label %for_loop.lr.ph, label %post_sync73

for_loop.lr.ph:                                   ; preds = %allocas
  %sub_z1_load_z0_load23 = sub i32 %z1, %z0
  br label %for_loop

for_loop:                                         ; preds = %post_sync, %for_loop.lr.ph
  %t.0167 = phi i32 [ %t0, %for_loop.lr.ph ], [ %t_load69_plus1, %post_sync ]
  %bitop = and i32 %t.0167, 1
  %equal_bitop_ = icmp eq i32 %bitop, 0
  %args_ptr = call i8* @ISPCAlloc(i8** %launch_group_handle, i64 96, i32 32)
  %funarg = bitcast i8* %args_ptr to i32*
  store i32 %x0, i32* %funarg, align 4
  %funarg24 = getelementptr i8* %args_ptr, i64 4
  %0 = bitcast i8* %funarg24 to i32*
  store i32 %x1, i32* %0, align 4
  %funarg25 = getelementptr i8* %args_ptr, i64 8
  %1 = bitcast i8* %funarg25 to i32*
  store i32 %y0, i32* %1, align 4
  %funarg26 = getelementptr i8* %args_ptr, i64 12
  %2 = bitcast i8* %funarg26 to i32*
  store i32 %y1, i32* %2, align 4
  %funarg27 = getelementptr i8* %args_ptr, i64 16
  %3 = bitcast i8* %funarg27 to i32*
  store i32 %z0, i32* %3, align 4
  %funarg28 = getelementptr i8* %args_ptr, i64 20
  %4 = bitcast i8* %funarg28 to i32*
  store i32 %Nx, i32* %4, align 4
  %funarg29 = getelementptr i8* %args_ptr, i64 24
  %5 = bitcast i8* %funarg29 to i32*
  store i32 %Ny, i32* %5, align 4
  %funarg30 = getelementptr i8* %args_ptr, i64 28
  %6 = bitcast i8* %funarg30 to i32*
  store i32 %Nz, i32* %6, align 4
  %funarg31 = getelementptr i8* %args_ptr, i64 32
  %7 = bitcast i8* %funarg31 to double**
  store double* %coef, double** %7, align 8
  %funarg32 = getelementptr i8* %args_ptr, i64 40
  %8 = bitcast i8* %funarg32 to double**
  store double* %vsq, double** %8, align 8
  %funarg33 = getelementptr i8* %args_ptr, i64 48
  %9 = bitcast i8* %funarg33 to double**
  br i1 %equal_bitop_, label %if_then, label %if_else

for_exit:                                         ; preds = %post_sync
  %launch_group_handle_load70.pre = load i8** %launch_group_handle, align 8
  %cmp71 = icmp eq i8* %launch_group_handle_load70.pre, null
  br i1 %cmp71, label %post_sync73, label %call_sync72

if_then:                                          ; preds = %for_loop
  store double* %Aeven, double** %9, align 8
  %funarg34 = getelementptr i8* %args_ptr, i64 56
  %10 = bitcast i8* %funarg34 to double**
  store double* %Aodd, double** %10, align 8
  %funarg_mask = getelementptr i8* %args_ptr, i64 64
  %11 = bitcast i8* %funarg_mask to <8 x i32>*
  store <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32>* %11, align 32
  call void @ISPCLaunch(i8** %launch_group_handle, i8* bitcast (void ({ i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_ to i8*), i8* %args_ptr, i32 %sub_z1_load_z0_load23, i32 1, i32 1)
  br label %if_exit

if_else:                                          ; preds = %for_loop
  store double* %Aodd, double** %9, align 8
  %funarg64 = getelementptr i8* %args_ptr, i64 56
  %12 = bitcast i8* %funarg64 to double**
  store double* %Aeven, double** %12, align 8
  %funarg_mask67 = getelementptr i8* %args_ptr, i64 64
  %13 = bitcast i8* %funarg_mask67 to <8 x i32>*
  store <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32>* %13, align 32
  call void @ISPCLaunch(i8** %launch_group_handle, i8* bitcast (void ({ i32, i32, i32, i32, i32, i32, i32, i32, double*, double*, double*, double*, <8 x i32> }*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_ to i8*), i8* %args_ptr, i32 %sub_z1_load_z0_load23, i32 1, i32 1)
  br label %if_exit

if_exit:                                          ; preds = %if_else, %if_then
  %launch_group_handle_load = load i8** %launch_group_handle, align 8
  %cmp = icmp eq i8* %launch_group_handle_load, null
  br i1 %cmp, label %post_sync, label %call_sync

call_sync:                                        ; preds = %if_exit
  call void @ISPCSync(i8* %launch_group_handle_load)
  store i8* null, i8** %launch_group_handle, align 8
  br label %post_sync

post_sync:                                        ; preds = %call_sync, %if_exit
  %t_load69_plus1 = add i32 %t.0167, 1
  %exitcond = icmp eq i32 %t_load69_plus1, %t1
  br i1 %exitcond, label %for_exit, label %for_loop

call_sync72:                                      ; preds = %for_exit
  call void @ISPCSync(i8* %launch_group_handle_load70.pre)
  store i8* null, i8** %launch_group_handle, align 8
  br label %post_sync73

post_sync73:                                      ; preds = %call_sync72, %for_exit, %allocas
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readonly }
attributes #3 = { nounwind "target-cpu"="corei7-avx" "target-features"="+avx,+popcnt,+cmov" }

!0 = metadata !{metadata !"stencil.ispc"}
!1 = metadata !{i32 68}
!2 = metadata !{i32 69}
!3 = metadata !{i32 113}
!4 = metadata !{i32 22}
!5 = metadata !{i32 66}
!6 = metadata !{i32 71}
!7 = metadata !{i32 23}
!8 = metadata !{i32 67}
!9 = metadata !{i32 74}
!10 = metadata !{i32 72}
!11 = metadata !{i32 24}
!12 = metadata !{i32 70}
!13 = metadata !{i32 114}
!14 = metadata !{i32 75}
!15 = metadata !{i32 115}
!16 = metadata !{i32 73}
!17 = metadata !{i32 76}
!18 = metadata !{i32 21}
!19 = metadata !{i32 64}
!20 = metadata !{i32 79}
!21 = metadata !{i32 112}
!22 = metadata !{i32 156}
!23 = metadata !{i32 80}
!24 = metadata !{i32 13}
