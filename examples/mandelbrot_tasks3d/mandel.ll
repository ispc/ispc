; ModuleID = 'mandelbrot_task.bc'
target datalayout = "e-p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-n16:32:64"
target triple = "nvptx64"

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() #0

; Function Attrs: alwaysinline nounwind readnone
define <1 x i8> @__vselect_i8(<1 x i8>, <1 x i8>, <1 x i32> %mask) #1 {
  %m = extractelement <1 x i32> %mask, i32 0
  %cmp = icmp eq i32 %m, 0
  %d0 = extractelement <1 x i8> %0, i32 0
  %d1 = extractelement <1 x i8> %1, i32 0
  %sel = select i1 %cmp, i8 %d0, i8 %d1
  %r = insertelement <1 x i8> undef, i8 %sel, i32 0
  ret <1 x i8> %r
}

; Function Attrs: alwaysinline nounwind readnone
define <1 x i16> @__vselect_i16(<1 x i16>, <1 x i16>, <1 x i32> %mask) #1 {
  %m = extractelement <1 x i32> %mask, i32 0
  %cmp = icmp eq i32 %m, 0
  %d0 = extractelement <1 x i16> %0, i32 0
  %d1 = extractelement <1 x i16> %1, i32 0
  %sel = select i1 %cmp, i16 %d0, i16 %d1
  %r = insertelement <1 x i16> undef, i16 %sel, i32 0
  ret <1 x i16> %r
}

; Function Attrs: alwaysinline nounwind readnone
define <1 x i64> @__vselect_i64(<1 x i64>, <1 x i64>, <1 x i32> %mask) #1 {
  %m = extractelement <1 x i32> %mask, i32 0
  %cmp = icmp eq i32 %m, 0
  %d0 = extractelement <1 x i64> %0, i32 0
  %d1 = extractelement <1 x i64> %1, i32 0
  %sel = select i1 %cmp, i64 %d0, i64 %d1
  %r = insertelement <1 x i64> undef, i64 %sel, i32 0
  ret <1 x i64> %r
}

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.rsqrt.approx.d(double) #0

; Function Attrs: alwaysinline nounwind
define void @__aos_to_soa4_float1(<1 x float> %v0, <1 x float> %v1, <1 x float> %v2, <1 x float> %v3, <1 x float>* noalias nocapture %out0, <1 x float>* noalias nocapture %out1, <1 x float>* noalias nocapture %out2, <1 x float>* noalias nocapture %out3) #2 {
  store <1 x float> %v0, <1 x float>* %out0, align 4
  store <1 x float> %v1, <1 x float>* %out1, align 4
  store <1 x float> %v2, <1 x float>* %out2, align 4
  store <1 x float> %v3, <1 x float>* %out3, align 4
  ret void
}

; Function Attrs: alwaysinline nounwind
define void @__soa_to_aos4_float1(<1 x float> %v0, <1 x float> %v1, <1 x float> %v2, <1 x float> %v3, <1 x float>* noalias nocapture %out0, <1 x float>* noalias nocapture %out1, <1 x float>* noalias nocapture %out2, <1 x float>* noalias nocapture %out3) #2 {
  store <1 x float> %v0, <1 x float>* %out0, align 4
  store <1 x float> %v1, <1 x float>* %out1, align 4
  store <1 x float> %v2, <1 x float>* %out2, align 4
  store <1 x float> %v3, <1 x float>* %out3, align 4
  ret void
}

; Function Attrs: nounwind
define void @__aos_to_soa3_float1(<1 x float> %v0, <1 x float> %v1, <1 x float> %v2, <1 x float>* nocapture %out0, <1 x float>* nocapture %out1, <1 x float>* nocapture %out2) #3 {
  store <1 x float> %v0, <1 x float>* %out0, align 4
  store <1 x float> %v1, <1 x float>* %out1, align 4
  store <1 x float> %v2, <1 x float>* %out2, align 4
  ret void
}

; Function Attrs: nounwind
define void @__soa_to_aos3_float1(<1 x float> %v0, <1 x float> %v1, <1 x float> %v2, <1 x float>* nocapture %out0, <1 x float>* nocapture %out1, <1 x float>* nocapture %out2) #3 {
  store <1 x float> %v0, <1 x float>* %out0, align 4
  store <1 x float> %v1, <1 x float>* %out1, align 4
  store <1 x float> %v2, <1 x float>* %out2, align 4
  ret void
}

; Function Attrs: alwaysinline nounwind readonly
define <1 x double> @__rsqrt_varying_double(<1 x double> %v) #4 {
  %vs = extractelement <1 x double> %v, i32 0
  %rs = tail call double @llvm.nvvm.rsqrt.approx.d(double %vs)
  %rv = insertelement <1 x double> undef, double %rs, i32 0
  ret <1 x double> %rv
}

; Function Attrs: nounwind
define void @mandelbrot_scanline___unfunfunfunfuniuniuniuniuniun_3C_uni_3E_({ float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* noalias nocapture, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) #5 {
allocas:
  %x01 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 0
  %x02 = load float* %x01, align 4
  %dx3 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 1
  %dx4 = load float* %dx3, align 4
  %y05 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 2
  %y06 = load float* %y05, align 4
  %dy7 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 3
  %dy8 = load float* %dy7, align 4
  %width9 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 4
  %width10 = load i32* %width9, align 4
  %height11 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 5
  %height12 = load i32* %height11, align 4
  %xspan13 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 6
  %xspan14 = load i32* %xspan13, align 4
  %yspan15 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 7
  %yspan16 = load i32* %yspan15, align 4
  %maxIterations17 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 8
  %maxIterations18 = load i32* %maxIterations17, align 4
  %output19 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 9
  %output20 = load i32** %output19, align 8
  %task_struct_mask = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* %0, i64 0, i32 10
  %mask = load <1 x i32>* %task_struct_mask, align 4
  %item.i = extractelement <1 x i32> %mask, i32 0
  %cmp.i = icmp slt i32 %item.i, 0
  %bid.i.i = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3
  %mul_calltmp_xspan_load = mul i32 %bid.i.i, %xspan14
  %add_xstart_load_xspan_load25 = add i32 %mul_calltmp_xspan_load, %xspan14
  %c.i.i = icmp slt i32 %add_xstart_load_xspan_load25, %width10
  %r.i.i = select i1 %c.i.i, i32 %add_xstart_load_xspan_load25, i32 %width10
  %bid.i.i177 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3
  %mul_calltmp31_yspan_load = mul i32 %bid.i.i177, %yspan16
  %add_ystart_load_yspan_load32 = add i32 %mul_calltmp31_yspan_load, %yspan16
  %c.i.i178 = icmp slt i32 %add_ystart_load_yspan_load32, %height12
  %r.i.i179 = select i1 %c.i.i178, i32 %add_ystart_load_yspan_load32, i32 %height12
  %less_yi_load_yend_load319 = icmp slt i32 %mul_calltmp31_yspan_load, %r.i.i179
  br i1 %cmp.i, label %for_test.preheader, label %for_test104.preheader

for_test104.preheader:                            ; preds = %allocas
  br i1 %less_yi_load_yend_load319, label %for_test115.preheader.lr.ph, label %for_exit

for_test115.preheader.lr.ph:                      ; preds = %for_test104.preheader
  %less_xi_load122_xend_load123331 = icmp slt i32 %mul_calltmp_xspan_load, %r.i.i
  %maxIterations_load140_broadcast_init = insertelement <1 x i32> undef, i32 %maxIterations18, i32 0
  %less_i_load_count_load.i321 = icmp sgt <1 x i32> %maxIterations_load140_broadcast_init, zeroinitializer
  %"oldMask&test.i322" = select <1 x i1> %less_i_load_count_load.i321, <1 x i32> <i32 -1>, <1 x i32> zeroinitializer
  %"internal_mask&function_mask10.i323" = and <1 x i32> %"oldMask&test.i322", %mask
  %item.i.i324 = extractelement <1 x i32> %"internal_mask&function_mask10.i323", i32 0
  %cmp.i.i325 = icmp slt i32 %item.i.i324, 0
  %11 = xor i32 %height12, -1
  %12 = add i32 %bid.i.i177, 1
  %13 = mul i32 %yspan16, %12
  %14 = xor i32 %13, -1
  %15 = icmp sgt i32 %11, %14
  %smax336 = select i1 %15, i32 %11, i32 %14
  %16 = xor i32 %smax336, -1
  br label %for_test115.preheader

for_test.preheader:                               ; preds = %allocas
  br i1 %less_yi_load_yend_load319, label %for_test40.preheader.lr.ph, label %for_exit

for_test40.preheader.lr.ph:                       ; preds = %for_test.preheader
  %less_xi_load_xend_load317 = icmp slt i32 %mul_calltmp_xspan_load, %r.i.i
  %maxIterations_load_broadcast_init = insertelement <1 x i32> undef, i32 %maxIterations18, i32 0
  %less_i_load_count_load.i204308 = icmp sgt <1 x i32> %maxIterations_load_broadcast_init, zeroinitializer
  %"oldMask&test.i205309" = select <1 x i1> %less_i_load_count_load.i204308, <1 x i32> <i32 -1>, <1 x i32> zeroinitializer
  %item.i.i206310 = extractelement <1 x i32> %"oldMask&test.i205309", i32 0
  %cmp.i.i207311 = icmp slt i32 %item.i.i206310, 0
  %output_load_ptr2int = ptrtoint i32* %output20 to i64
  %17 = xor i32 %height12, -1
  %18 = add i32 %bid.i.i177, 1
  %19 = mul i32 %yspan16, %18
  %20 = xor i32 %19, -1
  %21 = icmp sgt i32 %17, %20
  %smax = select i1 %21, i32 %17, i32 %20
  %22 = xor i32 %smax, -1
  br label %for_test40.preheader

for_test40.preheader:                             ; preds = %for_exit43, %for_test40.preheader.lr.ph
  %yi.0320 = phi i32 [ %mul_calltmp31_yspan_load, %for_test40.preheader.lr.ph ], [ %yi_load77_plus1, %for_exit43 ]
  br i1 %less_xi_load_xend_load317, label %for_loop42.lr.ph, label %for_exit43

for_loop42.lr.ph:                                 ; preds = %for_test40.preheader
  %yi_load52_to_float = sitofp i32 %yi.0320 to float
  %mul_yi_load52_to_float_dy_load = fmul float %dy8, %yi_load52_to_float
  %add_y0_load_mul_yi_load52_to_float_dy_load = fadd float %y06, %mul_yi_load52_to_float_dy_load
  %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init = insertelement <1 x float> undef, float %add_y0_load_mul_yi_load52_to_float_dy_load, i32 0
  %mul_yi_load56_width_load57 = mul i32 %yi.0320, %width10
  br i1 %cmp.i.i207311, label %for_loop.i229.lr.ph.us, label %mandel___vyfvyfvyi.exit244

mandel___vyfvyfvyi.exit244.us:                    ; preds = %for_step.i212.us
  %tid.i.i189.us = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %tid.i.i.i190.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %sub_calltmp3_.i191.us = add i32 %tid.i.i.i190.us, -1
  %bitop.i192.us = and i32 %sub_calltmp3_.i191.us, %tid.i.i189.us
  %add_xi_load62_calltmp65.us = add i32 %bitop.i192.us, %xi.0318.us
  %less_add_xi_load62_calltmp65_xend_load66.us = icmp slt i32 %add_xi_load62_calltmp65.us, %r.i.i
  br i1 %less_add_xi_load62_calltmp65_xend_load66.us, label %if_then.us, label %if_exit.us

if_then.us:                                       ; preds = %mandel___vyfvyfvyi.exit244.us
  %tid.i.i.i194.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %sub_calltmp3_.i195.us = add i32 %tid.i.i.i194.us, 1073741823
  %tid.i.i193.us = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %bitop.i196.us = and i32 %sub_calltmp3_.i195.us, %tid.i.i193.us
  %add_xi_load58_calltmp61.us = add i32 %xi.0318.us, %mul_yi_load56_width_load57
  %add_mul_yi_load56_width_load57_add_xi_load58_calltmp61.us = add i32 %add_xi_load58_calltmp61.us, %bitop.i196.us
  %23 = shl i32 %add_mul_yi_load56_width_load57_add_xi_load58_calltmp61.us, 2
  %iptr__id.i264.rhs.us = sext i32 %23 to i64
  %iptr__id.i264.us = add i64 %iptr__id.i264.rhs.us, %output_load_ptr2int
  %ptr__id.i265.us = inttoptr i64 %iptr__id.i264.us to i32*
  store i32 %sel.i.i291.us, i32* %ptr__id.i265.us, align 4
  br label %if_exit.us

if_exit.us:                                       ; preds = %if_then.us, %mandel___vyfvyfvyi.exit244.us
  %tid.i.i188.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %add_xi_load76_calltmp74.us = add i32 %tid.i.i188.us, %xi.0318.us
  %less_xi_load_xend_load.us = icmp slt i32 %add_xi_load76_calltmp74.us, %r.i.i
  br i1 %less_xi_load_xend_load.us, label %for_loop.i229.lr.ph.us, label %for_exit43

for_loop.i229.us:                                 ; preds = %for_loop.i229.lr.ph.us, %for_step.i212.us
  %"oldMask&test.i205316.us" = phi <1 x i32> [ %"oldMask&test.i205309", %for_loop.i229.lr.ph.us ], [ %"oldMask&test.i205.us", %for_step.i212.us ]
  %break_lanes_memory.0.i201315.us = phi <1 x i32> [ zeroinitializer, %for_loop.i229.lr.ph.us ], [ %"mask|break_mask.i220.us", %for_step.i212.us ]
  %r.i.i292295314.us = phi <1 x i32> [ zeroinitializer, %for_loop.i229.lr.ph.us ], [ %r.i.i292.us, %for_step.i212.us ]
  %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init301313.us = phi <1 x float> [ %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init.us, %for_loop.i229.lr.ph.us ], [ %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init300.us, %for_step.i212.us ]
  %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init303312.us = phi <1 x float> [ %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init, %for_loop.i229.lr.ph.us ], [ %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init302.us, %for_step.i212.us ]
  %mul_z_re_load_z_re_load13.i214.us = fmul <1 x float> %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init301313.us, %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init301313.us
  %mul_z_im_load_z_im_load14.i216.us = fmul <1 x float> %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init303312.us, %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init303312.us
  %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i217.us = fadd <1 x float> %mul_z_im_load_z_im_load14.i216.us, %mul_z_re_load_z_re_load13.i214.us
  %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i218.us = fcmp ugt <1 x float> %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i217.us, <float 4.000000e+00>
  %"oldMask&test16.i219.us" = select <1 x i1> %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i218.us, <1 x i32> %"oldMask&test.i205316.us", <1 x i32> zeroinitializer
  %"mask|break_mask.i220.us" = or <1 x i32> %"oldMask&test16.i219.us", %break_lanes_memory.0.i201315.us
  %item.i63.i222.us = extractelement <1 x i32> %"mask|break_mask.i220.us", i32 0
  %v.i64.i223.us = lshr i32 %item.i63.i222.us, 31
  %item.i62.i225.us = extractelement <1 x i32> %"oldMask&test.i205316.us", i32 0
  %v.i.i226.us = lshr i32 %item.i62.i225.us, 31
  %"equal_finished&func_internal_mask&function_mask12.i228.us" = icmp eq i32 %v.i64.i223.us, %v.i.i226.us
  br i1 %"equal_finished&func_internal_mask&function_mask12.i228.us", label %for_step.i212.us, label %not_all_continued_or_breaked.i243.us

not_all_continued_or_breaked.i243.us:             ; preds = %for_loop.i229.us
  %"!(break|continue)_lanes.i232.us" = xor <1 x i32> %"mask|break_mask.i220.us", <i32 -1>
  %new_mask28.i233.us = and <1 x i32> %"oldMask&test.i205316.us", %"!(break|continue)_lanes.i232.us"
  %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i238.us = fsub <1 x float> %mul_z_re_load_z_re_load13.i214.us, %mul_z_im_load_z_im_load14.i216.us
  %mul__z_re_load35.i239.us = fmul <1 x float> %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init301313.us, <float 2.000000e+00>
  %mul_mul__z_re_load35_z_im_load36.i240.us = fmul <1 x float> %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init303312.us, %mul__z_re_load35.i239.us
  %add_c_re_load42_new_re_load.i241.us = fadd <1 x float> %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init.us, %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i238.us
  %add_c_im_load44_new_im_load.i242.us = fadd <1 x float> %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init, %mul_mul__z_re_load35_z_im_load36.i240.us
  br label %for_step.i212.us

for_step.i212.us:                                 ; preds = %not_all_continued_or_breaked.i243.us, %for_loop.i229.us
  %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init302.us = phi <1 x float> [ %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init303312.us, %for_loop.i229.us ], [ %add_c_im_load44_new_im_load.i242.us, %not_all_continued_or_breaked.i243.us ]
  %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init300.us = phi <1 x float> [ %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init301313.us, %for_loop.i229.us ], [ %add_c_re_load42_new_re_load.i241.us, %not_all_continued_or_breaked.i243.us ]
  %internal_mask_memory.1.i209.us = phi <1 x i32> [ zeroinitializer, %for_loop.i229.us ], [ %new_mask28.i233.us, %not_all_continued_or_breaked.i243.us ]
  %m.i.i287.us = extractelement <1 x i32> %internal_mask_memory.1.i209.us, i32 0
  %d0.i.i289.us = extractelement <1 x i32> %r.i.i292295314.us, i32 0
  %not.cmp.i.i288.us = icmp ne i32 %m.i.i287.us, 0
  %d1.i.i290.us = zext i1 %not.cmp.i.i288.us to i32
  %sel.i.i291.us = add i32 %d0.i.i289.us, %d1.i.i290.us
  %r.i.i292.us = insertelement <1 x i32> undef, i32 %sel.i.i291.us, i32 0
  %less_i_load_count_load.i204.us = icmp slt <1 x i32> %r.i.i292.us, %maxIterations_load_broadcast_init
  %"oldMask&test.i205.us" = select <1 x i1> %less_i_load_count_load.i204.us, <1 x i32> %internal_mask_memory.1.i209.us, <1 x i32> zeroinitializer
  %item.i.i206.us = extractelement <1 x i32> %"oldMask&test.i205.us", i32 0
  %cmp.i.i207.us = icmp slt i32 %item.i.i206.us, 0
  br i1 %cmp.i.i207.us, label %for_loop.i229.us, label %mandel___vyfvyfvyi.exit244.us

for_loop.i229.lr.ph.us:                           ; preds = %if_exit.us, %for_loop42.lr.ph
  %xi.0318.us = phi i32 [ %add_xi_load76_calltmp74.us, %if_exit.us ], [ %mul_calltmp_xspan_load, %for_loop42.lr.ph ]
  %tid.i.i180.us = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %tid.i.i.i181.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %sub_calltmp3_.i182.us = add i32 %tid.i.i.i181.us, -1
  %bitop.i183.us = and i32 %sub_calltmp3_.i182.us, %tid.i.i180.us
  %add_xi_load48_calltmp51.us = add i32 %bitop.i183.us, %xi.0318.us
  %add_xi_load48_calltmp51_to_float.us = sitofp i32 %add_xi_load48_calltmp51.us to float
  %mul_add_xi_load48_calltmp51_to_float_dx_load.us = fmul float %dx4, %add_xi_load48_calltmp51_to_float.us
  %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load.us = fadd float %x02, %mul_add_xi_load48_calltmp51_to_float_dx_load.us
  %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init.us = insertelement <1 x float> undef, float %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load.us, i32 0
  br label %for_loop.i229.us

for_exit:                                         ; preds = %for_exit118, %for_exit43, %for_test.preheader, %for_test104.preheader
  ret void

mandel___vyfvyfvyi.exit244:                       ; preds = %if_exit, %for_loop42.lr.ph
  %xi.0318 = phi i32 [ %add_xi_load76_calltmp74, %if_exit ], [ %mul_calltmp_xspan_load, %for_loop42.lr.ph ]
  %tid.i.i189 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %tid.i.i.i190 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %sub_calltmp3_.i191 = add i32 %tid.i.i.i190, -1
  %bitop.i192 = and i32 %sub_calltmp3_.i191, %tid.i.i189
  %add_xi_load62_calltmp65 = add i32 %bitop.i192, %xi.0318
  %less_add_xi_load62_calltmp65_xend_load66 = icmp slt i32 %add_xi_load62_calltmp65, %r.i.i
  br i1 %less_add_xi_load62_calltmp65_xend_load66, label %if_then, label %if_exit

for_exit43:                                       ; preds = %if_exit, %if_exit.us, %for_test40.preheader
  %yi_load77_plus1 = add i32 %yi.0320, 1
  %exitcond = icmp eq i32 %yi_load77_plus1, %22
  br i1 %exitcond, label %for_exit, label %for_test40.preheader

if_then:                                          ; preds = %mandel___vyfvyfvyi.exit244
  %tid.i.i.i194 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %sub_calltmp3_.i195 = add i32 %tid.i.i.i194, 1073741823
  %tid.i.i193 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %bitop.i196 = and i32 %sub_calltmp3_.i195, %tid.i.i193
  %add_xi_load58_calltmp61 = add i32 %xi.0318, %mul_yi_load56_width_load57
  %add_mul_yi_load56_width_load57_add_xi_load58_calltmp61 = add i32 %add_xi_load58_calltmp61, %bitop.i196
  %24 = shl i32 %add_mul_yi_load56_width_load57_add_xi_load58_calltmp61, 2
  %iptr__id.i264.rhs = sext i32 %24 to i64
  %iptr__id.i264 = add i64 %iptr__id.i264.rhs, %output_load_ptr2int
  %ptr__id.i265 = inttoptr i64 %iptr__id.i264 to i32*
  store i32 0, i32* %ptr__id.i265, align 4
  br label %if_exit

if_exit:                                          ; preds = %if_then, %mandel___vyfvyfvyi.exit244
  %tid.i.i188 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %add_xi_load76_calltmp74 = add i32 %tid.i.i188, %xi.0318
  %less_xi_load_xend_load = icmp slt i32 %add_xi_load76_calltmp74, %r.i.i
  br i1 %less_xi_load_xend_load, label %mandel___vyfvyfvyi.exit244, label %for_exit43

for_test115.preheader:                            ; preds = %for_exit118, %for_test115.preheader.lr.ph
  %yi109.0335 = phi i32 [ %mul_calltmp31_yspan_load, %for_test115.preheader.lr.ph ], [ %yi_load171_plus1, %for_exit118 ]
  br i1 %less_xi_load122_xend_load123331, label %for_loop117.lr.ph, label %for_exit118

for_loop117.lr.ph:                                ; preds = %for_test115.preheader
  %yi_load135_to_float = sitofp i32 %yi109.0335 to float
  %mul_yi_load135_to_float_dy_load136 = fmul float %dy8, %yi_load135_to_float
  %add_y0_load134_mul_yi_load135_to_float_dy_load136 = fadd float %y06, %mul_yi_load135_to_float_dy_load136
  %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init = insertelement <1 x float> undef, float %add_y0_load134_mul_yi_load135_to_float_dy_load136, i32 0
  br i1 %cmp.i.i325, label %for_loop.i.lr.ph.us, label %if_exit159

if_exit159.us:                                    ; preds = %for_step.i.us
  %tid.i.i.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %add_xi120_load_calltmp169.us = add i32 %tid.i.i.us, %xi120.0332.us
  %less_xi_load122_xend_load123.us = icmp slt i32 %add_xi120_load_calltmp169.us, %r.i.i
  br i1 %less_xi_load122_xend_load123.us, label %for_loop.i.lr.ph.us, label %for_exit118

for_loop.i.us:                                    ; preds = %for_loop.i.lr.ph.us, %for_step.i.us
  %"oldMask&test.i329.us" = phi <1 x i32> [ %"oldMask&test.i322", %for_loop.i.lr.ph.us ], [ %"oldMask&test.i.us", %for_step.i.us ]
  %break_lanes_memory.0.i328.us = phi <1 x i32> [ zeroinitializer, %for_loop.i.lr.ph.us ], [ %"mask|break_mask.i.us", %for_step.i.us ]
  %25 = phi <1 x i32> [ zeroinitializer, %for_loop.i.lr.ph.us ], [ %r.i.i261.us, %for_step.i.us ]
  %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init305327.us = phi <1 x float> [ %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init.us, %for_loop.i.lr.ph.us ], [ %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init304.us, %for_step.i.us ]
  %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init307326.us = phi <1 x float> [ %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init, %for_loop.i.lr.ph.us ], [ %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init306.us, %for_step.i.us ]
  %"internal_mask&function_mask12.i.us" = and <1 x i32> %"oldMask&test.i329.us", %mask
  %mul_z_re_load_z_re_load13.i.us = fmul <1 x float> %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init305327.us, %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init305327.us
  %mul_z_im_load_z_im_load14.i.us = fmul <1 x float> %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init307326.us, %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init307326.us
  %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i.us = fadd <1 x float> %mul_z_im_load_z_im_load14.i.us, %mul_z_re_load_z_re_load13.i.us
  %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i.us = fcmp ugt <1 x float> %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i.us, <float 4.000000e+00>
  %"oldMask&test16.i.us" = select <1 x i1> %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i.us, <1 x i32> %"oldMask&test.i329.us", <1 x i32> zeroinitializer
  %"mask|break_mask.i.us" = or <1 x i32> %"oldMask&test16.i.us", %break_lanes_memory.0.i328.us
  %"finished&func.i.us" = and <1 x i32> %"mask|break_mask.i.us", %mask
  %item.i63.i.us = extractelement <1 x i32> %"finished&func.i.us", i32 0
  %v.i64.i.us = lshr i32 %item.i63.i.us, 31
  %item.i62.i.us = extractelement <1 x i32> %"internal_mask&function_mask12.i.us", i32 0
  %v.i.i.us = lshr i32 %item.i62.i.us, 31
  %"equal_finished&func_internal_mask&function_mask12.i.us" = icmp eq i32 %v.i64.i.us, %v.i.i.us
  br i1 %"equal_finished&func_internal_mask&function_mask12.i.us", label %for_step.i.us, label %not_all_continued_or_breaked.i.us

not_all_continued_or_breaked.i.us:                ; preds = %for_loop.i.us
  %"!(break|continue)_lanes.i.us" = xor <1 x i32> %"mask|break_mask.i.us", <i32 -1>
  %new_mask28.i.us = and <1 x i32> %"oldMask&test.i329.us", %"!(break|continue)_lanes.i.us"
  %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i.us = fsub <1 x float> %mul_z_re_load_z_re_load13.i.us, %mul_z_im_load_z_im_load14.i.us
  %mul__z_re_load35.i.us = fmul <1 x float> %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init305327.us, <float 2.000000e+00>
  %mul_mul__z_re_load35_z_im_load36.i.us = fmul <1 x float> %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init307326.us, %mul__z_re_load35.i.us
  %add_c_re_load42_new_re_load.i.us = fadd <1 x float> %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init.us, %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i.us
  %add_c_im_load44_new_im_load.i.us = fadd <1 x float> %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init, %mul_mul__z_re_load35_z_im_load36.i.us
  br label %for_step.i.us

for_step.i.us:                                    ; preds = %not_all_continued_or_breaked.i.us, %for_loop.i.us
  %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init306.us = phi <1 x float> [ %add_y0_load134_mul_yi_load135_to_float_dy_load136_broadcast_init307326.us, %for_loop.i.us ], [ %add_c_im_load44_new_im_load.i.us, %not_all_continued_or_breaked.i.us ]
  %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init304.us = phi <1 x float> [ %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init305327.us, %for_loop.i.us ], [ %add_c_re_load42_new_re_load.i.us, %not_all_continued_or_breaked.i.us ]
  %internal_mask_memory.1.i.us = phi <1 x i32> [ zeroinitializer, %for_loop.i.us ], [ %new_mask28.i.us, %not_all_continued_or_breaked.i.us ]
  %m.i.i.us = extractelement <1 x i32> %internal_mask_memory.1.i.us, i32 0
  %d0.i.i259.us = extractelement <1 x i32> %25, i32 0
  %not.cmp.i.i258.us = icmp ne i32 %m.i.i.us, 0
  %d1.i.i260.us = zext i1 %not.cmp.i.i258.us to i32
  %sel.i.i.us = add i32 %d0.i.i259.us, %d1.i.i260.us
  %r.i.i261.us = insertelement <1 x i32> undef, i32 %sel.i.i.us, i32 0
  %less_i_load_count_load.i.us = icmp slt <1 x i32> %r.i.i261.us, %maxIterations_load140_broadcast_init
  %"oldMask&test.i.us" = select <1 x i1> %less_i_load_count_load.i.us, <1 x i32> %internal_mask_memory.1.i.us, <1 x i32> zeroinitializer
  %"internal_mask&function_mask10.i.us" = and <1 x i32> %"oldMask&test.i.us", %mask
  %item.i.i.us = extractelement <1 x i32> %"internal_mask&function_mask10.i.us", i32 0
  %cmp.i.i.us = icmp slt i32 %item.i.i.us, 0
  br i1 %cmp.i.i.us, label %for_loop.i.us, label %if_exit159.us

for_loop.i.lr.ph.us:                              ; preds = %if_exit159.us, %for_loop117.lr.ph
  %xi120.0332.us = phi i32 [ %add_xi120_load_calltmp169.us, %if_exit159.us ], [ %mul_calltmp_xspan_load, %for_loop117.lr.ph ]
  %tid.i.i184.us = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
  %tid.i.i.i185.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %sub_calltmp3_.i186.us = add i32 %tid.i.i.i185.us, -1
  %bitop.i187.us = and i32 %sub_calltmp3_.i186.us, %tid.i.i184.us
  %add_xi_load128_calltmp131.us = add i32 %bitop.i187.us, %xi120.0332.us
  %add_xi_load128_calltmp131_to_float.us = sitofp i32 %add_xi_load128_calltmp131.us to float
  %mul_add_xi_load128_calltmp131_to_float_dx_load132.us = fmul float %dx4, %add_xi_load128_calltmp131_to_float.us
  %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132.us = fadd float %x02, %mul_add_xi_load128_calltmp131_to_float_dx_load132.us
  %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132_broadcast_init.us = insertelement <1 x float> undef, float %add_x0_load127_mul_add_xi_load128_calltmp131_to_float_dx_load132.us, i32 0
  br label %for_loop.i.us

for_exit118:                                      ; preds = %if_exit159, %if_exit159.us, %for_test115.preheader
  %yi_load171_plus1 = add i32 %yi109.0335, 1
  %exitcond337 = icmp eq i32 %yi_load171_plus1, %16
  br i1 %exitcond337, label %for_exit, label %for_test115.preheader

if_exit159:                                       ; preds = %if_exit159, %for_loop117.lr.ph
  %xi120.0332 = phi i32 [ %add_xi120_load_calltmp169, %if_exit159 ], [ %mul_calltmp_xspan_load, %for_loop117.lr.ph ]
  %tid.i.i = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #3
  %add_xi120_load_calltmp169 = add i32 %tid.i.i, %xi120.0332
  %less_xi_load122_xend_load123 = icmp slt i32 %add_xi120_load_calltmp169, %r.i.i
  br i1 %less_xi_load122_xend_load123, label %if_exit159, label %for_exit118
}

attributes #0 = { nounwind readnone }
attributes #1 = { alwaysinline nounwind readnone }
attributes #2 = { alwaysinline nounwind }
attributes #3 = { nounwind }
attributes #4 = { alwaysinline nounwind readonly }
attributes #5 = { nounwind "target-features"="+sm_35" }
!nvvm.annotations = !{!1}
!1 = metadata !{void ({ float, float, float, float, i32, i32, i32, i32, i32, i32*, <1 x i32> }* , i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @mandelbrot_scanline___unfunfunfunfuniuniuniuniuniun_3C_uni_3E_, metadata !"kernel", i32 1}
