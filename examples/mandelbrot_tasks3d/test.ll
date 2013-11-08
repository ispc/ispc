; ModuleID = 'test.bc'
target datalayout = "e-p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-n16:32:64"
target triple = "nvptx64"

; Function Attrs: alwaysinline nounwind readnone
define <1 x i8> @__vselect_i8(<1 x i8>, <1 x i8>, <1 x i32> %mask) #0 {
  %m = extractelement <1 x i32> %mask, i32 0
  %cmp = icmp eq i32 %m, 0
  %d0 = extractelement <1 x i8> %0, i32 0
  %d1 = extractelement <1 x i8> %1, i32 0
  %sel = select i1 %cmp, i8 %d0, i8 %d1
  %r = insertelement <1 x i8> undef, i8 %sel, i32 0
  ret <1 x i8> %r
}

; Function Attrs: alwaysinline nounwind readnone
define <1 x i16> @__vselect_i16(<1 x i16>, <1 x i16>, <1 x i32> %mask) #0 {
  %m = extractelement <1 x i32> %mask, i32 0
  %cmp = icmp eq i32 %m, 0
  %d0 = extractelement <1 x i16> %0, i32 0
  %d1 = extractelement <1 x i16> %1, i32 0
  %sel = select i1 %cmp, i16 %d0, i16 %d1
  %r = insertelement <1 x i16> undef, i16 %sel, i32 0
  ret <1 x i16> %r
}

; Function Attrs: alwaysinline nounwind readnone
define <1 x i64> @__vselect_i64(<1 x i64>, <1 x i64>, <1 x i32> %mask) #0 {
  %m = extractelement <1 x i32> %mask, i32 0
  %cmp = icmp eq i32 %m, 0
  %d0 = extractelement <1 x i64> %0, i32 0
  %d1 = extractelement <1 x i64> %1, i32 0
  %sel = select i1 %cmp, i64 %d0, i64 %d1
  %r = insertelement <1 x i64> undef, i64 %sel, i32 0
  ret <1 x i64> %r
}

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.rsqrt.approx.d(double) #1

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
declare i32 @getBlockIndex0___(<1 x i32>) #5

; Function Attrs: nounwind
declare i32 @getBlockIndex1___(<1 x i32>) #5

; Function Attrs: nounwind
declare i32 @getLaneIndex___(<1 x i32>) #5

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
  br i1 %cmp.i, label %all_on, label %some_on

all_on:                                           ; preds = %allocas
  %calltmp = call i32 @getBlockIndex0___(<1 x i32> <i32 -1>)
  %mul_calltmp_xspan_load = mul i32 %calltmp, %xspan14
  %add_xstart_load_xspan_load25 = add i32 %mul_calltmp_xspan_load, %xspan14
  %c.i.i = icmp slt i32 %add_xstart_load_xspan_load25, %width10
  %r.i.i = select i1 %c.i.i, i32 %add_xstart_load_xspan_load25, i32 %width10
  %calltmp31 = call i32 @getBlockIndex1___(<1 x i32> <i32 -1>)
  %mul_calltmp31_yspan_load = mul i32 %calltmp31, %yspan16
  %add_ystart_load_yspan_load32 = add i32 %mul_calltmp31_yspan_load, %yspan16
  %c.i.i166 = icmp slt i32 %add_ystart_load_yspan_load32, %height12
  %r.i.i167 = select i1 %c.i.i166, i32 %add_ystart_load_yspan_load32, i32 %height12
  %less_yi_load_yend_load294 = icmp slt i32 %mul_calltmp31_yspan_load, %r.i.i167
  br i1 %less_yi_load_yend_load294, label %for_test40.preheader.lr.ph, label %for_exit

for_test40.preheader.lr.ph:                       ; preds = %all_on
  %less_xi_load_xend_load292 = icmp slt i32 %mul_calltmp_xspan_load, %r.i.i
  %maxIterations_load_broadcast_init = insertelement <1 x i32> undef, i32 %maxIterations18, i32 0
  %less_i_load_count_load.i179283 = icmp sgt <1 x i32> %maxIterations_load_broadcast_init, zeroinitializer
  %"oldMask&test.i180284" = select <1 x i1> %less_i_load_count_load.i179283, <1 x i32> <i32 -1>, <1 x i32> zeroinitializer
  %item.i.i181285 = extractelement <1 x i32> %"oldMask&test.i180284", i32 0
  %cmp.i.i182286 = icmp slt i32 %item.i.i181285, 0
  %output_load_ptr2int = ptrtoint i32* %output20 to i64
  %11 = xor i32 %height12, -1
  %12 = add i32 %calltmp31, 1
  %13 = mul i32 %yspan16, %12
  %14 = xor i32 %13, -1
  %15 = icmp sgt i32 %11, %14
  %smax = select i1 %15, i32 %11, i32 %14
  %16 = xor i32 %smax, -1
  br label %for_test40.preheader

some_on:                                          ; preds = %allocas
  %calltmp80 = call i32 @getBlockIndex0___(<1 x i32> %mask)
  %mul_calltmp80_xspan_load81 = mul i32 %calltmp80, %xspan14
  %add_xstart_load83_xspan_load84 = add i32 %mul_calltmp80_xspan_load81, %xspan14
  %c.i.i168 = icmp slt i32 %add_xstart_load83_xspan_load84, %width10
  %r.i.i169 = select i1 %c.i.i168, i32 %add_xstart_load83_xspan_load84, i32 %width10
  %calltmp92 = call i32 @getBlockIndex1___(<1 x i32> %mask)
  %mul_calltmp92_yspan_load93 = mul i32 %calltmp92, %yspan16
  %add_ystart_load95_yspan_load96 = add i32 %mul_calltmp92_yspan_load93, %yspan16
  %c.i.i170 = icmp slt i32 %add_ystart_load95_yspan_load96, %height12
  %r.i.i171 = select i1 %c.i.i170, i32 %add_ystart_load95_yspan_load96, i32 %height12
  %less_yi_load108_yend_load109309 = icmp slt i32 %mul_calltmp92_yspan_load93, %r.i.i171
  br i1 %less_yi_load108_yend_load109309, label %for_test112.preheader.lr.ph, label %for_exit

for_test112.preheader.lr.ph:                      ; preds = %some_on
  %less_xi_load119_xend_load120306 = icmp slt i32 %mul_calltmp80_xspan_load81, %r.i.i169
  %maxIterations_load137_broadcast_init = insertelement <1 x i32> undef, i32 %maxIterations18, i32 0
  %less_i_load_count_load.i296 = icmp sgt <1 x i32> %maxIterations_load137_broadcast_init, zeroinitializer
  %"oldMask&test.i297" = select <1 x i1> %less_i_load_count_load.i296, <1 x i32> <i32 -1>, <1 x i32> zeroinitializer
  %"internal_mask&function_mask10.i298" = and <1 x i32> %"oldMask&test.i297", %mask
  %item.i.i299 = extractelement <1 x i32> %"internal_mask&function_mask10.i298", i32 0
  %cmp.i.i300 = icmp slt i32 %item.i.i299, 0
  %17 = xor i32 %height12, -1
  %18 = add i32 %calltmp92, 1
  %19 = mul i32 %yspan16, %18
  %20 = xor i32 %19, -1
  %21 = icmp sgt i32 %17, %20
  %smax311 = select i1 %21, i32 %17, i32 %20
  %22 = xor i32 %smax311, -1
  br label %for_test112.preheader

for_test40.preheader:                             ; preds = %for_exit43, %for_test40.preheader.lr.ph
  %yi.0295 = phi i32 [ %mul_calltmp31_yspan_load, %for_test40.preheader.lr.ph ], [ %yi_load74_plus1, %for_exit43 ]
  br i1 %less_xi_load_xend_load292, label %for_loop42.lr.ph, label %for_exit43

for_loop42.lr.ph:                                 ; preds = %for_test40.preheader
  %yi_load52_to_float = sitofp i32 %yi.0295 to float
  %mul_yi_load52_to_float_dy_load = fmul float %dy8, %yi_load52_to_float
  %add_y0_load_mul_yi_load52_to_float_dy_load = fadd float %y06, %mul_yi_load52_to_float_dy_load
  %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init = insertelement <1 x float> undef, float %add_y0_load_mul_yi_load52_to_float_dy_load, i32 0
  %mul_yi_load56_width_load57 = mul i32 %yi.0295, %width10
  br i1 %cmp.i.i182286, label %for_loop.i204.lr.ph.us, label %mandel___vyfvyfvyi.exit219

mandel___vyfvyfvyi.exit219.us:                    ; preds = %for_step.i187.us
  %calltmp61.us = call i32 @getLaneIndex___(<1 x i32> <i32 -1>)
  %calltmp65.us = call i32 @getLaneIndex___(<1 x i32> <i32 -1>)
  %add_xi_load62_calltmp65.us = add i32 %calltmp65.us, %xi.0293.us
  %less_add_xi_load62_calltmp65_xend_load66.us = icmp slt i32 %add_xi_load62_calltmp65.us, %r.i.i
  br i1 %less_add_xi_load62_calltmp65_xend_load66.us, label %if_then.us, label %if_exit.us

if_then.us:                                       ; preds = %mandel___vyfvyfvyi.exit219.us
  %add_xi_load58_calltmp61.us = add i32 %xi.0293.us, %mul_yi_load56_width_load57
  %add_mul_yi_load56_width_load57_add_xi_load58_calltmp61.us = add i32 %add_xi_load58_calltmp61.us, %calltmp61.us
  %23 = shl i32 %add_mul_yi_load56_width_load57_add_xi_load58_calltmp61.us, 2
  %iptr__id.i239.rhs.us = sext i32 %23 to i64
  %iptr__id.i239.us = add i64 %iptr__id.i239.rhs.us, %output_load_ptr2int
  %ptr__id.i240.us = inttoptr i64 %iptr__id.i239.us to i32*
  store i32 %sel.i.i266.us, i32* %ptr__id.i240.us, align 4
  br label %if_exit.us

if_exit.us:                                       ; preds = %if_then.us, %mandel___vyfvyfvyi.exit219.us
  %add_xi_load73_.us = add i32 %xi.0293.us, 32
  %less_xi_load_xend_load.us = icmp slt i32 %add_xi_load73_.us, %r.i.i
  br i1 %less_xi_load_xend_load.us, label %for_loop.i204.lr.ph.us, label %for_exit43

for_loop.i204.us:                                 ; preds = %for_loop.i204.lr.ph.us, %for_step.i187.us
  %"oldMask&test.i180291.us" = phi <1 x i32> [ %"oldMask&test.i180284", %for_loop.i204.lr.ph.us ], [ %"oldMask&test.i180.us", %for_step.i187.us ]
  %break_lanes_memory.0.i176290.us = phi <1 x i32> [ zeroinitializer, %for_loop.i204.lr.ph.us ], [ %"mask|break_mask.i195.us", %for_step.i187.us ]
  %r.i.i267270289.us = phi <1 x i32> [ zeroinitializer, %for_loop.i204.lr.ph.us ], [ %r.i.i267.us, %for_step.i187.us ]
  %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init276288.us = phi <1 x float> [ %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init.us, %for_loop.i204.lr.ph.us ], [ %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init275.us, %for_step.i187.us ]
  %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init278287.us = phi <1 x float> [ %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init, %for_loop.i204.lr.ph.us ], [ %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init277.us, %for_step.i187.us ]
  %mul_z_re_load_z_re_load13.i189.us = fmul <1 x float> %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init276288.us, %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init276288.us
  %mul_z_im_load_z_im_load14.i191.us = fmul <1 x float> %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init278287.us, %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init278287.us
  %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i192.us = fadd <1 x float> %mul_z_im_load_z_im_load14.i191.us, %mul_z_re_load_z_re_load13.i189.us
  %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i193.us = fcmp ugt <1 x float> %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i192.us, <float 4.000000e+00>
  %"oldMask&test16.i194.us" = select <1 x i1> %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i193.us, <1 x i32> %"oldMask&test.i180291.us", <1 x i32> zeroinitializer
  %"mask|break_mask.i195.us" = or <1 x i32> %"oldMask&test16.i194.us", %break_lanes_memory.0.i176290.us
  %item.i63.i197.us = extractelement <1 x i32> %"mask|break_mask.i195.us", i32 0
  %v.i64.i198.us = lshr i32 %item.i63.i197.us, 31
  %item.i62.i200.us = extractelement <1 x i32> %"oldMask&test.i180291.us", i32 0
  %v.i.i201.us = lshr i32 %item.i62.i200.us, 31
  %"equal_finished&func_internal_mask&function_mask12.i203.us" = icmp eq i32 %v.i64.i198.us, %v.i.i201.us
  br i1 %"equal_finished&func_internal_mask&function_mask12.i203.us", label %for_step.i187.us, label %not_all_continued_or_breaked.i218.us

not_all_continued_or_breaked.i218.us:             ; preds = %for_loop.i204.us
  %"!(break|continue)_lanes.i207.us" = xor <1 x i32> %"mask|break_mask.i195.us", <i32 -1>
  %new_mask28.i208.us = and <1 x i32> %"oldMask&test.i180291.us", %"!(break|continue)_lanes.i207.us"
  %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i213.us = fsub <1 x float> %mul_z_re_load_z_re_load13.i189.us, %mul_z_im_load_z_im_load14.i191.us
  %mul__z_re_load35.i214.us = fmul <1 x float> %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init276288.us, <float 2.000000e+00>
  %mul_mul__z_re_load35_z_im_load36.i215.us = fmul <1 x float> %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init278287.us, %mul__z_re_load35.i214.us
  %add_c_re_load42_new_re_load.i216.us = fadd <1 x float> %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init.us, %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i213.us
  %add_c_im_load44_new_im_load.i217.us = fadd <1 x float> %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init, %mul_mul__z_re_load35_z_im_load36.i215.us
  br label %for_step.i187.us

for_step.i187.us:                                 ; preds = %not_all_continued_or_breaked.i218.us, %for_loop.i204.us
  %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init277.us = phi <1 x float> [ %add_y0_load_mul_yi_load52_to_float_dy_load_broadcast_init278287.us, %for_loop.i204.us ], [ %add_c_im_load44_new_im_load.i217.us, %not_all_continued_or_breaked.i218.us ]
  %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init275.us = phi <1 x float> [ %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init276288.us, %for_loop.i204.us ], [ %add_c_re_load42_new_re_load.i216.us, %not_all_continued_or_breaked.i218.us ]
  %internal_mask_memory.1.i184.us = phi <1 x i32> [ zeroinitializer, %for_loop.i204.us ], [ %new_mask28.i208.us, %not_all_continued_or_breaked.i218.us ]
  %m.i.i262.us = extractelement <1 x i32> %internal_mask_memory.1.i184.us, i32 0
  %d0.i.i264.us = extractelement <1 x i32> %r.i.i267270289.us, i32 0
  %not.cmp.i.i263.us = icmp ne i32 %m.i.i262.us, 0
  %d1.i.i265.us = zext i1 %not.cmp.i.i263.us to i32
  %sel.i.i266.us = add i32 %d0.i.i264.us, %d1.i.i265.us
  %r.i.i267.us = insertelement <1 x i32> undef, i32 %sel.i.i266.us, i32 0
  %less_i_load_count_load.i179.us = icmp slt <1 x i32> %r.i.i267.us, %maxIterations_load_broadcast_init
  %"oldMask&test.i180.us" = select <1 x i1> %less_i_load_count_load.i179.us, <1 x i32> %internal_mask_memory.1.i184.us, <1 x i32> zeroinitializer
  %item.i.i181.us = extractelement <1 x i32> %"oldMask&test.i180.us", i32 0
  %cmp.i.i182.us = icmp slt i32 %item.i.i181.us, 0
  br i1 %cmp.i.i182.us, label %for_loop.i204.us, label %mandel___vyfvyfvyi.exit219.us

for_loop.i204.lr.ph.us:                           ; preds = %if_exit.us, %for_loop42.lr.ph
  %xi.0293.us = phi i32 [ %add_xi_load73_.us, %if_exit.us ], [ %mul_calltmp_xspan_load, %for_loop42.lr.ph ]
  %calltmp51.us = call i32 @getLaneIndex___(<1 x i32> <i32 -1>)
  %add_xi_load48_calltmp51.us = add i32 %calltmp51.us, %xi.0293.us
  %add_xi_load48_calltmp51_to_float.us = sitofp i32 %add_xi_load48_calltmp51.us to float
  %mul_add_xi_load48_calltmp51_to_float_dx_load.us = fmul float %dx4, %add_xi_load48_calltmp51_to_float.us
  %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load.us = fadd float %x02, %mul_add_xi_load48_calltmp51_to_float_dx_load.us
  %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load_broadcast_init.us = insertelement <1 x float> undef, float %add_x0_load_mul_add_xi_load48_calltmp51_to_float_dx_load.us, i32 0
  br label %for_loop.i204.us

for_exit:                                         ; preds = %for_exit115, %for_exit43, %some_on, %all_on
  ret void

mandel___vyfvyfvyi.exit219:                       ; preds = %if_exit, %for_loop42.lr.ph
  %xi.0293 = phi i32 [ %add_xi_load73_, %if_exit ], [ %mul_calltmp_xspan_load, %for_loop42.lr.ph ]
  %calltmp51 = call i32 @getLaneIndex___(<1 x i32> <i32 -1>)
  %calltmp61 = call i32 @getLaneIndex___(<1 x i32> <i32 -1>)
  %calltmp65 = call i32 @getLaneIndex___(<1 x i32> <i32 -1>)
  %add_xi_load62_calltmp65 = add i32 %calltmp65, %xi.0293
  %less_add_xi_load62_calltmp65_xend_load66 = icmp slt i32 %add_xi_load62_calltmp65, %r.i.i
  br i1 %less_add_xi_load62_calltmp65_xend_load66, label %if_then, label %if_exit

for_exit43:                                       ; preds = %if_exit, %if_exit.us, %for_test40.preheader
  %yi_load74_plus1 = add i32 %yi.0295, 1
  %exitcond = icmp eq i32 %yi_load74_plus1, %16
  br i1 %exitcond, label %for_exit, label %for_test40.preheader

if_then:                                          ; preds = %mandel___vyfvyfvyi.exit219
  %add_xi_load58_calltmp61 = add i32 %xi.0293, %mul_yi_load56_width_load57
  %add_mul_yi_load56_width_load57_add_xi_load58_calltmp61 = add i32 %add_xi_load58_calltmp61, %calltmp61
  %24 = shl i32 %add_mul_yi_load56_width_load57_add_xi_load58_calltmp61, 2
  %iptr__id.i239.rhs = sext i32 %24 to i64
  %iptr__id.i239 = add i64 %iptr__id.i239.rhs, %output_load_ptr2int
  %ptr__id.i240 = inttoptr i64 %iptr__id.i239 to i32*
  store i32 0, i32* %ptr__id.i240, align 4
  br label %if_exit

if_exit:                                          ; preds = %if_then, %mandel___vyfvyfvyi.exit219
  %add_xi_load73_ = add i32 %xi.0293, 32
  %less_xi_load_xend_load = icmp slt i32 %add_xi_load73_, %r.i.i
  br i1 %less_xi_load_xend_load, label %mandel___vyfvyfvyi.exit219, label %for_exit43

for_test112.preheader:                            ; preds = %for_exit115, %for_test112.preheader.lr.ph
  %yi106.0310 = phi i32 [ %mul_calltmp92_yspan_load93, %for_test112.preheader.lr.ph ], [ %yi_load165_plus1, %for_exit115 ]
  br i1 %less_xi_load119_xend_load120306, label %for_loop114.lr.ph, label %for_exit115

for_loop114.lr.ph:                                ; preds = %for_test112.preheader
  %yi_load132_to_float = sitofp i32 %yi106.0310 to float
  %mul_yi_load132_to_float_dy_load133 = fmul float %dy8, %yi_load132_to_float
  %add_y0_load131_mul_yi_load132_to_float_dy_load133 = fadd float %y06, %mul_yi_load132_to_float_dy_load133
  %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init = insertelement <1 x float> undef, float %add_y0_load131_mul_yi_load132_to_float_dy_load133, i32 0
  br i1 %cmp.i.i300, label %for_loop.i.lr.ph.us, label %if_exit156

if_exit156.us:                                    ; preds = %for_step.i.us
  %calltmp147.us = call i32 @getLaneIndex___(<1 x i32> %mask)
  %calltmp151.us = call i32 @getLaneIndex___(<1 x i32> %mask)
  %add_xi117_load_.us = add i32 %xi117.0307.us, 32
  %less_xi_load119_xend_load120.us = icmp slt i32 %add_xi117_load_.us, %r.i.i169
  br i1 %less_xi_load119_xend_load120.us, label %for_loop.i.lr.ph.us, label %for_exit115

for_loop.i.us:                                    ; preds = %for_loop.i.lr.ph.us, %for_step.i.us
  %"oldMask&test.i304.us" = phi <1 x i32> [ %"oldMask&test.i297", %for_loop.i.lr.ph.us ], [ %"oldMask&test.i.us", %for_step.i.us ]
  %break_lanes_memory.0.i303.us = phi <1 x i32> [ zeroinitializer, %for_loop.i.lr.ph.us ], [ %"mask|break_mask.i.us", %for_step.i.us ]
  %25 = phi <1 x i32> [ zeroinitializer, %for_loop.i.lr.ph.us ], [ %r.i.i236.us, %for_step.i.us ]
  %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init280302.us = phi <1 x float> [ %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init.us, %for_loop.i.lr.ph.us ], [ %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init279.us, %for_step.i.us ]
  %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init282301.us = phi <1 x float> [ %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init, %for_loop.i.lr.ph.us ], [ %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init281.us, %for_step.i.us ]
  %"internal_mask&function_mask12.i.us" = and <1 x i32> %"oldMask&test.i304.us", %mask
  %mul_z_re_load_z_re_load13.i.us = fmul <1 x float> %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init280302.us, %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init280302.us
  %mul_z_im_load_z_im_load14.i.us = fmul <1 x float> %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init282301.us, %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init282301.us
  %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i.us = fadd <1 x float> %mul_z_im_load_z_im_load14.i.us, %mul_z_re_load_z_re_load13.i.us
  %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i.us = fcmp ugt <1 x float> %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i.us, <float 4.000000e+00>
  %"oldMask&test16.i.us" = select <1 x i1> %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i.us, <1 x i32> %"oldMask&test.i304.us", <1 x i32> zeroinitializer
  %"mask|break_mask.i.us" = or <1 x i32> %"oldMask&test16.i.us", %break_lanes_memory.0.i303.us
  %"finished&func.i.us" = and <1 x i32> %"mask|break_mask.i.us", %mask
  %item.i63.i.us = extractelement <1 x i32> %"finished&func.i.us", i32 0
  %v.i64.i.us = lshr i32 %item.i63.i.us, 31
  %item.i62.i.us = extractelement <1 x i32> %"internal_mask&function_mask12.i.us", i32 0
  %v.i.i.us = lshr i32 %item.i62.i.us, 31
  %"equal_finished&func_internal_mask&function_mask12.i.us" = icmp eq i32 %v.i64.i.us, %v.i.i.us
  br i1 %"equal_finished&func_internal_mask&function_mask12.i.us", label %for_step.i.us, label %not_all_continued_or_breaked.i.us

not_all_continued_or_breaked.i.us:                ; preds = %for_loop.i.us
  %"!(break|continue)_lanes.i.us" = xor <1 x i32> %"mask|break_mask.i.us", <i32 -1>
  %new_mask28.i.us = and <1 x i32> %"oldMask&test.i304.us", %"!(break|continue)_lanes.i.us"
  %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i.us = fsub <1 x float> %mul_z_re_load_z_re_load13.i.us, %mul_z_im_load_z_im_load14.i.us
  %mul__z_re_load35.i.us = fmul <1 x float> %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init280302.us, <float 2.000000e+00>
  %mul_mul__z_re_load35_z_im_load36.i.us = fmul <1 x float> %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init282301.us, %mul__z_re_load35.i.us
  %add_c_re_load42_new_re_load.i.us = fadd <1 x float> %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init.us, %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i.us
  %add_c_im_load44_new_im_load.i.us = fadd <1 x float> %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init, %mul_mul__z_re_load35_z_im_load36.i.us
  br label %for_step.i.us

for_step.i.us:                                    ; preds = %not_all_continued_or_breaked.i.us, %for_loop.i.us
  %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init281.us = phi <1 x float> [ %add_y0_load131_mul_yi_load132_to_float_dy_load133_broadcast_init282301.us, %for_loop.i.us ], [ %add_c_im_load44_new_im_load.i.us, %not_all_continued_or_breaked.i.us ]
  %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init279.us = phi <1 x float> [ %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init280302.us, %for_loop.i.us ], [ %add_c_re_load42_new_re_load.i.us, %not_all_continued_or_breaked.i.us ]
  %internal_mask_memory.1.i.us = phi <1 x i32> [ zeroinitializer, %for_loop.i.us ], [ %new_mask28.i.us, %not_all_continued_or_breaked.i.us ]
  %m.i.i.us = extractelement <1 x i32> %internal_mask_memory.1.i.us, i32 0
  %d0.i.i234.us = extractelement <1 x i32> %25, i32 0
  %not.cmp.i.i233.us = icmp ne i32 %m.i.i.us, 0
  %d1.i.i235.us = zext i1 %not.cmp.i.i233.us to i32
  %sel.i.i.us = add i32 %d0.i.i234.us, %d1.i.i235.us
  %r.i.i236.us = insertelement <1 x i32> undef, i32 %sel.i.i.us, i32 0
  %less_i_load_count_load.i.us = icmp slt <1 x i32> %r.i.i236.us, %maxIterations_load137_broadcast_init
  %"oldMask&test.i.us" = select <1 x i1> %less_i_load_count_load.i.us, <1 x i32> %internal_mask_memory.1.i.us, <1 x i32> zeroinitializer
  %"internal_mask&function_mask10.i.us" = and <1 x i32> %"oldMask&test.i.us", %mask
  %item.i.i.us = extractelement <1 x i32> %"internal_mask&function_mask10.i.us", i32 0
  %cmp.i.i.us = icmp slt i32 %item.i.i.us, 0
  br i1 %cmp.i.i.us, label %for_loop.i.us, label %if_exit156.us

for_loop.i.lr.ph.us:                              ; preds = %if_exit156.us, %for_loop114.lr.ph
  %xi117.0307.us = phi i32 [ %add_xi117_load_.us, %if_exit156.us ], [ %mul_calltmp80_xspan_load81, %for_loop114.lr.ph ]
  %calltmp128.us = call i32 @getLaneIndex___(<1 x i32> %mask)
  %add_xi_load125_calltmp128.us = add i32 %calltmp128.us, %xi117.0307.us
  %add_xi_load125_calltmp128_to_float.us = sitofp i32 %add_xi_load125_calltmp128.us to float
  %mul_add_xi_load125_calltmp128_to_float_dx_load129.us = fmul float %dx4, %add_xi_load125_calltmp128_to_float.us
  %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129.us = fadd float %x02, %mul_add_xi_load125_calltmp128_to_float_dx_load129.us
  %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129_broadcast_init.us = insertelement <1 x float> undef, float %add_x0_load124_mul_add_xi_load125_calltmp128_to_float_dx_load129.us, i32 0
  br label %for_loop.i.us

for_exit115:                                      ; preds = %if_exit156, %if_exit156.us, %for_test112.preheader
  %yi_load165_plus1 = add i32 %yi106.0310, 1
  %exitcond312 = icmp eq i32 %yi_load165_plus1, %22
  br i1 %exitcond312, label %for_exit, label %for_test112.preheader

if_exit156:                                       ; preds = %if_exit156, %for_loop114.lr.ph
  %xi117.0307 = phi i32 [ %add_xi117_load_, %if_exit156 ], [ %mul_calltmp80_xspan_load81, %for_loop114.lr.ph ]
  %calltmp128 = call i32 @getLaneIndex___(<1 x i32> %mask)
  %calltmp147 = call i32 @getLaneIndex___(<1 x i32> %mask)
  %calltmp151 = call i32 @getLaneIndex___(<1 x i32> %mask)
  %add_xi117_load_ = add i32 %xi117.0307, 32
  %less_xi_load119_xend_load120 = icmp slt i32 %add_xi117_load_, %r.i.i169
  br i1 %less_xi_load119_xend_load120, label %if_exit156, label %for_exit115
}

attributes #0 = { alwaysinline nounwind readnone }
attributes #1 = { nounwind readnone }
attributes #2 = { alwaysinline nounwind }
attributes #3 = { nounwind }
attributes #4 = { alwaysinline nounwind readonly }
attributes #5 = { nounwind "target-features"="+sm_35" }
