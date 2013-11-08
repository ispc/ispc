; ModuleID = 'mandelbrot_task_nvptx64.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64"

@__str = internal constant [66 x i8] c"mandelbrot_task.ispc:55:3: Assertion failed: xspan >= vectorWidth\00"

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.warpsize() #0

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture) #1

; Function Attrs: noreturn
declare void @abort() #2

; Function Attrs: nounwind
define void @mandelbrot_scanline(float %x0, float %dx, float %y0, float %dy, i32 %width, i32 %height, i32 %xspan, i32 %yspan, i32 %maxIterations, i32* %output) #3 {
allocas:
  %bid.i.i = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %mul_calltmp_xspan_load = mul i32 %bid.i.i, %xspan
  %add_xstart_load_xspan_load13 = add i32 %mul_calltmp_xspan_load, %xspan
  %c.i.i = icmp slt i32 %add_xstart_load_xspan_load13, %width
  %r.i.i = select i1 %c.i.i, i32 %add_xstart_load_xspan_load13, i32 %width
  %bid.i.i77 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1
  %mul_calltmp19_yspan_load = mul i32 %bid.i.i77, %yspan
  %add_ystart_load_yspan_load20 = add i32 %mul_calltmp19_yspan_load, %yspan
  %tid.i.i80 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #1
  %greaterequal_xspan_load24_calltmp27 = icmp sgt i32 %tid.i.i80, %xspan
  br i1 %greaterequal_xspan_load24_calltmp27, label %fail.i, label %for_test.preheader

for_test.preheader:                               ; preds = %allocas
  %c.i.i78 = icmp slt i32 %add_ystart_load_yspan_load20, %height
  %r.i.i79 = select i1 %c.i.i78, i32 %add_ystart_load_yspan_load20, i32 %height
  %less_yi_load_yend_load113 = icmp slt i32 %mul_calltmp19_yspan_load, %r.i.i79
  br i1 %less_yi_load_yend_load113, label %for_test34.preheader.lr.ph, label %for_exit

for_test34.preheader.lr.ph:                       ; preds = %for_test.preheader
  %less_xi_load_xend_load111 = icmp slt i32 %mul_calltmp_xspan_load, %r.i.i
  %maxIterations_load_broadcast_init = insertelement <1 x i32> undef, i32 %maxIterations, i32 0
  %less_i_load_count_load.i102 = icmp sgt <1 x i32> %maxIterations_load_broadcast_init, zeroinitializer
  %v.i.i104 = extractelement <1 x i1> %less_i_load_count_load.i102, i32 0
  %output_load_ptr2int = ptrtoint i32* %output to i64
  %0 = xor i32 %height, -1
  %1 = add i32 %bid.i.i77, 1
  %2 = mul i32 %1, %yspan
  %3 = xor i32 %2, -1
  %4 = icmp sgt i32 %0, %3
  %smax = select i1 %4, i32 %0, i32 %3
  %5 = xor i32 %smax, -1
  br label %for_test34.preheader

fail.i:                                           ; preds = %allocas
  %call.i = call i32 @puts(i8* getelementptr inbounds ([66 x i8]* @__str, i64 0, i64 0)) #1
  call void @abort() #4
  unreachable

for_test34.preheader:                             ; preds = %for_exit37, %for_test34.preheader.lr.ph
  %yi.0114 = phi i32 [ %mul_calltmp19_yspan_load, %for_test34.preheader.lr.ph ], [ %yi_load71_plus1, %for_exit37 ]
  br i1 %less_xi_load_xend_load111, label %for_loop36.lr.ph, label %for_exit37

for_loop36.lr.ph:                                 ; preds = %for_test34.preheader
  %yi_load46_to_float = sitofp i32 %yi.0114 to float
  %mul_yi_load46_to_float_dy_load = fmul float %yi_load46_to_float, %dy
  %add_y0_load_mul_yi_load46_to_float_dy_load = fadd float %mul_yi_load46_to_float_dy_load, %y0
  %add_y0_load_mul_yi_load46_to_float_dy_load_broadcast_init = insertelement <1 x float> undef, float %add_y0_load_mul_yi_load46_to_float_dy_load, i32 0
  %mul_yi_load50_width_load51 = mul i32 %yi.0114, %width
  br i1 %v.i.i104, label %for_loop.i.lr.ph.us, label %mandel___vyfvyfvyi.exit

mandel___vyfvyfvyi.exit.us:                       ; preds = %for_step.i.us
  %tid.i.i72.us = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %tid.i.i.i.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #1
  %sub_calltmp3_.i.us = add i32 %tid.i.i.i.us, -1
  %bitop.i.us = and i32 %sub_calltmp3_.i.us, %tid.i.i72.us
  %add_xi_load56_calltmp59.us = add i32 %bitop.i.us, %xi.0112.us
  %less_add_xi_load56_calltmp59_xend_load60.us = icmp slt i32 %add_xi_load56_calltmp59.us, %r.i.i
  br i1 %less_add_xi_load56_calltmp59_xend_load60.us, label %if_then.us, label %if_exit.us

if_then.us:                                       ; preds = %mandel___vyfvyfvyi.exit.us
  %tid.i.i.i74.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #1
  %sub_calltmp3_.i75.us = add i32 %tid.i.i.i74.us, 1073741823
  %tid.i.i73.us = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %bitop.i76.us = and i32 %sub_calltmp3_.i75.us, %tid.i.i73.us
  %add_xi_load52_calltmp55.us = add i32 %xi.0112.us, %mul_yi_load50_width_load51
  %add_mul_yi_load50_width_load51_add_xi_load52_calltmp55.us = add i32 %add_xi_load52_calltmp55.us, %bitop.i76.us
  %6 = shl i32 %add_mul_yi_load50_width_load51_add_xi_load52_calltmp55.us, 2
  %iptr__id.i.rhs.us = sext i32 %6 to i64
  %iptr__id.i.us = add i64 %iptr__id.i.rhs.us, %output_load_ptr2int
  %ptr__id.i.us = inttoptr i64 %iptr__id.i.us to i32*
  %val__id.i.us = extractelement <1 x i32> %v1.i92.us, i32 0
  store i32 %val__id.i.us, i32* %ptr__id.i.us, align 4
  br label %if_exit.us

if_exit.us:                                       ; preds = %if_then.us, %mandel___vyfvyfvyi.exit.us
  %tid.i.i.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #1
  %add_xi_load70_calltmp68.us = add i32 %tid.i.i.us, %xi.0112.us
  %less_xi_load_xend_load.us = icmp slt i32 %add_xi_load70_calltmp68.us, %r.i.i
  br i1 %less_xi_load_xend_load.us, label %for_loop.i.lr.ph.us, label %for_exit37

for_loop.i.us:                                    ; preds = %for_loop.i.lr.ph.us, %for_step.i.us
  %less_i_load_count_load.i110.us = phi <1 x i1> [ %less_i_load_count_load.i102, %for_loop.i.lr.ph.us ], [ %less_i_load_count_load.i.us, %for_step.i.us ]
  %internal_mask_memory.0.i109.us = phi <1 x i1> [ <i1 true>, %for_loop.i.lr.ph.us ], [ %internal_mask_memory.1.i.us, %for_step.i.us ]
  %break_lanes_memory.0.i108.us = phi <1 x i1> [ zeroinitializer, %for_loop.i.lr.ph.us ], [ %"mask|break_mask.i.us", %for_step.i.us ]
  %v1.i9096107.us = phi <1 x float> [ %add_x0_load_mul_add_xi_load42_calltmp45_to_float_dx_load_broadcast_init.us, %for_loop.i.lr.ph.us ], [ %v1.i9095.us, %for_step.i.us ]
  %v1.i8898106.us = phi <1 x float> [ %add_y0_load_mul_yi_load46_to_float_dy_load_broadcast_init, %for_loop.i.lr.ph.us ], [ %v1.i8897.us, %for_step.i.us ]
  %v1.i9299105.us = phi <1 x i32> [ zeroinitializer, %for_loop.i.lr.ph.us ], [ %v1.i92.us, %for_step.i.us ]
  %"oldMask&test.i.us" = and <1 x i1> %internal_mask_memory.0.i109.us, %less_i_load_count_load.i110.us
  %mul_z_re_load_z_re_load13.i.us = fmul <1 x float> %v1.i9096107.us, %v1.i9096107.us
  %mul_z_im_load_z_im_load14.i.us = fmul <1 x float> %v1.i8898106.us, %v1.i8898106.us
  %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i.us = fadd <1 x float> %mul_z_im_load_z_im_load14.i.us, %mul_z_re_load_z_re_load13.i.us
  %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i.us = fcmp ugt <1 x float> %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i.us, <float 4.000000e+00>
  %"oldMask&test16.i.us" = and <1 x i1> %"oldMask&test.i.us", %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i.us
  %"mask|break_mask.i.us" = or <1 x i1> %"oldMask&test16.i.us", %break_lanes_memory.0.i108.us
  %v.i2.i.us = extractelement <1 x i1> %"mask|break_mask.i.us", i32 0
  %v.i1.i.us = extractelement <1 x i1> %"oldMask&test.i.us", i32 0
  %"equal_finished&func_internal_mask&function_mask12.itmp.us" = xor i1 %v.i2.i.us, %v.i1.i.us
  br i1 %"equal_finished&func_internal_mask&function_mask12.itmp.us", label %not_all_continued_or_breaked.i.us, label %for_step.i.us

not_all_continued_or_breaked.i.us:                ; preds = %for_loop.i.us
  %"!(break|continue)_lanes.i.us" = xor <1 x i1> %"mask|break_mask.i.us", <i1 true>
  %new_mask28.i.us = and <1 x i1> %"oldMask&test.i.us", %"!(break|continue)_lanes.i.us"
  %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i.us = fsub <1 x float> %mul_z_re_load_z_re_load13.i.us, %mul_z_im_load_z_im_load14.i.us
  %mul__z_re_load35.i.us = fmul <1 x float> %v1.i9096107.us, <float 2.000000e+00>
  %mul_mul__z_re_load35_z_im_load36.i.us = fmul <1 x float> %v1.i8898106.us, %mul__z_re_load35.i.us
  %add_c_re_load42_new_re_load.i.us = fadd <1 x float> %add_x0_load_mul_add_xi_load42_calltmp45_to_float_dx_load_broadcast_init.us, %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i.us
  %add_c_im_load44_new_im_load.i.us = fadd <1 x float> %add_y0_load_mul_yi_load46_to_float_dy_load_broadcast_init, %mul_mul__z_re_load35_z_im_load36.i.us
  br label %for_step.i.us

for_step.i.us:                                    ; preds = %not_all_continued_or_breaked.i.us, %for_loop.i.us
  %v1.i8897.us = phi <1 x float> [ %v1.i8898106.us, %for_loop.i.us ], [ %add_c_im_load44_new_im_load.i.us, %not_all_continued_or_breaked.i.us ]
  %v1.i9095.us = phi <1 x float> [ %v1.i9096107.us, %for_loop.i.us ], [ %add_c_re_load42_new_re_load.i.us, %not_all_continued_or_breaked.i.us ]
  %internal_mask_memory.1.i.us = phi <1 x i1> [ zeroinitializer, %for_loop.i.us ], [ %new_mask28.i.us, %not_all_continued_or_breaked.i.us ]
  %i_load53_plus1.i.us = add <1 x i32> %v1.i9299105.us, <i32 1>
  %v1.i92.us = select <1 x i1> %internal_mask_memory.1.i.us, <1 x i32> %i_load53_plus1.i.us, <1 x i32> %v1.i9299105.us
  %less_i_load_count_load.i.us = icmp slt <1 x i32> %v1.i92.us, %maxIterations_load_broadcast_init
  %"internal_mask&function_mask10.i.us" = and <1 x i1> %internal_mask_memory.1.i.us, %less_i_load_count_load.i.us
  %v.i.i.us = extractelement <1 x i1> %"internal_mask&function_mask10.i.us", i32 0
  br i1 %v.i.i.us, label %for_loop.i.us, label %mandel___vyfvyfvyi.exit.us

for_loop.i.lr.ph.us:                              ; preds = %if_exit.us, %for_loop36.lr.ph
  %xi.0112.us = phi i32 [ %add_xi_load70_calltmp68.us, %if_exit.us ], [ %mul_calltmp_xspan_load, %for_loop36.lr.ph ]
  %tid.i.i81.us = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %tid.i.i.i82.us = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #1
  %sub_calltmp3_.i83.us = add i32 %tid.i.i.i82.us, -1
  %bitop.i84.us = and i32 %sub_calltmp3_.i83.us, %tid.i.i81.us
  %add_xi_load42_calltmp45.us = add i32 %bitop.i84.us, %xi.0112.us
  %add_xi_load42_calltmp45_to_float.us = sitofp i32 %add_xi_load42_calltmp45.us to float
  %mul_add_xi_load42_calltmp45_to_float_dx_load.us = fmul float %add_xi_load42_calltmp45_to_float.us, %dx
  %add_x0_load_mul_add_xi_load42_calltmp45_to_float_dx_load.us = fadd float %mul_add_xi_load42_calltmp45_to_float_dx_load.us, %x0
  %add_x0_load_mul_add_xi_load42_calltmp45_to_float_dx_load_broadcast_init.us = insertelement <1 x float> undef, float %add_x0_load_mul_add_xi_load42_calltmp45_to_float_dx_load.us, i32 0
  br label %for_loop.i.us

for_exit:                                         ; preds = %for_exit37, %for_test.preheader
  ret void

mandel___vyfvyfvyi.exit:                          ; preds = %if_exit, %for_loop36.lr.ph
  %xi.0112 = phi i32 [ %add_xi_load70_calltmp68, %if_exit ], [ %mul_calltmp_xspan_load, %for_loop36.lr.ph ]
  %tid.i.i72 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %tid.i.i.i = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #1
  %sub_calltmp3_.i = add i32 %tid.i.i.i, -1
  %bitop.i = and i32 %sub_calltmp3_.i, %tid.i.i72
  %add_xi_load56_calltmp59 = add i32 %bitop.i, %xi.0112
  %less_add_xi_load56_calltmp59_xend_load60 = icmp slt i32 %add_xi_load56_calltmp59, %r.i.i
  br i1 %less_add_xi_load56_calltmp59_xend_load60, label %if_then, label %if_exit

for_exit37:                                       ; preds = %if_exit, %if_exit.us, %for_test34.preheader
  %yi_load71_plus1 = add i32 %yi.0114, 1
  %exitcond = icmp eq i32 %yi_load71_plus1, %5
  br i1 %exitcond, label %for_exit, label %for_test34.preheader

if_then:                                          ; preds = %mandel___vyfvyfvyi.exit
  %tid.i.i.i74 = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #1
  %sub_calltmp3_.i75 = add i32 %tid.i.i.i74, 1073741823
  %tid.i.i73 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %bitop.i76 = and i32 %sub_calltmp3_.i75, %tid.i.i73
  %add_xi_load52_calltmp55 = add i32 %xi.0112, %mul_yi_load50_width_load51
  %add_mul_yi_load50_width_load51_add_xi_load52_calltmp55 = add i32 %add_xi_load52_calltmp55, %bitop.i76
  %7 = shl i32 %add_mul_yi_load50_width_load51_add_xi_load52_calltmp55, 2
  %iptr__id.i.rhs = sext i32 %7 to i64
  %iptr__id.i = add i64 %iptr__id.i.rhs, %output_load_ptr2int
  %ptr__id.i = inttoptr i64 %iptr__id.i to i32*
  store i32 0, i32* %ptr__id.i, align 4
  br label %if_exit

if_exit:                                          ; preds = %if_then, %mandel___vyfvyfvyi.exit
  %tid.i.i = call i32 @llvm.nvvm.read.ptx.sreg.warpsize() #1
  %add_xi_load70_calltmp68 = add i32 %tid.i.i, %xi.0112
  %less_xi_load_xend_load = icmp slt i32 %add_xi_load70_calltmp68, %r.i.i
  br i1 %less_xi_load_xend_load, label %mandel___vyfvyfvyi.exit, label %for_exit37
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { noreturn }
attributes #3 = { nounwind "target-features"="+sm_35" }
attributes #4 = { noreturn nounwind }

!nvvm.annotations = !{!0}

!0 = metadata !{void (float, float, float, float, i32, i32, i32, i32, i32, i32*)* @mandelbrot_scanline, metadata !"kernel", i32 1}
