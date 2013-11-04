; ModuleID = 'mandelbrot_task.bc'
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare i8* @ISPCAlloc(i8**, i64, i32)

declare void @ISPCLaunch(i8**, i8*, i8*, i32, i32, i32)

declare void @ISPCSync(i8*)

declare <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32>, <4 x i32>) nounwind readnone

declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>) nounwind readnone

declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>) nounwind readnone

define void @mandelbrot_scanline___unfunfunfunfuniuniuniuniuniun_3C_uni_3E_({ float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
allocas:
  %x01 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 0
  %x02 = load float* %x01, align 4
  %dx3 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 1
  %dx4 = load float* %dx3, align 4
  %y05 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 2
  %y06 = load float* %y05, align 4
  %dy7 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 3
  %dy8 = load float* %dy7, align 4
  %width9 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 4
  %width10 = load i32* %width9, align 4
  %height11 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 5
  %height12 = load i32* %height11, align 4
  %xspan13 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 6
  %xspan14 = load i32* %xspan13, align 4
  %yspan15 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 7
  %yspan16 = load i32* %yspan15, align 4
  %maxIterations17 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 8
  %maxIterations18 = load i32* %maxIterations17, align 4
  %output19 = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 9
  %output20 = load i32** %output19, align 8
  %task_struct_mask = getelementptr { float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }* %0, i64 0, i32 10
  %mask = load <8 x i32>* %task_struct_mask, align 32
  %floatmask.i = bitcast <8 x i32> %mask to <8 x float>
  %v.i = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i)
  %cmp.i = icmp eq i32 %v.i, 255
  %mul_taskIndex0_load_xspan_load = mul i32 %xspan14, %5
  %add_xstart_load_xspan_load23 = add i32 %mul_taskIndex0_load_xspan_load, %xspan14
  %ret_veca.i.i = insertelement <4 x i32> undef, i32 %add_xstart_load_xspan_load23, i32 0
  %ret_vecb.i.i = insertelement <4 x i32> undef, i32 %width10, i32 0
  %ret_val.i.i = call <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32> %ret_veca.i.i, <4 x i32> %ret_vecb.i.i)
  %ret.i.i = extractelement <4 x i32> %ret_val.i.i, i32 0
  %mul_taskIndex1_load_yspan_load = mul i32 %yspan16, %6
  %add_ystart_load_yspan_load26 = add i32 %mul_taskIndex1_load_yspan_load, %yspan16
  %ret_veca.i.i220 = insertelement <4 x i32> undef, i32 %add_ystart_load_yspan_load26, i32 0
  %ret_vecb.i.i221 = insertelement <4 x i32> undef, i32 %height12, i32 0
  %ret_val.i.i222 = call <4 x i32> @llvm.x86.sse41.pminsd(<4 x i32> %ret_veca.i.i220, <4 x i32> %ret_vecb.i.i221)
  %ret.i.i223 = extractelement <4 x i32> %ret_val.i.i222, i32 0
  %less_yi_load_yend_load345 = icmp slt i32 %mul_taskIndex1_load_yspan_load, %ret.i.i223
  br i1 %cmp.i, label %for_test.preheader, label %for_test92.preheader

for_test92.preheader:                             ; preds = %allocas
  br i1 %less_yi_load_yend_load345, label %for_test103.preheader.lr.ph, label %for_exit

for_test103.preheader.lr.ph:                      ; preds = %for_test92.preheader
  %less_xi_load110_xend_load111360 = icmp slt i32 %mul_taskIndex0_load_xspan_load, %ret.i.i
  %x0_load115_broadcast_init = insertelement <8 x float> undef, float %x02, i32 0
  %x0_load115_broadcast = shufflevector <8 x float> %x0_load115_broadcast_init, <8 x float> undef, <8 x i32> zeroinitializer
  %dx_load117_broadcast_init = insertelement <8 x float> undef, float %dx4, i32 0
  %dx_load117_broadcast = shufflevector <8 x float> %dx_load117_broadcast_init, <8 x float> undef, <8 x i32> zeroinitializer
  %maxIterations_load125_broadcast_init = insertelement <8 x i32> undef, i32 %maxIterations18, i32 0
  %maxIterations_load125_broadcast = shufflevector <8 x i32> %maxIterations_load125_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  %less_i_load_count_load.i347 = icmp sgt <8 x i32> %maxIterations_load125_broadcast, zeroinitializer
  %"oldMask&test.i348" = select <8 x i1> %less_i_load_count_load.i347, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer
  %"internal_mask&function_mask10.i349" = and <8 x i32> %"oldMask&test.i348", %mask
  %floatmask.i.i350 = bitcast <8 x i32> %"internal_mask&function_mask10.i349" to <8 x float>
  %xend_load134_broadcast_init = insertelement <8 x i32> undef, i32 %ret.i.i, i32 0
  %xend_load134_broadcast = shufflevector <8 x i32> %xend_load134_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  %output_load145_ptr2int_2void = bitcast i32* %output20 to i8*
  br label %for_test103.preheader

for_test.preheader:                               ; preds = %allocas
  br i1 %less_yi_load_yend_load345, label %for_test34.preheader.lr.ph, label %for_exit

for_test34.preheader.lr.ph:                       ; preds = %for_test.preheader
  %less_xi_load_xend_load343 = icmp slt i32 %mul_taskIndex0_load_xspan_load, %ret.i.i
  %x0_load_broadcast_init = insertelement <8 x float> undef, float %x02, i32 0
  %x0_load_broadcast = shufflevector <8 x float> %x0_load_broadcast_init, <8 x float> undef, <8 x i32> zeroinitializer
  %dx_load_broadcast_init = insertelement <8 x float> undef, float %dx4, i32 0
  %dx_load_broadcast = shufflevector <8 x float> %dx_load_broadcast_init, <8 x float> undef, <8 x i32> zeroinitializer
  %maxIterations_load_broadcast_init = insertelement <8 x i32> undef, i32 %maxIterations18, i32 0
  %maxIterations_load_broadcast = shufflevector <8 x i32> %maxIterations_load_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  %less_i_load_count_load.i181332 = icmp sgt <8 x i32> %maxIterations_load_broadcast, zeroinitializer
  %"oldMask&test.i182333" = select <8 x i1> %less_i_load_count_load.i181332, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer
  %floatmask.i.i183334 = bitcast <8 x i32> %"oldMask&test.i182333" to <8 x float>
  %xend_load51_broadcast_init = insertelement <8 x i32> undef, i32 %ret.i.i, i32 0
  %xend_load51_broadcast = shufflevector <8 x i32> %xend_load51_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  %output_load_ptr2int_2void = bitcast i32* %output20 to i8*
  br label %for_test34.preheader

for_test34.preheader:                             ; preds = %for_exit37, %for_test34.preheader.lr.ph
  %yi.0346 = phi i32 [ %mul_taskIndex1_load_yspan_load, %for_test34.preheader.lr.ph ], [ %yi_load69_plus1, %for_exit37 ]
  br i1 %less_xi_load_xend_load343, label %for_loop36.lr.ph, label %for_exit37

for_loop36.lr.ph:                                 ; preds = %for_test34.preheader
  %yi_load43_to_float = sitofp i32 %yi.0346 to float
  %mul_yi_load43_to_float_dy_load = fmul float %dy8, %yi_load43_to_float
  %add_y0_load_mul_yi_load43_to_float_dy_load = fadd float %y06, %mul_yi_load43_to_float_dy_load
  %add_y0_load_mul_yi_load43_to_float_dy_load_broadcast_init = insertelement <8 x float> undef, float %add_y0_load_mul_yi_load43_to_float_dy_load, i32 0
  %add_y0_load_mul_yi_load43_to_float_dy_load_broadcast = shufflevector <8 x float> %add_y0_load_mul_yi_load43_to_float_dy_load_broadcast_init, <8 x float> undef, <8 x i32> zeroinitializer
  %mul_yi_load47_width_load48 = mul i32 %yi.0346, %width10
  %mul_yi_load47_width_load48_broadcast_init = insertelement <8 x i32> undef, i32 %mul_yi_load47_width_load48, i32 0
  %mul_yi_load47_width_load48_broadcast = shufflevector <8 x i32> %mul_yi_load47_width_load48_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  br label %for_loop36

for_exit:                                         ; preds = %for_exit106, %for_exit37, %for_test.preheader, %for_test92.preheader
  ret void

for_loop36:                                       ; preds = %safe_if_after_true, %for_loop36.lr.ph
  %xi.0344 = phi i32 [ %mul_taskIndex0_load_xspan_load, %for_loop36.lr.ph ], [ %add_xi_load68_, %safe_if_after_true ]
  %xi_load42_broadcast_init = insertelement <8 x i32> undef, i32 %xi.0344, i32 0
  %xi_load42_broadcast = shufflevector <8 x i32> %xi_load42_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  %add_xi_load42_broadcast_ = add <8 x i32> %xi_load42_broadcast, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_xi_load42_broadcast__to_float = sitofp <8 x i32> %add_xi_load42_broadcast_ to <8 x float>
  %mul_add_xi_load42_broadcast__to_float_dx_load_broadcast = fmul <8 x float> %dx_load_broadcast, %add_xi_load42_broadcast__to_float
  %add_x0_load_broadcast_mul_add_xi_load42_broadcast__to_float_dx_load_broadcast = fadd <8 x float> %x0_load_broadcast, %mul_add_xi_load42_broadcast__to_float_dx_load_broadcast
  %v.i.i184335 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i.i183334)
  %cmp.i.i185336 = icmp eq i32 %v.i.i184335, 0
  br i1 %cmp.i.i185336, label %mandel___vyfvyfvyi.exit219, label %for_loop.i207

for_step.i192:                                    ; preds = %not_all_continued_or_breaked.i218, %for_loop.i207
  %z_re.1.i187 = phi <8 x float> [ %z_re.0.i176338, %for_loop.i207 ], [ %add_c_re_load42_new_re_load.i216, %not_all_continued_or_breaked.i218 ]
  %z_im.1.i188 = phi <8 x float> [ %z_im.0.i177339, %for_loop.i207 ], [ %add_c_im_load44_new_im_load.i217, %not_all_continued_or_breaked.i218 ]
  %internal_mask_memory.1.i189 = phi <8 x i32> [ zeroinitializer, %for_loop.i207 ], [ %new_mask28.i210, %not_all_continued_or_breaked.i218 ]
  %i_load53_plus1.i191 = add <8 x i32> %blendAsInt.i328337, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %mask_as_float.i = bitcast <8 x i32> %internal_mask_memory.1.i189 to <8 x float>
  %oldAsFloat.i = bitcast <8 x i32> %blendAsInt.i328337 to <8 x float>
  %newAsFloat.i = bitcast <8 x i32> %i_load53_plus1.i191 to <8 x float>
  %blend.i = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %oldAsFloat.i, <8 x float> %newAsFloat.i, <8 x float> %mask_as_float.i)
  %blendAsInt.i = bitcast <8 x float> %blend.i to <8 x i32>
  %less_i_load_count_load.i181 = icmp slt <8 x i32> %blendAsInt.i, %maxIterations_load_broadcast
  %"oldMask&test.i182" = select <8 x i1> %less_i_load_count_load.i181, <8 x i32> %internal_mask_memory.1.i189, <8 x i32> zeroinitializer
  %floatmask.i.i183 = bitcast <8 x i32> %"oldMask&test.i182" to <8 x float>
  %v.i.i184 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i.i183)
  %cmp.i.i185 = icmp eq i32 %v.i.i184, 0
  br i1 %cmp.i.i185, label %mandel___vyfvyfvyi.exit219, label %for_loop.i207

for_loop.i207:                                    ; preds = %for_step.i192, %for_loop36
  %v.i.i184342 = phi i32 [ %v.i.i184, %for_step.i192 ], [ %v.i.i184335, %for_loop36 ]
  %"oldMask&test.i182341" = phi <8 x i32> [ %"oldMask&test.i182", %for_step.i192 ], [ %"oldMask&test.i182333", %for_loop36 ]
  %break_lanes_memory.0.i178340 = phi <8 x i32> [ %"mask|break_mask.i198", %for_step.i192 ], [ zeroinitializer, %for_loop36 ]
  %z_im.0.i177339 = phi <8 x float> [ %z_im.1.i188, %for_step.i192 ], [ %add_y0_load_mul_yi_load43_to_float_dy_load_broadcast, %for_loop36 ]
  %z_re.0.i176338 = phi <8 x float> [ %z_re.1.i187, %for_step.i192 ], [ %add_x0_load_broadcast_mul_add_xi_load42_broadcast__to_float_dx_load_broadcast, %for_loop36 ]
  %blendAsInt.i328337 = phi <8 x i32> [ %blendAsInt.i, %for_step.i192 ], [ zeroinitializer, %for_loop36 ]
  %mul_z_re_load_z_re_load13.i193 = fmul <8 x float> %z_re.0.i176338, %z_re.0.i176338
  %mul_z_im_load_z_im_load14.i194 = fmul <8 x float> %z_im.0.i177339, %z_im.0.i177339
  %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i195 = fadd <8 x float> %mul_z_re_load_z_re_load13.i193, %mul_z_im_load_z_im_load14.i194
  %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i196 = fcmp ugt <8 x float> %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i195, <float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00>
  %"oldMask&test16.i197" = select <8 x i1> %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i196, <8 x i32> %"oldMask&test.i182341", <8 x i32> zeroinitializer
  %"mask|break_mask.i198" = or <8 x i32> %"oldMask&test16.i197", %break_lanes_memory.0.i178340
  %floatmask.i67.i200 = bitcast <8 x i32> %"mask|break_mask.i198" to <8 x float>
  %v.i68.i201 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i67.i200)
  %"equal_finished&func_internal_mask&function_mask12.i206" = icmp eq i32 %v.i68.i201, %v.i.i184342
  br i1 %"equal_finished&func_internal_mask&function_mask12.i206", label %for_step.i192, label %not_all_continued_or_breaked.i218

not_all_continued_or_breaked.i218:                ; preds = %for_loop.i207
  %"!(break|continue)_lanes.i209" = xor <8 x i32> %"mask|break_mask.i198", <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %new_mask28.i210 = and <8 x i32> %"oldMask&test.i182341", %"!(break|continue)_lanes.i209"
  %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i213 = fsub <8 x float> %mul_z_re_load_z_re_load13.i193, %mul_z_im_load_z_im_load14.i194
  %mul__z_re_load35.i214 = fmul <8 x float> %z_re.0.i176338, <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
  %mul_mul__z_re_load35_z_im_load36.i215 = fmul <8 x float> %mul__z_re_load35.i214, %z_im.0.i177339
  %add_c_re_load42_new_re_load.i216 = fadd <8 x float> %add_x0_load_broadcast_mul_add_xi_load42_broadcast__to_float_dx_load_broadcast, %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i213
  %add_c_im_load44_new_im_load.i217 = fadd <8 x float> %add_y0_load_mul_yi_load43_to_float_dy_load_broadcast, %mul_mul__z_re_load35_z_im_load36.i215
  br label %for_step.i192

mandel___vyfvyfvyi.exit219:                       ; preds = %for_step.i192, %for_loop36
  %blendAsInt.i328.lcssa = phi <8 x i32> [ zeroinitializer, %for_loop36 ], [ %blendAsInt.i, %for_step.i192 ]
  %less_add_xi_load50_broadcast__xend_load51_broadcast = icmp slt <8 x i32> %add_xi_load42_broadcast_, %xend_load51_broadcast
  %floatmask.i172 = select <8 x i1> %less_add_xi_load50_broadcast__xend_load51_broadcast, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, <8 x float> zeroinitializer
  %v.i173 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i172)
  %cmp.i174 = icmp eq i32 %v.i173, 0
  br i1 %cmp.i174, label %safe_if_after_true, label %safe_if_run_true

for_exit37:                                       ; preds = %safe_if_after_true, %for_test34.preheader
  %yi_load69_plus1 = add i32 %yi.0346, 1
  %exitcond = icmp eq i32 %yi_load69_plus1, %ret.i.i223
  br i1 %exitcond, label %for_exit, label %for_test34.preheader

safe_if_after_true:                               ; preds = %pl_dolane.7.i326, %pl_loopend.6.i318, %mandel___vyfvyfvyi.exit219
  %add_xi_load68_ = add i32 %xi.0344, 8
  %less_xi_load_xend_load = icmp slt i32 %add_xi_load68_, %ret.i.i
  br i1 %less_xi_load_xend_load, label %for_loop36, label %for_exit37

safe_if_run_true:                                 ; preds = %mandel___vyfvyfvyi.exit219
  %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast = add <8 x i32> %mul_yi_load47_width_load48_broadcast, %xi_load42_broadcast
  %v.i.i239 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i172)
  %v64.i.i240 = zext i32 %v.i.i239 to i64
  %pl_and.i241 = and i64 %v64.i.i240, 1
  %pl_doit.i242 = icmp eq i64 %pl_and.i241, 0
  br i1 %pl_doit.i242, label %pl_loopend.i252, label %pl_dolane.i249

pl_dolane.i249:                                   ; preds = %safe_if_run_true
  %offset32.i.i243 = extractelement <8 x i32> %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast, i32 0
  %offset64.i.i244 = sext i32 %offset32.i.i243 to i64
  %finalptr.i.i246331 = getelementptr i32* %output20, i64 %offset64.i.i244
  %storeval.i.i248 = extractelement <8 x i32> %blendAsInt.i328.lcssa, i32 0
  store i32 %storeval.i.i248, i32* %finalptr.i.i246331, align 4
  br label %pl_loopend.i252

pl_loopend.i252:                                  ; preds = %pl_dolane.i249, %safe_if_run_true
  %pl_and.1.i250 = and i64 %v64.i.i240, 2
  %pl_doit.1.i251 = icmp eq i64 %pl_and.1.i250, 0
  br i1 %pl_doit.1.i251, label %pl_loopend.1.i263, label %pl_dolane.1.i260

pl_dolane.1.i260:                                 ; preds = %pl_loopend.i252
  %offset32.i.1.i253 = extractelement <8 x i32> %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast, i32 1
  %offset64.i.1.i254 = sext i32 %offset32.i.1.i253 to i64
  %offset.i.1.i255 = shl nsw i64 %offset64.i.1.i254, 2
  %ptroffset.sum.i.1.i256 = add i64 %offset.i.1.i255, 8
  %finalptr.i.1.i257 = getelementptr i8* %output_load_ptr2int_2void, i64 %ptroffset.sum.i.1.i256
  %ptrcast.i.1.i258 = bitcast i8* %finalptr.i.1.i257 to i32*
  %storeval.i.1.i259 = extractelement <8 x i32> %blendAsInt.i328.lcssa, i32 1
  store i32 %storeval.i.1.i259, i32* %ptrcast.i.1.i258, align 4
  br label %pl_loopend.1.i263

pl_loopend.1.i263:                                ; preds = %pl_dolane.1.i260, %pl_loopend.i252
  %pl_and.2.i261 = and i64 %v64.i.i240, 4
  %pl_doit.2.i262 = icmp eq i64 %pl_and.2.i261, 0
  br i1 %pl_doit.2.i262, label %pl_loopend.2.i274, label %pl_dolane.2.i271

pl_dolane.2.i271:                                 ; preds = %pl_loopend.1.i263
  %offset32.i.2.i264 = extractelement <8 x i32> %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast, i32 2
  %offset64.i.2.i265 = sext i32 %offset32.i.2.i264 to i64
  %offset.i.2.i266 = shl nsw i64 %offset64.i.2.i265, 2
  %ptroffset.sum.i.2.i267 = add i64 %offset.i.2.i266, 16
  %finalptr.i.2.i268 = getelementptr i8* %output_load_ptr2int_2void, i64 %ptroffset.sum.i.2.i267
  %ptrcast.i.2.i269 = bitcast i8* %finalptr.i.2.i268 to i32*
  %storeval.i.2.i270 = extractelement <8 x i32> %blendAsInt.i328.lcssa, i32 2
  store i32 %storeval.i.2.i270, i32* %ptrcast.i.2.i269, align 4
  br label %pl_loopend.2.i274

pl_loopend.2.i274:                                ; preds = %pl_dolane.2.i271, %pl_loopend.1.i263
  %pl_and.3.i272 = and i64 %v64.i.i240, 8
  %pl_doit.3.i273 = icmp eq i64 %pl_and.3.i272, 0
  br i1 %pl_doit.3.i273, label %pl_loopend.3.i285, label %pl_dolane.3.i282

pl_dolane.3.i282:                                 ; preds = %pl_loopend.2.i274
  %offset32.i.3.i275 = extractelement <8 x i32> %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast, i32 3
  %offset64.i.3.i276 = sext i32 %offset32.i.3.i275 to i64
  %offset.i.3.i277 = shl nsw i64 %offset64.i.3.i276, 2
  %ptroffset.sum.i.3.i278 = add i64 %offset.i.3.i277, 24
  %finalptr.i.3.i279 = getelementptr i8* %output_load_ptr2int_2void, i64 %ptroffset.sum.i.3.i278
  %ptrcast.i.3.i280 = bitcast i8* %finalptr.i.3.i279 to i32*
  %storeval.i.3.i281 = extractelement <8 x i32> %blendAsInt.i328.lcssa, i32 3
  store i32 %storeval.i.3.i281, i32* %ptrcast.i.3.i280, align 4
  br label %pl_loopend.3.i285

pl_loopend.3.i285:                                ; preds = %pl_dolane.3.i282, %pl_loopend.2.i274
  %pl_and.4.i283 = and i64 %v64.i.i240, 16
  %pl_doit.4.i284 = icmp eq i64 %pl_and.4.i283, 0
  br i1 %pl_doit.4.i284, label %pl_loopend.4.i296, label %pl_dolane.4.i293

pl_dolane.4.i293:                                 ; preds = %pl_loopend.3.i285
  %offset32.i.4.i286 = extractelement <8 x i32> %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast, i32 4
  %offset64.i.4.i287 = sext i32 %offset32.i.4.i286 to i64
  %offset.i.4.i288 = shl nsw i64 %offset64.i.4.i287, 2
  %ptroffset.sum.i.4.i289 = add i64 %offset.i.4.i288, 32
  %finalptr.i.4.i290 = getelementptr i8* %output_load_ptr2int_2void, i64 %ptroffset.sum.i.4.i289
  %ptrcast.i.4.i291 = bitcast i8* %finalptr.i.4.i290 to i32*
  %storeval.i.4.i292 = extractelement <8 x i32> %blendAsInt.i328.lcssa, i32 4
  store i32 %storeval.i.4.i292, i32* %ptrcast.i.4.i291, align 4
  br label %pl_loopend.4.i296

pl_loopend.4.i296:                                ; preds = %pl_dolane.4.i293, %pl_loopend.3.i285
  %pl_and.5.i294 = and i64 %v64.i.i240, 32
  %pl_doit.5.i295 = icmp eq i64 %pl_and.5.i294, 0
  br i1 %pl_doit.5.i295, label %pl_loopend.5.i307, label %pl_dolane.5.i304

pl_dolane.5.i304:                                 ; preds = %pl_loopend.4.i296
  %offset32.i.5.i297 = extractelement <8 x i32> %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast, i32 5
  %offset64.i.5.i298 = sext i32 %offset32.i.5.i297 to i64
  %offset.i.5.i299 = shl nsw i64 %offset64.i.5.i298, 2
  %ptroffset.sum.i.5.i300 = add i64 %offset.i.5.i299, 40
  %finalptr.i.5.i301 = getelementptr i8* %output_load_ptr2int_2void, i64 %ptroffset.sum.i.5.i300
  %ptrcast.i.5.i302 = bitcast i8* %finalptr.i.5.i301 to i32*
  %storeval.i.5.i303 = extractelement <8 x i32> %blendAsInt.i328.lcssa, i32 5
  store i32 %storeval.i.5.i303, i32* %ptrcast.i.5.i302, align 4
  br label %pl_loopend.5.i307

pl_loopend.5.i307:                                ; preds = %pl_dolane.5.i304, %pl_loopend.4.i296
  %pl_and.6.i305 = and i64 %v64.i.i240, 64
  %pl_doit.6.i306 = icmp eq i64 %pl_and.6.i305, 0
  br i1 %pl_doit.6.i306, label %pl_loopend.6.i318, label %pl_dolane.6.i315

pl_dolane.6.i315:                                 ; preds = %pl_loopend.5.i307
  %offset32.i.6.i308 = extractelement <8 x i32> %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast, i32 6
  %offset64.i.6.i309 = sext i32 %offset32.i.6.i308 to i64
  %offset.i.6.i310 = shl nsw i64 %offset64.i.6.i309, 2
  %ptroffset.sum.i.6.i311 = add i64 %offset.i.6.i310, 48
  %finalptr.i.6.i312 = getelementptr i8* %output_load_ptr2int_2void, i64 %ptroffset.sum.i.6.i311
  %ptrcast.i.6.i313 = bitcast i8* %finalptr.i.6.i312 to i32*
  %storeval.i.6.i314 = extractelement <8 x i32> %blendAsInt.i328.lcssa, i32 6
  store i32 %storeval.i.6.i314, i32* %ptrcast.i.6.i313, align 4
  br label %pl_loopend.6.i318

pl_loopend.6.i318:                                ; preds = %pl_dolane.6.i315, %pl_loopend.5.i307
  %pl_and.7.i316 = and i64 %v64.i.i240, 128
  %pl_doit.7.i317 = icmp eq i64 %pl_and.7.i316, 0
  br i1 %pl_doit.7.i317, label %safe_if_after_true, label %pl_dolane.7.i326

pl_dolane.7.i326:                                 ; preds = %pl_loopend.6.i318
  %offset32.i.7.i319 = extractelement <8 x i32> %add_mul_yi_load47_width_load48_broadcast_xi_load49_broadcast, i32 7
  %offset64.i.7.i320 = sext i32 %offset32.i.7.i319 to i64
  %offset.i.7.i321 = shl nsw i64 %offset64.i.7.i320, 2
  %ptroffset.sum.i.7.i322 = add i64 %offset.i.7.i321, 56
  %finalptr.i.7.i323 = getelementptr i8* %output_load_ptr2int_2void, i64 %ptroffset.sum.i.7.i322
  %ptrcast.i.7.i324 = bitcast i8* %finalptr.i.7.i323 to i32*
  %storeval.i.7.i325 = extractelement <8 x i32> %blendAsInt.i328.lcssa, i32 7
  store i32 %storeval.i.7.i325, i32* %ptrcast.i.7.i324, align 4
  br label %safe_if_after_true

for_test103.preheader:                            ; preds = %for_exit106, %for_test103.preheader.lr.ph
  %yi97.0364 = phi i32 [ %mul_taskIndex1_load_yspan_load, %for_test103.preheader.lr.ph ], [ %yi_load164_plus1, %for_exit106 ]
  br i1 %less_xi_load110_xend_load111360, label %for_loop105.lr.ph, label %for_exit106

for_loop105.lr.ph:                                ; preds = %for_test103.preheader
  %yi_load120_to_float = sitofp i32 %yi97.0364 to float
  %mul_yi_load120_to_float_dy_load121 = fmul float %dy8, %yi_load120_to_float
  %add_y0_load119_mul_yi_load120_to_float_dy_load121 = fadd float %y06, %mul_yi_load120_to_float_dy_load121
  %add_y0_load119_mul_yi_load120_to_float_dy_load121_broadcast_init = insertelement <8 x float> undef, float %add_y0_load119_mul_yi_load120_to_float_dy_load121, i32 0
  %add_y0_load119_mul_yi_load120_to_float_dy_load121_broadcast = shufflevector <8 x float> %add_y0_load119_mul_yi_load120_to_float_dy_load121_broadcast_init, <8 x float> undef, <8 x i32> zeroinitializer
  %mul_yi_load130_width_load131 = mul i32 %yi97.0364, %width10
  %mul_yi_load130_width_load131_broadcast_init = insertelement <8 x i32> undef, i32 %mul_yi_load130_width_load131, i32 0
  %mul_yi_load130_width_load131_broadcast = shufflevector <8 x i32> %mul_yi_load130_width_load131_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  br label %for_loop105

for_loop105:                                      ; preds = %safe_if_after_true137, %for_loop105.lr.ph
  %xi108.0361 = phi i32 [ %mul_taskIndex0_load_xspan_load, %for_loop105.lr.ph ], [ %add_xi108_load_, %safe_if_after_true137 ]
  %xi_load116_broadcast_init = insertelement <8 x i32> undef, i32 %xi108.0361, i32 0
  %xi_load116_broadcast = shufflevector <8 x i32> %xi_load116_broadcast_init, <8 x i32> undef, <8 x i32> zeroinitializer
  %add_xi_load116_broadcast_ = add <8 x i32> %xi_load116_broadcast, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %add_xi_load116_broadcast__to_float = sitofp <8 x i32> %add_xi_load116_broadcast_ to <8 x float>
  %mul_add_xi_load116_broadcast__to_float_dx_load117_broadcast = fmul <8 x float> %dx_load117_broadcast, %add_xi_load116_broadcast__to_float
  %add_x0_load115_broadcast_mul_add_xi_load116_broadcast__to_float_dx_load117_broadcast = fadd <8 x float> %x0_load115_broadcast, %mul_add_xi_load116_broadcast__to_float_dx_load117_broadcast
  %v.i.i351 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i.i350)
  %cmp.i.i352 = icmp eq i32 %v.i.i351, 0
  br i1 %cmp.i.i352, label %mandel___vyfvyfvyi.exit, label %for_loop.i

for_step.i:                                       ; preds = %not_all_continued_or_breaked.i, %for_loop.i
  %z_re.1.i = phi <8 x float> [ %z_re.0.i354, %for_loop.i ], [ %add_c_re_load42_new_re_load.i, %not_all_continued_or_breaked.i ]
  %z_im.1.i = phi <8 x float> [ %z_im.0.i355, %for_loop.i ], [ %add_c_im_load44_new_im_load.i, %not_all_continued_or_breaked.i ]
  %internal_mask_memory.1.i = phi <8 x i32> [ zeroinitializer, %for_loop.i ], [ %new_mask28.i, %not_all_continued_or_breaked.i ]
  %i_load53_plus1.i = add <8 x i32> %blendAsInt.i237329353, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %mask_as_float.i232 = bitcast <8 x i32> %internal_mask_memory.1.i to <8 x float>
  %oldAsFloat.i234 = bitcast <8 x i32> %blendAsInt.i237329353 to <8 x float>
  %newAsFloat.i235 = bitcast <8 x i32> %i_load53_plus1.i to <8 x float>
  %blend.i236 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %oldAsFloat.i234, <8 x float> %newAsFloat.i235, <8 x float> %mask_as_float.i232)
  %blendAsInt.i237 = bitcast <8 x float> %blend.i236 to <8 x i32>
  %less_i_load_count_load.i = icmp slt <8 x i32> %blendAsInt.i237, %maxIterations_load125_broadcast
  %"oldMask&test.i" = select <8 x i1> %less_i_load_count_load.i, <8 x i32> %internal_mask_memory.1.i, <8 x i32> zeroinitializer
  %"internal_mask&function_mask10.i" = and <8 x i32> %"oldMask&test.i", %mask
  %floatmask.i.i = bitcast <8 x i32> %"internal_mask&function_mask10.i" to <8 x float>
  %v.i.i = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i.i)
  %cmp.i.i = icmp eq i32 %v.i.i, 0
  br i1 %cmp.i.i, label %mandel___vyfvyfvyi.exit, label %for_loop.i

for_loop.i:                                       ; preds = %for_step.i, %for_loop105
  %v.i.i358 = phi i32 [ %v.i.i, %for_step.i ], [ %v.i.i351, %for_loop105 ]
  %"oldMask&test.i357" = phi <8 x i32> [ %"oldMask&test.i", %for_step.i ], [ %"oldMask&test.i348", %for_loop105 ]
  %break_lanes_memory.0.i356 = phi <8 x i32> [ %"mask|break_mask.i", %for_step.i ], [ zeroinitializer, %for_loop105 ]
  %z_im.0.i355 = phi <8 x float> [ %z_im.1.i, %for_step.i ], [ %add_y0_load119_mul_yi_load120_to_float_dy_load121_broadcast, %for_loop105 ]
  %z_re.0.i354 = phi <8 x float> [ %z_re.1.i, %for_step.i ], [ %add_x0_load115_broadcast_mul_add_xi_load116_broadcast__to_float_dx_load117_broadcast, %for_loop105 ]
  %blendAsInt.i237329353 = phi <8 x i32> [ %blendAsInt.i237, %for_step.i ], [ zeroinitializer, %for_loop105 ]
  %mul_z_re_load_z_re_load13.i = fmul <8 x float> %z_re.0.i354, %z_re.0.i354
  %mul_z_im_load_z_im_load14.i = fmul <8 x float> %z_im.0.i355, %z_im.0.i355
  %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i = fadd <8 x float> %mul_z_re_load_z_re_load13.i, %mul_z_im_load_z_im_load14.i
  %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i = fcmp ugt <8 x float> %add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14.i, <float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00>
  %"oldMask&test16.i" = select <8 x i1> %greater_add_mul_z_re_load_z_re_load13_mul_z_im_load_z_im_load14_.i, <8 x i32> %"oldMask&test.i357", <8 x i32> zeroinitializer
  %"mask|break_mask.i" = or <8 x i32> %"oldMask&test16.i", %break_lanes_memory.0.i356
  %"finished&func.i" = and <8 x i32> %"mask|break_mask.i", %mask
  %floatmask.i67.i = bitcast <8 x i32> %"finished&func.i" to <8 x float>
  %v.i68.i = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i67.i)
  %"equal_finished&func_internal_mask&function_mask12.i" = icmp eq i32 %v.i68.i, %v.i.i358
  br i1 %"equal_finished&func_internal_mask&function_mask12.i", label %for_step.i, label %not_all_continued_or_breaked.i

not_all_continued_or_breaked.i:                   ; preds = %for_loop.i
  %"!(break|continue)_lanes.i" = xor <8 x i32> %"mask|break_mask.i", <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %new_mask28.i = and <8 x i32> %"oldMask&test.i357", %"!(break|continue)_lanes.i"
  %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i = fsub <8 x float> %mul_z_re_load_z_re_load13.i, %mul_z_im_load_z_im_load14.i
  %mul__z_re_load35.i = fmul <8 x float> %z_re.0.i354, <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>
  %mul_mul__z_re_load35_z_im_load36.i = fmul <8 x float> %mul__z_re_load35.i, %z_im.0.i355
  %add_c_re_load42_new_re_load.i = fadd <8 x float> %add_x0_load115_broadcast_mul_add_xi_load116_broadcast__to_float_dx_load117_broadcast, %sub_mul_z_re_load31_z_re_load32_mul_z_im_load33_z_im_load34.i
  %add_c_im_load44_new_im_load.i = fadd <8 x float> %add_y0_load119_mul_yi_load120_to_float_dy_load121_broadcast, %mul_mul__z_re_load35_z_im_load36.i
  br label %for_step.i

mandel___vyfvyfvyi.exit:                          ; preds = %for_step.i, %for_loop105
  %blendAsInt.i237329.lcssa = phi <8 x i32> [ zeroinitializer, %for_loop105 ], [ %blendAsInt.i237, %for_step.i ]
  %less_add_xi_load133_broadcast__xend_load134_broadcast = icmp slt <8 x i32> %add_xi_load116_broadcast_, %xend_load134_broadcast
  %"oldMask&test139" = select <8 x i1> %less_add_xi_load133_broadcast__xend_load134_broadcast, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer
  %"internal_mask&function_mask143" = and <8 x i32> %"oldMask&test139", %mask
  %floatmask.i169 = bitcast <8 x i32> %"internal_mask&function_mask143" to <8 x float>
  %v.i170 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i169)
  %cmp.i171 = icmp eq i32 %v.i170, 0
  br i1 %cmp.i171, label %safe_if_after_true137, label %safe_if_run_true138

for_exit106:                                      ; preds = %safe_if_after_true137, %for_test103.preheader
  %yi_load164_plus1 = add i32 %yi97.0364, 1
  %exitcond365 = icmp eq i32 %yi_load164_plus1, %ret.i.i223
  br i1 %exitcond365, label %for_exit, label %for_test103.preheader

safe_if_after_true137:                            ; preds = %pl_dolane.7.i, %pl_loopend.6.i, %mandel___vyfvyfvyi.exit
  %add_xi108_load_ = add i32 %xi108.0361, 8
  %less_xi_load110_xend_load111 = icmp slt i32 %add_xi108_load_, %ret.i.i
  br i1 %less_xi_load110_xend_load111, label %for_loop105, label %for_exit106

safe_if_run_true138:                              ; preds = %mandel___vyfvyfvyi.exit
  %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast = add <8 x i32> %mul_yi_load130_width_load131_broadcast, %xi_load116_broadcast
  %v.i.i231 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i169)
  %v64.i.i = zext i32 %v.i.i231 to i64
  %pl_and.i = and i64 %v64.i.i, 1
  %pl_doit.i = icmp eq i64 %pl_and.i, 0
  br i1 %pl_doit.i, label %pl_loopend.i, label %pl_dolane.i

pl_dolane.i:                                      ; preds = %safe_if_run_true138
  %offset32.i.i = extractelement <8 x i32> %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast, i32 0
  %offset64.i.i = sext i32 %offset32.i.i to i64
  %finalptr.i.i330 = getelementptr i32* %output20, i64 %offset64.i.i
  %storeval.i.i = extractelement <8 x i32> %blendAsInt.i237329.lcssa, i32 0
  store i32 %storeval.i.i, i32* %finalptr.i.i330, align 4
  br label %pl_loopend.i

pl_loopend.i:                                     ; preds = %pl_dolane.i, %safe_if_run_true138
  %pl_and.1.i = and i64 %v64.i.i, 2
  %pl_doit.1.i = icmp eq i64 %pl_and.1.i, 0
  br i1 %pl_doit.1.i, label %pl_loopend.1.i, label %pl_dolane.1.i

pl_dolane.1.i:                                    ; preds = %pl_loopend.i
  %offset32.i.1.i = extractelement <8 x i32> %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast, i32 1
  %offset64.i.1.i = sext i32 %offset32.i.1.i to i64
  %offset.i.1.i = shl nsw i64 %offset64.i.1.i, 2
  %ptroffset.sum.i.1.i = add i64 %offset.i.1.i, 8
  %finalptr.i.1.i = getelementptr i8* %output_load145_ptr2int_2void, i64 %ptroffset.sum.i.1.i
  %ptrcast.i.1.i = bitcast i8* %finalptr.i.1.i to i32*
  %storeval.i.1.i = extractelement <8 x i32> %blendAsInt.i237329.lcssa, i32 1
  store i32 %storeval.i.1.i, i32* %ptrcast.i.1.i, align 4
  br label %pl_loopend.1.i

pl_loopend.1.i:                                   ; preds = %pl_dolane.1.i, %pl_loopend.i
  %pl_and.2.i = and i64 %v64.i.i, 4
  %pl_doit.2.i = icmp eq i64 %pl_and.2.i, 0
  br i1 %pl_doit.2.i, label %pl_loopend.2.i, label %pl_dolane.2.i

pl_dolane.2.i:                                    ; preds = %pl_loopend.1.i
  %offset32.i.2.i = extractelement <8 x i32> %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast, i32 2
  %offset64.i.2.i = sext i32 %offset32.i.2.i to i64
  %offset.i.2.i = shl nsw i64 %offset64.i.2.i, 2
  %ptroffset.sum.i.2.i = add i64 %offset.i.2.i, 16
  %finalptr.i.2.i = getelementptr i8* %output_load145_ptr2int_2void, i64 %ptroffset.sum.i.2.i
  %ptrcast.i.2.i = bitcast i8* %finalptr.i.2.i to i32*
  %storeval.i.2.i = extractelement <8 x i32> %blendAsInt.i237329.lcssa, i32 2
  store i32 %storeval.i.2.i, i32* %ptrcast.i.2.i, align 4
  br label %pl_loopend.2.i

pl_loopend.2.i:                                   ; preds = %pl_dolane.2.i, %pl_loopend.1.i
  %pl_and.3.i = and i64 %v64.i.i, 8
  %pl_doit.3.i = icmp eq i64 %pl_and.3.i, 0
  br i1 %pl_doit.3.i, label %pl_loopend.3.i, label %pl_dolane.3.i

pl_dolane.3.i:                                    ; preds = %pl_loopend.2.i
  %offset32.i.3.i = extractelement <8 x i32> %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast, i32 3
  %offset64.i.3.i = sext i32 %offset32.i.3.i to i64
  %offset.i.3.i = shl nsw i64 %offset64.i.3.i, 2
  %ptroffset.sum.i.3.i = add i64 %offset.i.3.i, 24
  %finalptr.i.3.i = getelementptr i8* %output_load145_ptr2int_2void, i64 %ptroffset.sum.i.3.i
  %ptrcast.i.3.i = bitcast i8* %finalptr.i.3.i to i32*
  %storeval.i.3.i = extractelement <8 x i32> %blendAsInt.i237329.lcssa, i32 3
  store i32 %storeval.i.3.i, i32* %ptrcast.i.3.i, align 4
  br label %pl_loopend.3.i

pl_loopend.3.i:                                   ; preds = %pl_dolane.3.i, %pl_loopend.2.i
  %pl_and.4.i = and i64 %v64.i.i, 16
  %pl_doit.4.i = icmp eq i64 %pl_and.4.i, 0
  br i1 %pl_doit.4.i, label %pl_loopend.4.i, label %pl_dolane.4.i

pl_dolane.4.i:                                    ; preds = %pl_loopend.3.i
  %offset32.i.4.i = extractelement <8 x i32> %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast, i32 4
  %offset64.i.4.i = sext i32 %offset32.i.4.i to i64
  %offset.i.4.i = shl nsw i64 %offset64.i.4.i, 2
  %ptroffset.sum.i.4.i = add i64 %offset.i.4.i, 32
  %finalptr.i.4.i = getelementptr i8* %output_load145_ptr2int_2void, i64 %ptroffset.sum.i.4.i
  %ptrcast.i.4.i = bitcast i8* %finalptr.i.4.i to i32*
  %storeval.i.4.i = extractelement <8 x i32> %blendAsInt.i237329.lcssa, i32 4
  store i32 %storeval.i.4.i, i32* %ptrcast.i.4.i, align 4
  br label %pl_loopend.4.i

pl_loopend.4.i:                                   ; preds = %pl_dolane.4.i, %pl_loopend.3.i
  %pl_and.5.i = and i64 %v64.i.i, 32
  %pl_doit.5.i = icmp eq i64 %pl_and.5.i, 0
  br i1 %pl_doit.5.i, label %pl_loopend.5.i, label %pl_dolane.5.i

pl_dolane.5.i:                                    ; preds = %pl_loopend.4.i
  %offset32.i.5.i = extractelement <8 x i32> %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast, i32 5
  %offset64.i.5.i = sext i32 %offset32.i.5.i to i64
  %offset.i.5.i = shl nsw i64 %offset64.i.5.i, 2
  %ptroffset.sum.i.5.i = add i64 %offset.i.5.i, 40
  %finalptr.i.5.i = getelementptr i8* %output_load145_ptr2int_2void, i64 %ptroffset.sum.i.5.i
  %ptrcast.i.5.i = bitcast i8* %finalptr.i.5.i to i32*
  %storeval.i.5.i = extractelement <8 x i32> %blendAsInt.i237329.lcssa, i32 5
  store i32 %storeval.i.5.i, i32* %ptrcast.i.5.i, align 4
  br label %pl_loopend.5.i

pl_loopend.5.i:                                   ; preds = %pl_dolane.5.i, %pl_loopend.4.i
  %pl_and.6.i = and i64 %v64.i.i, 64
  %pl_doit.6.i = icmp eq i64 %pl_and.6.i, 0
  br i1 %pl_doit.6.i, label %pl_loopend.6.i, label %pl_dolane.6.i

pl_dolane.6.i:                                    ; preds = %pl_loopend.5.i
  %offset32.i.6.i = extractelement <8 x i32> %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast, i32 6
  %offset64.i.6.i = sext i32 %offset32.i.6.i to i64
  %offset.i.6.i = shl nsw i64 %offset64.i.6.i, 2
  %ptroffset.sum.i.6.i = add i64 %offset.i.6.i, 48
  %finalptr.i.6.i = getelementptr i8* %output_load145_ptr2int_2void, i64 %ptroffset.sum.i.6.i
  %ptrcast.i.6.i = bitcast i8* %finalptr.i.6.i to i32*
  %storeval.i.6.i = extractelement <8 x i32> %blendAsInt.i237329.lcssa, i32 6
  store i32 %storeval.i.6.i, i32* %ptrcast.i.6.i, align 4
  br label %pl_loopend.6.i

pl_loopend.6.i:                                   ; preds = %pl_dolane.6.i, %pl_loopend.5.i
  %pl_and.7.i = and i64 %v64.i.i, 128
  %pl_doit.7.i = icmp eq i64 %pl_and.7.i, 0
  br i1 %pl_doit.7.i, label %safe_if_after_true137, label %pl_dolane.7.i

pl_dolane.7.i:                                    ; preds = %pl_loopend.6.i
  %offset32.i.7.i = extractelement <8 x i32> %add_mul_yi_load130_width_load131_broadcast_xi_load132_broadcast, i32 7
  %offset64.i.7.i = sext i32 %offset32.i.7.i to i64
  %offset.i.7.i = shl nsw i64 %offset64.i.7.i, 2
  %ptroffset.sum.i.7.i = add i64 %offset.i.7.i, 56
  %finalptr.i.7.i = getelementptr i8* %output_load145_ptr2int_2void, i64 %ptroffset.sum.i.7.i
  %ptrcast.i.7.i = bitcast i8* %finalptr.i.7.i to i32*
  %storeval.i.7.i = extractelement <8 x i32> %blendAsInt.i237329.lcssa, i32 7
  store i32 %storeval.i.7.i, i32* %ptrcast.i.7.i, align 4
  br label %safe_if_after_true137
}

define void @mandelbrot_ispc___unfunfunfunfuniuniuniun_3C_uni_3E_(float %x0, float %y0, float %x1, float %y1, i32 %width, i32 %height, i32 %maxIterations, i32* %output, <8 x i32> %__mask) {
allocas:
  %launch_group_handle = alloca i8*, align 8
  store i8* null, i8** %launch_group_handle, align 8
  %floatmask.i = bitcast <8 x i32> %__mask to <8 x float>
  %v.i = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %floatmask.i)
  %cmp.i = icmp eq i32 %v.i, 255
  %sub_x1_load_x0_load = fsub float %x1, %x0
  %width_load_to_float = sitofp i32 %width to float
  %div_sub_x1_load_x0_load_width_load_to_float = fdiv float %sub_x1_load_x0_load, %width_load_to_float
  %sub_y1_load_y0_load = fsub float %y1, %y0
  %height_load_to_float = sitofp i32 %height to float
  %div_sub_y1_load_y0_load_height_load_to_float = fdiv float %sub_y1_load_y0_load, %height_load_to_float
  %div_width_load15_ = sdiv i32 %width, 16
  %div_height_load16_yspan_load = sdiv i32 %height, 16
  %args_ptr = call i8* @ISPCAlloc(i8** %launch_group_handle, i64 96, i32 32)
  %funarg = bitcast i8* %args_ptr to float*
  store float %x0, float* %funarg, align 4
  %funarg17 = getelementptr i8* %args_ptr, i64 4
  %0 = bitcast i8* %funarg17 to float*
  store float %div_sub_x1_load_x0_load_width_load_to_float, float* %0, align 4
  %funarg18 = getelementptr i8* %args_ptr, i64 8
  %1 = bitcast i8* %funarg18 to float*
  store float %y0, float* %1, align 4
  %funarg19 = getelementptr i8* %args_ptr, i64 12
  %2 = bitcast i8* %funarg19 to float*
  store float %div_sub_y1_load_y0_load_height_load_to_float, float* %2, align 4
  %funarg20 = getelementptr i8* %args_ptr, i64 16
  %3 = bitcast i8* %funarg20 to i32*
  store i32 %width, i32* %3, align 4
  %funarg21 = getelementptr i8* %args_ptr, i64 20
  %4 = bitcast i8* %funarg21 to i32*
  store i32 %height, i32* %4, align 4
  %funarg22 = getelementptr i8* %args_ptr, i64 24
  %5 = bitcast i8* %funarg22 to i32*
  store i32 16, i32* %5, align 4
  %funarg23 = getelementptr i8* %args_ptr, i64 28
  %6 = bitcast i8* %funarg23 to i32*
  store i32 16, i32* %6, align 4
  %funarg24 = getelementptr i8* %args_ptr, i64 32
  %7 = bitcast i8* %funarg24 to i32*
  store i32 %maxIterations, i32* %7, align 4
  %funarg25 = getelementptr i8* %args_ptr, i64 40
  %8 = bitcast i8* %funarg25 to i32**
  store i32* %output, i32** %8, align 8
  %funarg_mask = getelementptr i8* %args_ptr, i64 64
  %9 = bitcast i8* %funarg_mask to <8 x i32>*
  br i1 %cmp.i, label %all_on, label %some_on

all_on:                                           ; preds = %allocas
  store <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32>* %9, align 32
  call void @ISPCLaunch(i8** %launch_group_handle, i8* bitcast (void ({ float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @mandelbrot_scanline___unfunfunfunfuniuniuniuniuniun_3C_uni_3E_ to i8*), i8* %args_ptr, i32 %div_width_load15_, i32 %div_height_load16_yspan_load, i32 1)
  %launch_group_handle_load = load i8** %launch_group_handle, align 8
  %cmp = icmp eq i8* %launch_group_handle_load, null
  br i1 %cmp, label %post_sync, label %call_sync

some_on:                                          ; preds = %allocas
  store <8 x i32> %__mask, <8 x i32>* %9, align 32
  call void @ISPCLaunch(i8** %launch_group_handle, i8* bitcast (void ({ float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @mandelbrot_scanline___unfunfunfunfuniuniuniuniuniun_3C_uni_3E_ to i8*), i8* %args_ptr, i32 %div_width_load15_, i32 %div_height_load16_yspan_load, i32 1)
  %launch_group_handle_load67 = load i8** %launch_group_handle, align 8
  %cmp68 = icmp eq i8* %launch_group_handle_load67, null
  br i1 %cmp68, label %post_sync, label %call_sync69

call_sync:                                        ; preds = %all_on
  call void @ISPCSync(i8* %launch_group_handle_load)
  store i8* null, i8** %launch_group_handle, align 8
  br label %post_sync

post_sync:                                        ; preds = %call_sync69, %call_sync, %some_on, %all_on
  ret void

call_sync69:                                      ; preds = %some_on
  call void @ISPCSync(i8* %launch_group_handle_load67)
  store i8* null, i8** %launch_group_handle, align 8
  br label %post_sync
}

define void @mandelbrot_ispc(float %x0, float %y0, float %x1, float %y1, i32 %width, i32 %height, i32 %maxIterations, i32* %output) {
allocas:
  %launch_group_handle = alloca i8*, align 8
  store i8* null, i8** %launch_group_handle, align 8
  %sub_x1_load_x0_load = fsub float %x1, %x0
  %width_load_to_float = sitofp i32 %width to float
  %div_sub_x1_load_x0_load_width_load_to_float = fdiv float %sub_x1_load_x0_load, %width_load_to_float
  %sub_y1_load_y0_load = fsub float %y1, %y0
  %height_load_to_float = sitofp i32 %height to float
  %div_sub_y1_load_y0_load_height_load_to_float = fdiv float %sub_y1_load_y0_load, %height_load_to_float
  %div_width_load15_ = sdiv i32 %width, 16
  %div_height_load16_yspan_load = sdiv i32 %height, 16
  %args_ptr = call i8* @ISPCAlloc(i8** %launch_group_handle, i64 96, i32 32)
  %funarg = bitcast i8* %args_ptr to float*
  store float %x0, float* %funarg, align 4
  %funarg17 = getelementptr i8* %args_ptr, i64 4
  %0 = bitcast i8* %funarg17 to float*
  store float %div_sub_x1_load_x0_load_width_load_to_float, float* %0, align 4
  %funarg18 = getelementptr i8* %args_ptr, i64 8
  %1 = bitcast i8* %funarg18 to float*
  store float %y0, float* %1, align 4
  %funarg19 = getelementptr i8* %args_ptr, i64 12
  %2 = bitcast i8* %funarg19 to float*
  store float %div_sub_y1_load_y0_load_height_load_to_float, float* %2, align 4
  %funarg20 = getelementptr i8* %args_ptr, i64 16
  %3 = bitcast i8* %funarg20 to i32*
  store i32 %width, i32* %3, align 4
  %funarg21 = getelementptr i8* %args_ptr, i64 20
  %4 = bitcast i8* %funarg21 to i32*
  store i32 %height, i32* %4, align 4
  %funarg22 = getelementptr i8* %args_ptr, i64 24
  %5 = bitcast i8* %funarg22 to i32*
  store i32 16, i32* %5, align 4
  %funarg23 = getelementptr i8* %args_ptr, i64 28
  %6 = bitcast i8* %funarg23 to i32*
  store i32 16, i32* %6, align 4
  %funarg24 = getelementptr i8* %args_ptr, i64 32
  %7 = bitcast i8* %funarg24 to i32*
  store i32 %maxIterations, i32* %7, align 4
  %funarg25 = getelementptr i8* %args_ptr, i64 40
  %8 = bitcast i8* %funarg25 to i32**
  store i32* %output, i32** %8, align 8
  %funarg_mask = getelementptr i8* %args_ptr, i64 64
  %9 = bitcast i8* %funarg_mask to <8 x i32>*
  store <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32>* %9, align 32
  call void @ISPCLaunch(i8** %launch_group_handle, i8* bitcast (void ({ float, float, float, float, i32, i32, i32, i32, i32, i32*, <8 x i32> }*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @mandelbrot_scanline___unfunfunfunfuniuniuniuniuniun_3C_uni_3E_ to i8*), i8* %args_ptr, i32 %div_width_load15_, i32 %div_height_load16_yspan_load, i32 1)
  %launch_group_handle_load = load i8** %launch_group_handle, align 8
  %cmp = icmp eq i8* %launch_group_handle_load, null
  br i1 %cmp, label %post_sync, label %call_sync

call_sync:                                        ; preds = %allocas
  call void @ISPCSync(i8* %launch_group_handle_load)
  store i8* null, i8** %launch_group_handle, align 8
  br label %post_sync

post_sync:                                        ; preds = %call_sync, %allocas
  ret void
}
