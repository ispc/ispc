; ModuleID = 'task.bc'
target datalayout = "e-p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-n16:32:64"
target triple = "nvptx64"

@data = external global [1024 x i32]

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
declare i32 @foo1___(<1 x i32>) #5

; Function Attrs: nounwind
define void @foo___(<1 x i32> %__mask) #5 {
allocas:
  %calltmp = tail call i32 @foo1___(<1 x i32> %__mask)
  %calltmp_to_int64 = sext i32 %calltmp to i64
  %data_offset = getelementptr [1024 x i32]* @data, i64 0, i64 %calltmp_to_int64
  store i32 0, i32* %data_offset, align 4
  ret void
}

attributes #0 = { alwaysinline nounwind readnone }
attributes #1 = { nounwind readnone }
attributes #2 = { alwaysinline nounwind }
attributes #3 = { nounwind }
attributes #4 = { alwaysinline nounwind readonly }
attributes #5 = { nounwind "target-features"="+sm_35" }
