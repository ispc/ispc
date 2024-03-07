; RUN: %{ispc-opt} --passes=replace-masked-memory-ops %s -o - | FileCheck %s

; REQUIRES: LLVM_17_0+

declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32, <4 x i1>, <4 x i32>)
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>)

; CHECK-LABEL: @foo
; CHECK-NEXT: [[HL:%.*]] = load <2 x i32>, ptr %a, align 16
; CHECK-NEXT: [[VEC:%.*]] = shufflevector <2 x i32> [[HL]], <2 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT: [[HS:%.*]] = shufflevector <4 x i32> [[VEC]], <4 x i32> undef, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT: store <2 x i32> [[HS]], ptr %b, align 16
; CHECK-NEXT: ret void
define void @foo(<4 x i32>* %a, <4 x i32>* %b) {
  %vec = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %a, i32 16,
                                                        <4 x i1> <i1 true, i1 true, i1 false, i1 false>,
                                                        <4 x i32> <i32 poison, i32 poison, i32 0, i32 0>)
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %vec, <4 x i32>* %b, i32 16,
                                             <4 x i1> <i1 true, i1 true, i1 false, i1 false>)
  ret void
}

; CHECK-LABEL: @bad_mask
; CHECK-NEXT: call <4 x i32> @llvm.masked.load
; CHECK-NEXT: ret void
define void @bad_mask(<4 x i32>* %a) {
  %vec1 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %a, i32 1,
                                                        <4 x i1> <i1 true, i1 true, i1 true, i1 false>,
                                                        <4 x i32> <i32 poison, i32 poison, i32 0, i32 0>)

  ret void
}

; CHECK-LABEL: @oneelemmask
; CHECK-NEXT: [[V1:%.*]] = load <1 x i32>, ptr %a, align 1
; CHECK-NEXT: [[V2:%.*]] = shufflevector <1 x i32> [[V1]], <1 x i32> zeroinitializer, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT: {{%.*}} = shufflevector <2 x i32> [[V2]], <2 x i32> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT: ret void
define void@oneelemmask(<4 x i32>* %a) {
  %vec2 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %a, i32 1,
                                                        <4 x i1> <i1 true, i1 false, i1 false, i1 false>,
                                                        <4 x i32> <i32 poison, i32 0, i32 0, i32 0>)
  ret void
}

; CHECK-LABEL: @passthrough
; CHECK-NEXT: [[V1:%.*]] = load <2 x i32>, ptr %a, align 8
; CHECK-NEXT: {{.*}} = shufflevector <2 x i32> [[V1]], <2 x i32> <i32 42, i32 43>, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT: ret void
define void @passthrough(<4 x i32>* %a, <4 x i32>* %b) {
  %vec = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %a, i32 8,
                                                        <4 x i1> <i1 true, i1 true, i1 false, i1 false>,
                                                        <4 x i32> <i32 poison, i32 poison, i32 42, i32 43>)
  ret void
}

; non-const mask values are not supported.
; CHECK-LABEL: @notconstmaskargs
; CHECK-NEXT: @llvm.masked.load
; CHECK-NEXT: @llvm.masked.store
; CHECK-NEXT: ret void
define void @notconstmaskargs(ptr %a, <4 x i1> %m) {
  %vec = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(ptr %a, i32 8, <4 x i1> %m,
                                                        <4 x i32> <i32 poison, i32 poison, i32 42, i32 43>)
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %vec, ptr %a, i32 16, <4 x i1> %m)
  ret void
}

; non-const passthrough values are not supported.
; CHECK-LABEL: @notconstpassthrough
; CHECK-NEXT: @llvm.masked.load
; CHECK-NEXT: ret void
define void @notconstpassthrough(ptr %a, <4 x i32> %b) {
  %vec = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(ptr %a, i32 8,
                                                        <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i32> %b)
  ret void
}

declare <8 x float> @llvm.masked.load.v8f32.p0v8f32(ptr, i32, <8 x i1>, <8 x float>)
; Reducing to quarters is not supported at the moment or lesser parts are
; useful for x16 and wider targets.
; CHECK-LABEL: @quarter
; CHECK-NEXT: [[V1:%.*]] = load <2 x float>, ptr %a, align 1
; CHECK-NEXT: [[V2:%.*]] = shufflevector <2 x float> [[V1]], <2 x float> <float 1.000000e+00, float 0.000000e+00>, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT: [[V:%.*]] = shufflevector <4 x float> [[V2]], <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
define void @quarter(ptr %a) {
  %vec1 = call <8 x float> @llvm.masked.load.v8f32.p0v8f32(ptr %a, i32 1,
                    <8 x i1> <i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>,
                    <8 x float> <float poison, float poison, float 1.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>)
  ret void
}

; do not optimize if the mask is all false because such loads are removed by other passes.
; CHECK-LABEL: @allzeromask
; CHECK-NEXT: @llvm.masked.load
; CHECK-NEXT: ret <4 x i32> %vec
define <4 x i32> @allzeromask(<4 x i32>* %a, <4 x i32>* %b) {
  %vec = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %a, i32 8,
                                                        <4 x i1> <i1 false, i1 false, i1 false, i1 false>,
                                                        <4 x i32> <i32 poison, i32 poison, i32 42, i32 43>)
  ret <4 x i32> %vec
}
