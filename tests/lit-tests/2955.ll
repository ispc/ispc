; RUN: %{ispc-opt} --passes=replace-masked-memory-ops %s -o - | FileCheck %s

declare <8 x float> @llvm.masked.load.v8f32.p0(ptr nocapture, i32 immarg, <8 x i1>, <8 x float>) #1
declare void @llvm.masked.store.v8f32.p0(<8 x float>, ptr nocapture, i32 immarg, <8 x i1>) #2

; CHECK-LABEL: @foo
; CHECK-NEXT:  [[HL:%.*]] = load <3 x float>, ptr %0, align 1
; CHECK-NEXT:  [[V4:%.*]] = shufflevector <3 x float> [[HL]], <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:  [[VEC:%.*]] = shufflevector <4 x float> [[V4]], <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK-NEXT:  [[HS:%.*]] = shufflevector <8 x float> [[VEC]], <8 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-NEXT:  store <3 x float> [[HS]], ptr %1, align 1
; CHECK-NEXT:  ret void

define void @foo(ptr noalias %0, ptr noalias %1) local_unnamed_addr {
  %x = tail call <8 x float> @llvm.masked.load.v8f32.p0(ptr %0, i32 1, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>, <8 x float> poison)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %x, ptr %1, i32 1, <8 x i1> <i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false>)
  ret void
}
