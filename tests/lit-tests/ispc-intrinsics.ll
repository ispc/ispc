; RUN: %{ispc-opt} --target=host --passes=lower-ispc-intrinsics %s -o - | FileCheck %s

; REQUIRES: OPAQUE_PTRS_ENABLED

; CHECK-LABEL: @atomicrmw
; CHECK-DAG: [[RES:%.*]] = atomicrmw xchg ptr %ptr, float %val seq_cst
; CHECK-DAG: ret float [[RES]]
declare float @llvm.ispc.atomicrmw.xchg.seq_cst.f32(ptr, float)
define float @atomicrmw(ptr %ptr, float %val) {
  %res = call float @llvm.ispc.atomicrmw.xchg.seq_cst.f32(ptr %ptr, float %val)
  ret float %res
}

; CHECK-LABEL: @bitcast
; CHECK-DAG: [[RES:%.*]] = bitcast i16 %x to half
; CHECK-DAG: ret half [[RES]]
declare half @llvm.ispc.bitcast.i16.f16(i16, half)
define half @bitcast(i16 %x) {
  %calltmp = tail call half @llvm.ispc.bitcast.i16.f16(i16 %x, half 0xH0000)
  ret half %calltmp
}

; CHECK-LABEL: @concat
; CHECK-DAG: [[RES:%.*]] = shufflevector <2 x i32> %x, <2 x i32> %y, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-DAG: ret <4 x i32> [[RES]]
declare <4 x i32> @llvm.ispc.concat.v2i32(<2 x i32>, <2 x i32>)
define <4 x i32> @concat(<2 x i32> %x, <2 x i32> %y) {
  %calltmp = call <4 x i32> @llvm.ispc.concat.v2i32(<2 x i32> %x, <2 x i32> %y)
  ret <4 x i32> %calltmp
}

; CHECK-LABEL: define i32 @cmpxchg(ptr %ptr, i32 %cmp, i32 %val) {
; CHECK-DAG:   [[CMPXCHG:%.*]] = cmpxchg ptr %ptr, i32 %cmp, i32 %val seq_cst seq_cst, align 4
; CHECK-DAG:   [[RES:%.*]] = extractvalue { i32, i1 } [[CMPXCHG]], 0
; CHECK-DAG:   ret i32 [[RES]]
declare i32 @llvm.ispc.cmpxchg.seq_cst.seq_cst.i32(ptr, i32, i32)
define i32 @cmpxchg(ptr %ptr, i32 %cmp, i32 %val) {
  %res = call i32 @llvm.ispc.cmpxchg.seq_cst.seq_cst.i32(ptr %ptr, i32 %cmp, i32 %val)
  ret i32 %res
}

; CHECK-LABEL: @extract
; CHECK-DAG: [[RES:%.*]] = extractelement <2 x i64> %x, i32 %idx
; CHECK-DAG: ret i64 [[RES]]
declare i64 @llvm.ispc.extract.v2i64(<2 x i64>, i32)
define i64 @extract(<2 x i64> %x, i32 %idx) {
  %calltmp = call i64 @llvm.ispc.extract.v2i64(<2 x i64> %x, i32 %idx)
  ret i64 %calltmp
}

; CHECK-LABEL: @fence
; CHECK-DAG: fence seq_cst
; CHECK-DAG: ret void
declare void @llvm.ispc.fence.seq_cst()
define void @fence() {
  call void @llvm.ispc.fence.seq_cst()
  ret void
}

; CHECK-LABEL: @insert
; CHECK-DAG: [[RES:%.*]] = insertelement <8 x i16> %x, i16 %val, i32 %idx
; CHECK-DAG: ret <8 x i16> [[RES]]
declare <8 x i16> @llvm.ispc.insert.v8i16(<8 x i16>, i32, i16)
define <8 x i16> @insert(<8 x i16> %x, i16 %val, i32 %idx) {
  %calltmp = call <8 x i16> @llvm.ispc.insert.v8i16(<8 x i16> %x, i32 %idx, i16 %val)
  ret <8 x i16> %calltmp
}

; CHECK-LABEL: @packmask
; CHECK-DAG: [[BITCAST:%.*]] = bitcast <8 x i1> %mask to [[TYPE:i.*]]
; CHECK-DAG: [[RES:%.*]] = zext [[TYPE]] [[BITCAST]] to i64
; CHECK-DAG: ret i64 [[RES]]
declare i64 @llvm.ispc.packmask.v8i1(<8 x i1>)
define i64 @packmask(<8 x i1> %mask) {
  %res = call i64 @llvm.ispc.packmask.v8i1(<8 x i1> %mask)
  ret i64 %res
}

; CHECK-LABEL: @select
; CHECK-DAG: [[RES:%.*]] = select <8 x i1> %mask, <8 x float> %a, <8 x float> %b
; CHECK-DAG: ret <8 x float> [[RES]]
declare <8 x float> @llvm.ispc.select.v8f32(<8 x i1>, <8 x float>, <8 x float>)
define <8 x float> @select(<8 x i1> %mask, <8 x float> %a, <8 x float> %b) {
  %res = call <8 x float> @llvm.ispc.select.v8f32(<8 x i1> %mask, <8 x float> %a, <8 x float> %b)
  ret <8 x float> %res
}

; CHECK-LABEL: @streaming_load
; CHECK-DAG: [[RES:%.*]] = load i8, ptr %ptr, align 1, !nontemporal [[META:!.*]]
; CHECK-DAG: ret i8 [[RES]]
declare i8 @llvm.ispc.stream_load.i8(ptr, i8)
define i8 @streaming_load(ptr %ptr, i8 %dummy) {
  %res = call i8 @llvm.ispc.stream_load.i8(ptr %ptr, i8 %dummy)
  ret i8 %res
}

; CHECK-LABEL: @streaming_store
; CHECK-DAG: store i16 %val, ptr %ptr, align 2, !nontemporal !0
; CHECK-DAG: ret void
declare void @llvm.ispc.stream_store.i16(ptr, i16)
define void @streaming_store(ptr %ptr, i16 %val) {
  call void @llvm.ispc.stream_store.i16(ptr %ptr, i16 %val)
  ret void
}

; CHECK: [[META]] = !{i32 1}
