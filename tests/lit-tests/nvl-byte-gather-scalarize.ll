; REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_22_0+
; RUN: %{ispc-opt} --target=avx10.2nvl-x16 --arch=x86-64 --passes=scalarize %s -o - | FileCheck %s --check-prefix=NVL
; RUN: %{ispc-opt} --target=avx10.2nvl-x8 --arch=x86-64 --passes=scalarize %s -o - | FileCheck %s --check-prefix=X8

; NVL-LABEL: define i8 @ordinary(
; NVL: [[LANE:%.*]] = extractelement <16 x i32> %indices, i64 3
; NVL-NEXT: [[OFFSET:%.*]] = zext nneg i32 [[LANE]] to i64
; NVL-NEXT: %address = getelementptr i8, ptr %base, i64 [[OFFSET]]
; X8-LABEL: define i8 @ordinary(
; X8: %wide = zext nneg <16 x i32> %indices to <16 x i64>
; X8-NEXT: %offset = extractelement <16 x i64> %wide, i64 3
define i8 @ordinary(ptr %idx, ptr %base) {
  %indices = load <16 x i32>, ptr %idx, align 4
  %wide = zext nneg <16 x i32> %indices to <16 x i64>
  %offset = extractelement <16 x i64> %wide, i64 3
  %address = getelementptr i8, ptr %base, i64 %offset
  %byte = load i8, ptr %address, align 1
  ret i8 %byte
}

; NVL-LABEL: define i64 @multi_use(
; NVL: %offset = extractelement <16 x i64> %wide, i64 3
define i64 @multi_use(ptr %idx, ptr %base) {
  %indices = load <16 x i32>, ptr %idx, align 4
  %wide = zext <16 x i32> %indices to <16 x i64>
  %offset = extractelement <16 x i64> %wide, i64 3
  %address = getelementptr i8, ptr %base, i64 %offset
  %byte = load i8, ptr %address, align 1
  %value = zext i8 %byte to i64
  %result = add i64 %value, %offset
  ret i64 %result
}

; NVL-LABEL: define i8 @masked(
; NVL: %offset = extractelement <16 x i64> %wide, i64 3
define i8 @masked(ptr %idx, ptr %base, <16 x i1> %mask) {
  %indices = call <16 x i32> @llvm.masked.load.v16i32.p0(ptr %idx, <16 x i1> %mask, <16 x i32> zeroinitializer)
  %wide = zext <16 x i32> %indices to <16 x i64>
  %offset = extractelement <16 x i64> %wide, i64 3
  %address = getelementptr i8, ptr %base, i64 %offset
  %byte = load i8, ptr %address, align 1
  ret i8 %byte
}
declare <16 x i32> @llvm.masked.load.v16i32.p0(ptr, <16 x i1>, <16 x i32>)
