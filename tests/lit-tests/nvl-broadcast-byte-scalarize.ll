; REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_22_0+
; RUN: %{ispc-opt} --target=avx10.2nvl-x8 --arch=x86-64 --passes=scalarize %s -o - | FileCheck %s --check-prefix=NVL
; RUN: %{ispc-opt} --target=avx10.2nvl-x16 --arch=x86-64 --passes=scalarize %s -o - | FileCheck %s --check-prefixes=OTHER,NVL16
; RUN: %{ispc-opt} --target=avx512spr-x8 --arch=x86-64 --passes=scalarize %s -o - | FileCheck %s --check-prefix=SPR8
; RUN: %{ispc-opt} --target=avx512spr-x16 --arch=x86-64 --passes=scalarize %s -o - | FileCheck %s --check-prefix=SPR16

; NVL-LABEL: define i8 @ordinary(
; NVL: [[OFFSET:%.*]] = or i32 %base_index, 4
; NVL-NEXT: [[WIDE:%.*]] = zext nneg i32 [[OFFSET]] to i64
; NVL-NEXT: %address = getelementptr i8, ptr %base, i64 [[WIDE]]
; SPR8-LABEL: define i8 @ordinary(
; SPR8: %lane = extractelement <8 x i64> %wide, i64 5
; OTHER-LABEL: define i8 @ordinary(
; OTHER: %lane = extractelement <8 x i64> %wide, i64 5
define i8 @ordinary(ptr %base, i32 %base_index) {
  %seed = insertelement <8 x i32> poison, i32 %base_index, i64 0
  %splat = shufflevector <8 x i32> %seed, <8 x i32> poison, <8 x i32> zeroinitializer
  %offsets = or disjoint <8 x i32> %splat, <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  %wide = zext nneg <8 x i32> %offsets to <8 x i64>
  %lane = extractelement <8 x i64> %wide, i64 5
  %address = getelementptr i8, ptr %base, i64 %lane
  %byte = load i8, ptr %address, align 1
  ret i8 %byte
}

; NVL-LABEL: define i8 @plain_zext(
; NVL: [[PLAIN_OFFSET:%.*]] = or i32 %base_index, 4
; NVL-NEXT: [[PLAIN_WIDE:%.*]] = zext i32 [[PLAIN_OFFSET]] to i64
; NVL-NEXT: %address = getelementptr i8, ptr %base, i64 [[PLAIN_WIDE]]
define i8 @plain_zext(ptr %base, i32 %base_index) {
  %seed = insertelement <8 x i32> poison, i32 %base_index, i64 0
  %splat = shufflevector <8 x i32> %seed, <8 x i32> poison, <8 x i32> zeroinitializer
  %offsets = or <8 x i32> %splat, <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  %wide = zext <8 x i32> %offsets to <8 x i64>
  %lane = extractelement <8 x i64> %wide, i64 5
  %address = getelementptr i8, ptr %base, i64 %lane
  %byte = load i8, ptr %address, align 1
  ret i8 %byte
}

; NVL-LABEL: define i8 @masked(
; NVL: %lane = extractelement <8 x i64> %wide, i64 5
define i8 @masked(ptr %base, i32 %base_index, <8 x i1> %mask) {
  %seed = insertelement <8 x i32> poison, i32 %base_index, i64 0
  %splat = shufflevector <8 x i32> %seed, <8 x i32> poison, <8 x i32> zeroinitializer
  %offsets = or <8 x i32> %splat, <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  %active = select <8 x i1> %mask, <8 x i32> %offsets, <8 x i32> zeroinitializer
  %wide = zext <8 x i32> %active to <8 x i64>
  %lane = extractelement <8 x i64> %wide, i64 5
  %address = getelementptr i8, ptr %base, i64 %lane
  %byte = load i8, ptr %address, align 1
  ret i8 %byte
}

; SPR16-LABEL: define i8 @ordinary16(
; SPR16: %lane = extractelement <16 x i64> %wide, i64 13
; NVL16-LABEL: define i8 @ordinary16(
; NVL16: [[OFFSET16:%.*]] = or i32 %base_index, 12
; NVL16-NEXT: [[WIDE16:%.*]] = zext nneg i32 [[OFFSET16]] to i64
; NVL16-NEXT: %address = getelementptr i8, ptr %base, i64 [[WIDE16]]
define i8 @ordinary16(ptr %base, i32 %base_index) {
  %seed = insertelement <16 x i32> poison, i32 %base_index, i64 0
  %splat = shufflevector <16 x i32> %seed, <16 x i32> poison, <16 x i32> zeroinitializer
  %offsets = or disjoint <16 x i32> %splat, <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12>
  %wide = zext nneg <16 x i32> %offsets to <16 x i64>
  %lane = extractelement <16 x i64> %wide, i64 13
  %address = getelementptr i8, ptr %base, i64 %lane
  %byte = load i8, ptr %address, align 1
  ret i8 %byte
}

; NVL-LABEL: define i8 @volatile(
; NVL: %lane = extractelement <8 x i64> %wide, i64 5
; NVL-NEXT: %address = getelementptr i8, ptr %base, i64 %lane
define i8 @volatile(ptr %base, i32 %base_index) {
  %seed = insertelement <8 x i32> poison, i32 %base_index, i64 0
  %splat = shufflevector <8 x i32> %seed, <8 x i32> poison, <8 x i32> zeroinitializer
  %offsets = or disjoint <8 x i32> %splat, <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  %wide = zext nneg <8 x i32> %offsets to <8 x i64>
  %lane = extractelement <8 x i64> %wide, i64 5
  %address = getelementptr i8, ptr %base, i64 %lane
  %byte = load volatile i8, ptr %address, align 1
  ret i8 %byte
}

; NVL-LABEL: define i8 @atomic(
; NVL: %lane = extractelement <8 x i64> %wide, i64 5
; NVL-NEXT: %address = getelementptr i8, ptr %base, i64 %lane
define i8 @atomic(ptr %base, i32 %base_index) {
  %seed = insertelement <8 x i32> poison, i32 %base_index, i64 0
  %splat = shufflevector <8 x i32> %seed, <8 x i32> poison, <8 x i32> zeroinitializer
  %offsets = or disjoint <8 x i32> %splat, <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  %wide = zext nneg <8 x i32> %offsets to <8 x i64>
  %lane = extractelement <8 x i64> %wide, i64 5
  %address = getelementptr i8, ptr %base, i64 %lane
  %byte = load atomic i8, ptr %address monotonic, align 1
  ret i8 %byte
}

; NVL-LABEL: define i8 @shared_offset(
; NVL: %lane = extractelement <8 x i64> %wide, i64 5
; NVL-NEXT: %address = getelementptr i8, ptr %base, i64 %lane
define i8 @shared_offset(ptr %base, i32 %base_index) {
  %seed = insertelement <8 x i32> poison, i32 %base_index, i64 0
  %splat = shufflevector <8 x i32> %seed, <8 x i32> poison, <8 x i32> zeroinitializer
  %offsets = or disjoint <8 x i32> %splat, <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  %wide = zext nneg <8 x i32> %offsets to <8 x i64>
  %lane = extractelement <8 x i64> %wide, i64 5
  %address = getelementptr i8, ptr %base, i64 %lane
  %byte = load i8, ptr %address, align 1
  %offset_byte = trunc i64 %lane to i8
  %result = xor i8 %byte, %offset_byte
  ret i8 %result
}
