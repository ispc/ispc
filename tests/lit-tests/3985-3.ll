; Regression test for https://github.com/ispc/ispc/issues/3985
; A gather with zero-extended offsets that are linear for the lanes that are on
; becomes a vector load. Its address comes from lane 0 if the zext operand is
; uniform, constant in [0, INT32_MAX] (also unfolded, without poison generating
; flags), or a uniform value plus lane constants combined with an add nuw whose
; smallest constant is in lane 0 or an or disjoint whose constant is 0 in lane
; 0, possibly under a shl nuw or mul nuw by a constant. Otherwise, as for x - 1,
; x - uniform, splat(uniform) + lane constants or a varying loop index, it comes
; from the first lane that is on.

; RUN: %{ispc-opt} --target=sse4-i32x4 --addressing=64 --passes=improve-memory-ops %s -o - | FileCheck %s

; REQUIRES: X86_ENABLED

declare <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x i8> @__masked_load_i8(ptr, <4 x i32>)
declare <4 x i16> @__pseudo_gather_factored_base_offsets64_i16(ptr, <4 x i64>, i32, <4 x i64>, <4 x i32>)
declare <4 x i16> @__masked_load_i16(ptr, <4 x i32>)

; The loop counter %i is the same in all lanes: lane 0.
; CHECK-LABEL: @uniform_phi(
; CHECK-NOT: @llvm.cttz
; CHECK: call <4 x i8> @__masked_load_i8(
; CHECK-NOT: @llvm.cttz
; CHECK: ret
define <4 x i8> @uniform_phi(ptr %base, i32 %n, <4 x i32> %mask) {
entry:
  br label %loop
loop:
  %i = phi <4 x i32> [ zeroinitializer, %entry ], [ %next, %loop ]
  %k = phi i32 [ 0, %entry ], [ %knext, %loop ]
  %lanes = or disjoint <4 x i32> %i, <i32 0, i32 1, i32 2, i32 3>
  %off = zext <4 x i32> %lanes to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> %off, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
  %next = add nuw nsw <4 x i32> %i, <i32 4, i32 4, i32 4, i32 4>
  %knext = add i32 %k, 1
  %again = icmp ult i32 %knext, %n
  br i1 %again, label %loop, label %exit
exit:
  ret <4 x i8> %v
}

; %x is <0,1,2,3>, then <0,2,4,6>: not uniform, so the first lane that is on.
; CHECK-LABEL: @varying_phi(
; CHECK: call {{.*}} @llvm.cttz
define i8 @varying_phi(ptr %base) {
entry:
  br label %loop
loop:
  %x = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %entry ], [ %next, %loop ]
  %k = phi i32 [ 0, %entry ], [ %knext, %loop ]
  %off = zext <4 x i32> %x to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> %off, i32 1, <4 x i64> zeroinitializer, <4 x i32> <i32 0, i32 -1, i32 0, i32 0>)
  %next = add nuw <4 x i32> %x, %x
  %knext = add i32 %k, 1
  %again = icmp ult i32 %knext, 2
  br i1 %again, label %loop, label %exit
exit:
  %r = extractelement <4 x i8> %v, i32 1
  ret i8 %r
}

; Lane 0 of the nuw add is poison; folding would hide it.
; CHECK-LABEL: @poison_lane0(
; CHECK: call {{.*}} @llvm.cttz
define <4 x i8> @poison_lane0(ptr %base) {
  %x = add nuw <4 x i32> <i32 -1, i32 0, i32 1, i32 2>, <i32 1, i32 1, i32 1, i32 1>
  %off = zext <4 x i32> %x to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> zeroinitializer, i32 1, <4 x i64> %off, <4 x i32> <i32 0, i32 -1, i32 -1, i32 -1>)
  ret <4 x i8> %v
}

; An unfolded constant without flags, as split off from an offset: lane 0.
; CHECK-LABEL: @unfolded_constant(
; CHECK-NOT: @llvm.cttz
; CHECK: call <4 x i16> @__masked_load_i16(
; CHECK: ret
define <4 x i16> @unfolded_constant(ptr %base, <4 x i32> %mask) {
  %x = shl <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <i32 1, i32 1, i32 1, i32 1>
  %off = zext <4 x i32> %x to <4 x i64>
  %v = call <4 x i16> @__pseudo_gather_factored_base_offsets64_i16(ptr %base, <4 x i64> zeroinitializer, i32 1, <4 x i64> %off, <4 x i32> %mask)
  ret <4 x i16> %v
}

; A varying loop index <0,1,2,3> + 4k, without nuw, may wrap.
; CHECK-LABEL: @varying_index(
; CHECK: call {{.*}} @llvm.cttz
define <4 x i8> @varying_index(ptr %base, i32 %n, <4 x i32> %mask) {
entry:
  br label %loop
loop:
  %i = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %entry ], [ %next, %loop ]
  %k = phi i32 [ 0, %entry ], [ %knext, %loop ]
  %off = zext <4 x i32> %i to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> %off, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
  %next = add <4 x i32> %i, <i32 4, i32 4, i32 4, i32 4>
  %knext = add i32 %k, 1
  %again = icmp ult i32 %knext, %n
  br i1 %again, label %loop, label %exit
exit:
  ret <4 x i8> %v
}

; programIndex - d with a uniform d: lane 0 wraps for d = 1.
; CHECK-LABEL: @minus_uniform(
; CHECK: call {{.*}} @llvm.cttz
define <4 x i8> @minus_uniform(ptr %base, i32 %d, <4 x i32> %mask) {
  %s = insertelement <4 x i32> poison, i32 %d, i32 0
  %u = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %x = sub <4 x i32> <i32 0, i32 1, i32 2, i32 3>, %u
  %off = zext <4 x i32> %x to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> %off, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
  ret <4 x i8> %v
}

; A varying loop index minus 1, j + splat(-1): lane 0 wraps for j = 0.
; CHECK-LABEL: @varying_index_minus_1(
; CHECK: call {{.*}} @llvm.cttz
define <4 x i8> @varying_index_minus_1(ptr %base, i32 %n, <4 x i32> %mask) {
entry:
  br label %loop
loop:
  %i = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %entry ], [ %next, %loop ]
  %k = phi i32 [ 0, %entry ], [ %knext, %loop ]
  %j = add <4 x i32> %i, <i32 -1, i32 -1, i32 -1, i32 -1>
  %off = zext <4 x i32> %j to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> %off, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
  %next = add <4 x i32> %i, <i32 4, i32 4, i32 4, i32 4>
  %knext = add i32 %k, 1
  %again = icmp ult i32 %knext, %n
  br i1 %again, label %loop, label %exit
exit:
  ret <4 x i8> %v
}

; j - d with j = b + programIndex, as InstCombine reassociates it: lane 0 of
; splat(b - d) + <0,1,2,3> wraps for b = 0, d = 1.
; CHECK-LABEL: @reassociated_minus_uniform(
; CHECK: call {{.*}} @llvm.cttz
define <4 x i8> @reassociated_minus_uniform(ptr %base, i32 %b, i32 %d, <4 x i32> %mask) {
  %bd = sub i32 %b, %d
  %s = insertelement <4 x i32> poison, i32 %bd, i32 0
  %u = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %x = add <4 x i32> %u, <i32 0, i32 1, i32 2, i32 3>
  %off = zext <4 x i32> %x to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> %off, i32 1, <4 x i64> zeroinitializer, <4 x i32> %mask)
  ret <4 x i8> %v
}

; The same with nuw under a shl nuw: lane 0 is the smallest lane.
; CHECK-LABEL: @nuw_uniform(
; CHECK-NOT: @llvm.cttz
; CHECK: call <4 x i16> @__masked_load_i16(
; CHECK: ret
define <4 x i16> @nuw_uniform(ptr %base, i32 %b, <4 x i32> %mask) {
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %u = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %x = add nuw <4 x i32> %u, <i32 0, i32 1, i32 2, i32 3>
  %y = shl nuw <4 x i32> %x, <i32 1, i32 1, i32 1, i32 1>
  %off = zext <4 x i32> %y to <4 x i64>
  %v = call <4 x i16> @__pseudo_gather_factored_base_offsets64_i16(ptr %base, <4 x i64> zeroinitializer, i32 1, <4 x i64> %off, <4 x i32> %mask)
  ret <4 x i16> %v
}

; Without nuw on the shl, (0x7FFFFFFF + <0,1,2,3>) << 1 wraps from lane 1 on.
; CHECK-LABEL: @shl_without_nuw(
; CHECK: call {{.*}} @llvm.cttz
define <4 x i16> @shl_without_nuw(ptr %base, i32 %b, <4 x i32> %mask) {
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %u = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %x = add nuw <4 x i32> %u, <i32 0, i32 1, i32 2, i32 3>
  %y = shl <4 x i32> %x, <i32 1, i32 1, i32 1, i32 1>
  %off = zext <4 x i32> %y to <4 x i64>
  %v = call <4 x i16> @__pseudo_gather_factored_base_offsets64_i16(ptr %base, <4 x i64> zeroinitializer, i32 1, <4 x i64> %off, <4 x i32> %mask)
  ret <4 x i16> %v
}

; Lane 0 of an or disjoint whose constant isn't 0 in lane 0 may be poison while
; lane 1 isn't, like for b = 1.
; CHECK-LABEL: @or_disjoint_lane0_nonzero(
; CHECK: call {{.*}} @llvm.cttz
define <4 x i8> @or_disjoint_lane0_nonzero(ptr %base, i32 %b, <4 x i32> %mask) {
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %u = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %x = or disjoint <4 x i32> %u, <i32 1, i32 2, i32 3, i32 4>
  %off = zext <4 x i32> %x to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> zeroinitializer, i32 1, <4 x i64> %off, <4 x i32> %mask)
  ret <4 x i8> %v
}

; An add nuw whose lane 0 constant isn't the smallest: lane 0 of
; b + <-1,0,1,2> is poison for b = 0 while lane 1 isn't.
; CHECK-LABEL: @add_nuw_lane0_not_smallest(
; CHECK: call {{.*}} @llvm.cttz
define <4 x i8> @add_nuw_lane0_not_smallest(ptr %base, i32 %b, <4 x i32> %mask) {
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %u = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %x = add nuw <4 x i32> %u, <i32 -1, i32 0, i32 1, i32 2>
  %off = zext <4 x i32> %x to <4 x i64>
  %v = call <4 x i8> @__pseudo_gather_factored_base_offsets64_i8(ptr %base, <4 x i64> zeroinitializer, i32 1, <4 x i64> %off, <4 x i32> %mask)
  ret <4 x i8> %v
}

; A mul nuw by a constant keeps lane 0 the smallest lane: lane 0.
; CHECK-LABEL: @mul_nuw_uniform(
; CHECK-NOT: @llvm.cttz
; CHECK: call <4 x i16> @__masked_load_i16(
; CHECK: ret
define <4 x i16> @mul_nuw_uniform(ptr %base, i32 %b, <4 x i32> %mask) {
  %s = insertelement <4 x i32> poison, i32 %b, i32 0
  %u = shufflevector <4 x i32> %s, <4 x i32> poison, <4 x i32> zeroinitializer
  %x = add nuw <4 x i32> %u, <i32 0, i32 1, i32 2, i32 3>
  %y = mul nuw <4 x i32> %x, <i32 2, i32 2, i32 2, i32 2>
  %off = zext <4 x i32> %y to <4 x i64>
  %v = call <4 x i16> @__pseudo_gather_factored_base_offsets64_i16(ptr %base, <4 x i64> zeroinitializer, i32 1, <4 x i64> %off, <4 x i32> %mask)
  ret <4 x i16> %v
}
