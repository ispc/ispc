; The decoder of varying addresses (AddressDecoder.h) on hand-written pointer
; IR: the base, the object, the origin, one term per variable GEP index and
; the constant byte offsets of the lanes, for every element type spelling and
; GEP shape that LLVM 20 to trunk produce.
;
; RUN: %{ispc-opt} --target=avx2-i32x4 --passes=print-memory-access %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: X86_ENABLED

; The data layout of the x86 targets: address space 271 has 32-bit pointers.
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

%S = type { float, i32, float }

@g = global [64 x float] zeroinitializer
@h = global [64 x float] zeroinitializer
@g1 = addrspace(1) global [64 x float] zeroinitializer

declare <4 x float> @__pseudo_gather64_float(<4 x i64>, <4 x i32>)
declare <4 x float> @__pseudo_gather32_float(<4 x i32>, <4 x i32>)
declare <4 x ptr> @make_ptrs()

; The index under its extension, with the element size as the coefficient.
; CHECK-LABEL: memory accesses of @typed_float:
; CHECK:      address: uniform base ptr %a (scalar, gep), one object
; CHECK-NEXT:   term: sext <4 x i32> %i * 4
; CHECK-NEXT:   lanes: 0 0 0 0 (i64)
define <4 x float> @typed_float(ptr %a, <4 x i32> %x, <4 x i32> %m) {
  %i = add nsw <4 x i32> %x, splat (i32 1)
  %e = sext <4 x i32> %i to <4 x i64>
  %p = getelementptr float, ptr %a, <4 x i64> %e
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  ret <4 x float> %v
}

; LLVM 23 and later spell the element type as bytes.
; CHECK-LABEL: memory accesses of @typed_bytes:
; CHECK:      address: uniform base ptr %a (scalar, gep), one object
; CHECK-NEXT:   term: zext nneg <4 x i32> %x * 4
; CHECK-NEXT:   lanes: 0 0 0 0 (i64)
define <4 x float> @typed_bytes(ptr %a, <4 x i32> %x, <4 x i32> %m) {
  %e = zext nneg <4 x i32> %x to <4 x i64>
  %p = getelementptr [4 x i8], ptr %a, <4 x i64> %e
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  ret <4 x float> %v
}

; LLVM 20 and 21 keep a member access in a multi-index GEP; later versions
; split it into a byte GEP. Both give the struct size and the member offset.
; CHECK-LABEL: memory accesses of @struct_member:
; CHECK:      address: uniform base ptr %a (scalar, gep), one object
; CHECK-NEXT:   term: zext <4 x i32> %x * 12
; CHECK-NEXT:   lanes: 8 8 8 8 (i64)
; CHECK:      address: uniform base ptr %a (scalar, gep), one object
; CHECK-NEXT:   term: zext <4 x i32> %x * 12
; CHECK-NEXT:   lanes: 8 8 8 8 (i64)
define <4 x float> @struct_member(ptr %a, <4 x i32> %x, <4 x i32> %m) {
  %e = zext <4 x i32> %x to <4 x i64>
  %p = getelementptr %S, ptr %a, <4 x i64> %e, i32 2
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  %q0 = getelementptr [12 x i8], ptr %a, <4 x i64> %e
  %q = getelementptr i8, <4 x ptr> %q0, i64 8
  %bd = ptrtoint <4 x ptr> %q to <4 x i64>
  %w = call <4 x float> @__pseudo_gather64_float(<4 x i64> %bd, <4 x i32> %m)
  %r = fadd <4 x float> %v, %w
  ret <4 x float> %r
}

; A scalar GEP by a uniform index, then lane constants: the frontend's split
; of u + C. The uniform part is the scalar base.
; CHECK-LABEL: memory accesses of @uniform_then_lanes:
; CHECK:      address: uniform base ptr %p0 (scalar, gep), one object
; CHECK-NEXT:   lanes: 0 4 8 12 (i64)
define <4 x float> @uniform_then_lanes(ptr %a, i64 %u, <4 x i32> %m) {
  %p0 = getelementptr float, ptr %a, i64 %u
  %p = getelementptr float, ptr %p0, <4 x i64> <i64 0, i64 1, i64 2, i64 3>
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  ret <4 x float> %v
}

; A splat index vector is uniform too, and a splat of a scalar pointer is the
; uniform base.
; CHECK-LABEL: memory accesses of @splats:
; CHECK:      address: uniform base ptr %a (scalar, gep), one object
; CHECK-NEXT:   term: uniform sext i32 %u * 4
; CHECK-NEXT:   lanes: 0 0 0 0 (i64)
define <4 x float> @splats(ptr %a, i32 %u, <4 x i32> %m) {
  %s = insertelement <4 x ptr> poison, ptr %a, i32 0
  %as = shufflevector <4 x ptr> %s, <4 x ptr> poison, <4 x i32> zeroinitializer
  %ue = sext i32 %u to i64
  %t = insertelement <4 x i64> poison, i64 %ue, i32 0
  %us = shufflevector <4 x i64> %t, <4 x i64> poison, <4 x i32> zeroinitializer
  %p = getelementptr float, <4 x ptr> %as, <4 x i64> %us
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  ret <4 x float> %v
}

; In a byte GEP, a multiply or shift by a constant at the index width is the
; coefficient, and adds and subtracts at the index width are split exactly.
; Nothing is looked through an extension.
; CHECK-LABEL: memory accesses of @byte_arithmetic:
; CHECK:      address: uniform base ptr %a (scalar, gep), one object
; CHECK-NEXT:   term: sext <4 x i32> %x * 12
; CHECK-NEXT:   term: zext nneg <4 x i32> %y * -4
; CHECK-NEXT:   lanes: 36 36 36 36 (i64)
; CHECK:      address: uniform base ptr %a (scalar, gep), one object
; CHECK-NEXT:   term: sext <4 x i32> %xs * 4
; CHECK-NEXT:   lanes: 0 0 0 0 (i64)
define <4 x float> @byte_arithmetic(ptr %a, <4 x i32> %x, <4 x i32> %y, <4 x i32> %m) {
  %xe = sext <4 x i32> %x to <4 x i64>
  %ye = zext nneg <4 x i32> %y to <4 x i64>
  %o0 = mul <4 x i64> %xe, splat (i64 12)
  %o1 = shl <4 x i64> %ye, splat (i64 2)
  %o2 = add <4 x i64> %o0, splat (i64 36)
  %o = sub <4 x i64> %o2, %o1
  %p = getelementptr i8, ptr %a, <4 x i64> %o
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  %xs = add <4 x i32> %x, splat (i32 1)
  %xse = sext <4 x i32> %xs to <4 x i64>
  %q = getelementptr float, ptr %a, <4 x i64> %xse
  %bd = ptrtoint <4 x ptr> %q to <4 x i64>
  %w = call <4 x float> @__pseudo_gather64_float(<4 x i64> %bd, <4 x i32> %m)
  %r = fadd <4 x float> %v, %w
  ret <4 x float> %r
}

; A GEP sign extends an index narrower than the index width.
; CHECK-LABEL: memory accesses of @narrow_index:
; CHECK:      address: uniform base ptr %a (scalar, gep), one object
; CHECK-NEXT:   term: sext <4 x i32> %x * 4
define <4 x float> @narrow_index(ptr %a, <4 x i32> %x, <4 x i32> %m) {
  %p = getelementptr float, ptr %a, <4 x i32> %x
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  ret <4 x float> %v
}

; Per-lane pointers have no common base, whatever they are indexed with.
; CHECK-LABEL: memory accesses of @per_lane:
; CHECK:      address: per-lane base <4 x ptr> %q (argument, gep)
; CHECK-NEXT:   term: sext <4 x i32> %x * 4
; CHECK:      address: per-lane base <4 x ptr> %l (load)
; CHECK:      address: per-lane base <4 x ptr> %s (phi/select)
; CHECK:      address: per-lane base <4 x ptr> %c (call)
define <4 x float> @per_lane(<4 x ptr> %q, ptr %pp, <4 x i1> %b, <4 x i32> %x, <4 x i32> %m) {
  %e = sext <4 x i32> %x to <4 x i64>
  %p = getelementptr float, <4 x ptr> %q, <4 x i64> %e
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v0 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  %l = load <4 x ptr>, ptr %pp
  %ld = ptrtoint <4 x ptr> %l to <4 x i64>
  %v1 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ld, <4 x i32> %m)
  %s = select <4 x i1> %b, <4 x ptr> %q, <4 x ptr> %l
  %sd = ptrtoint <4 x ptr> %s to <4 x i64>
  %v2 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %sd, <4 x i32> %m)
  %c = call <4 x ptr> @make_ptrs()
  %cd = ptrtoint <4 x ptr> %c to <4 x i64>
  %v3 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %cd, <4 x i32> %m)
  %r0 = fadd <4 x float> %v0, %v1
  %r1 = fadd <4 x float> %v2, %v3
  %r = fadd <4 x float> %r0, %r1
  ret <4 x float> %r
}

; A pointer built from an integer is left to the integer recovery.
; CHECK-LABEL: memory accesses of @from_integer:
; CHECK:      address: integer base <4 x ptr> %q (inttoptr, gep)
; CHECK-NEXT:   term: sext <4 x i32> %x * 4
define <4 x float> @from_integer(<4 x i64> %n, <4 x i32> %x, <4 x i32> %m) {
  %q = inttoptr <4 x i64> %n to <4 x ptr>
  %e = sext <4 x i32> %x to <4 x i64>
  %p = getelementptr float, <4 x ptr> %q, <4 x i64> %e
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  ret <4 x float> %v
}

; Constant addresses into one global, as a constant vector of GEPs or as a
; vector GEP constant expression, and into a stack object; constants into two
; globals have no common base.
; CHECK-LABEL: memory accesses of @objects:
; CHECK:      address: uniform base ptr @g (scalar), one object of 256 bytes
; CHECK-NEXT:   lanes: 4 8 12 16 (i64)
; CHECK:      address: uniform base ptr @g (scalar, gep), one object of 256 bytes
; CHECK-NEXT:   lanes: 0 4 8 12 (i64)
; CHECK:      address: uniform base ptr %buf (scalar, gep), one object of 64 bytes
; CHECK-NEXT:   term: sext <4 x i32> %x * 4
; CHECK:      address: per-lane base <4 x ptr> <ptr @g, ptr @h, ptr @g, ptr @h> (other)
; CHECK:      address: uniform base ptr getelementptr (i8, ptr @g, i64 12) (scalar, gep), one object of 256 bytes at offset 12
; CHECK-NEXT:   lanes: 0 4 8 12 (i64)
define <4 x float> @objects(<4 x i32> %x, <4 x i32> %m) {
  %ad = ptrtoint <4 x ptr> <ptr getelementptr (i8, ptr @g, i64 4), ptr getelementptr (i8, ptr @g, i64 8), ptr getelementptr (i8, ptr @g, i64 12), ptr getelementptr (i8, ptr @g, i64 16)> to <4 x i64>
  %v0 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  %bd = ptrtoint <4 x ptr> getelementptr (float, ptr @g, <4 x i64> <i64 0, i64 1, i64 2, i64 3>) to <4 x i64>
  %v1 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %bd, <4 x i32> %m)
  %buf = alloca [16 x i32]
  %p = getelementptr i32, ptr %buf, <4 x i32> %x
  %cd = ptrtoint <4 x ptr> %p to <4 x i64>
  %v2 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %cd, <4 x i32> %m)
  %dd = ptrtoint <4 x ptr> <ptr @g, ptr @h, ptr @g, ptr @h> to <4 x i64>
  %v3 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %dd, <4 x i32> %m)
  %q = getelementptr i8, ptr getelementptr (i8, ptr @g, i64 12), <4 x i64> <i64 0, i64 4, i64 8, i64 12>
  %ed = ptrtoint <4 x ptr> %q to <4 x i64>
  %v4 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ed, <4 x i32> %m)
  %r0 = fadd <4 x float> %v0, %v1
  %r1 = fadd <4 x float> %v2, %v3
  %r2 = fadd <4 x float> %r0, %r1
  %r = fadd <4 x float> %r2, %v4
  ret <4 x float> %r
}

; An address space cast is looked through. A poison index lane, or integer
; addresses truncated from the pointers, are not decoded.
; CHECK-LABEL: memory accesses of @not_decoded:
; CHECK:      address: uniform base ptr addrspace(1) %a (scalar, gep), one object
; CHECK-NEXT:   term: sext <4 x i32> %x * 4
; CHECK:      address: unknown
; CHECK-NOT:  address:
; CHECK-LABEL: memory accesses of @constant_cast:
define <4 x float> @not_decoded(ptr addrspace(1) %a, ptr %b, <4 x i32> %x, <4 x i32> %m) {
  %p1 = getelementptr float, ptr addrspace(1) %a, <4 x i32> %x
  %p = addrspacecast <4 x ptr addrspace(1)> %p1 to <4 x ptr>
  %ad = ptrtoint <4 x ptr> %p to <4 x i64>
  %v0 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  %q = getelementptr float, ptr %b, <4 x i64> <i64 0, i64 poison, i64 2, i64 3>
  %bd = ptrtoint <4 x ptr> %q to <4 x i64>
  %v1 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %bd, <4 x i32> %m)
  %cd = ptrtoint <4 x ptr> %q to <4 x i32>
  %v2 = call <4 x float> @__pseudo_gather32_float(<4 x i32> %cd, <4 x i32> %m)
  %r0 = fadd <4 x float> %v0, %v1
  %r = fadd <4 x float> %r0, %v2
  ret <4 x float> %r
}

; Constant lanes cast from the 32-bit address space 271 are not decoded: the
; cast zero extends the 32-bit pointers, so null - 4 becomes 0xFFFFFFFC, not
; -4, also when the cast is inside a chain of casts that starts and ends in
; 64-bit spaces. A cast between spaces of the same widths is looked through.
; CHECK:      address: unknown
; CHECK:      address: unknown
; CHECK:      address: uniform base ptr addrspace(1) @g1 (scalar), one object of 256 bytes
; CHECK-NEXT:   lanes: 4 8 12 16 (i64)
define <4 x float> @constant_cast(<4 x i32> %m) {
  %ad = ptrtoint <4 x ptr> <
    ptr addrspacecast (ptr addrspace(271) getelementptr (i8, ptr addrspace(271) null, i32 -4) to ptr),
    ptr addrspacecast (ptr addrspace(271) getelementptr (i8, ptr addrspace(271) null, i32 -8) to ptr),
    ptr addrspacecast (ptr addrspace(271) getelementptr (i8, ptr addrspace(271) null, i32 1) to ptr),
    ptr addrspacecast (ptr addrspace(271) getelementptr (i8, ptr addrspace(271) null, i32 2) to ptr)
  > to <4 x i64>
  %v0 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %ad, <4 x i32> %m)
  %cd = ptrtoint <4 x ptr> <
    ptr addrspacecast (ptr addrspace(271) getelementptr (i8, ptr addrspace(271) addrspacecast (ptr getelementptr (i8, ptr null, i64 1) to ptr addrspace(271)), i32 -5) to ptr),
    ptr addrspacecast (ptr addrspace(271) getelementptr (i8, ptr addrspace(271) addrspacecast (ptr getelementptr (i8, ptr null, i64 1) to ptr addrspace(271)), i32 -9) to ptr),
    ptr addrspacecast (ptr addrspace(271) getelementptr (i8, ptr addrspace(271) addrspacecast (ptr getelementptr (i8, ptr null, i64 1) to ptr addrspace(271)), i32 0) to ptr),
    ptr addrspacecast (ptr addrspace(271) getelementptr (i8, ptr addrspace(271) addrspacecast (ptr getelementptr (i8, ptr null, i64 1) to ptr addrspace(271)), i32 1) to ptr)
  > to <4 x i64>
  %v2 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %cd, <4 x i32> %m)
  %bd = ptrtoint <4 x ptr> <
    ptr addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @g1, i64 4) to ptr),
    ptr addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @g1, i64 8) to ptr),
    ptr addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @g1, i64 12) to ptr),
    ptr addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @g1, i64 16) to ptr)
  > to <4 x i64>
  %v1 = call <4 x float> @__pseudo_gather64_float(<4 x i64> %bd, <4 x i32> %m)
  %r0 = fadd <4 x float> %v0, %v2
  %r = fadd <4 x float> %r0, %v1
  ret <4 x float> %r
}

define void @end() {
  ret void
}
