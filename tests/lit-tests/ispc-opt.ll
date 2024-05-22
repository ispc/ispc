; RUN: %{ispc-opt} --print-passes | FileCheck --check-prefix=CHECK-PRINT %s

; CHECK-PRINT: Module passes:
; CHECK-PRINT-NEXT:   make-internal-funcs-static
; CHECK-PRINT-NEXT: Function passes:
; CHECK-PRINT-NEXT:   gather-coalesce
; CHECK-PRINT-NEXT:   improve-memory-ops
; CHECK-PRINT-NEXT:   instruction-simplify
; CHECK-PRINT-NEXT:   intrinsics-opt
; CHECK-PRINT-NEXT:   is-compile-time-constant
; CHECK-PRINT-NEXT:   peephole
; CHECK-PRINT-NEXT:   replace-masked-memory-ops
; CHECK-PRINT-NEXT:   replace-pseudo-memory-ops
; CHECK-PRINT-NEXT:   replace-stdlib-shift

; RUN: %{ispc-opt} --target=neon-i32x4 --passes=peephole %s -o - | FileCheck --check-prefix=CHECK-PEEPHOLE %s

declare <4 x i16> @llvm.aarch64.neon.uhadd.v4i16(<4 x i16>, <4 x i16>)
define <4 x i16> @__avg_down_uint16(<4 x i16> %0, <4 x i16> %1) {
  %r = call <4 x i16> @llvm.aarch64.neon.uhadd.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %r
}

; CHECK-PEEPHOLE-LABEL: @test
; CHECK-PEEPHOLE-NEXT: entry:
; CHECK-PEEPHOLE-NEXT:   %a_ext = zext <4 x i16> %a to <4 x i32>
; CHECK-PEEPHOLE-NEXT:   %b_ext = zext <4 x i16> %b to <4 x i32>
; CHECK-PEEPHOLE-NEXT:   %add = add <4 x i32> %a_ext, %b_ext
; CHECK-PEEPHOLE-NEXT:   %div = udiv <4 x i32> %add, <i32 2, i32 2, i32 2, i32 2>
; CHECK-PEEPHOLE-NEXT:   %__avg_down_uint16 = call <4 x i16> @__avg_down_uint16(<4 x i16> %a, <4 x i16> %b)
; CHECK-PEEPHOLE-NEXT:   ret <4 x i16> %__avg_down_uint16
; CHECK-PEEPHOLE-NEXT: }

define <4 x i16> @test(<4 x i16> %a, <4 x i16> %b) {
entry:
  %a_ext = zext <4 x i16> %a to <4 x i32>
  %b_ext = zext <4 x i16> %b to <4 x i32>
  %add = add <4 x i32> %a_ext, %b_ext
  %div = udiv <4 x i32> %add, <i32 2, i32 2, i32 2, i32 2>
  %result = trunc <4 x i32> %div to <4 x i16>
  ret <4 x i16> %result
}


