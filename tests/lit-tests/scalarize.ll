; RUN: %{ispc-opt} --passes=scalarize %s -o - | FileCheck %s

; CHECK-LABEL: @add
; CHECK-NEXT:  %1 = add i32 %a, 1
; CHECK-NEXT:  %2 = insertelement <4 x i32> undef, i32 %1, i64 0
; CHECK-NEXT:  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:  ret <4 x i32> %3
define <4 x i32> @add(i32 %a) {
  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
  %sum = add <4 x i32> %op1, <i32 1, i32 poison, i32 poison, i32 poison>
  %shuffled = shufflevector <4 x i32> %sum, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %shuffled
}

; CHECK-LABEL: @sub
; CHECK-NEXT:  %1 = sub i32 1, %a
; CHECK-NEXT:  %2 = insertelement <4 x i32> undef, i32 %1, i64 0
; CHECK-NEXT:  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:  ret <4 x i32> %3
define <4 x i32> @sub(i32 %a) {
  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
  %binop = sub <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, %op1
  %shuffled = shufflevector <4 x i32> %binop, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %shuffled
}

; CHECK-LABEL: @undef
; CHECK-NEXT:  %1 = add i32 %a, 1
; CHECK-NEXT:  %2 = insertelement <4 x i32> undef, i32 %1, i64 0
; CHECK-NEXT:  %3 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:  ret <4 x i32> %3
define <4 x i32> @undef(i32 %a) {
  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
  %sum = add <4 x i32> %op1, <i32 1, i32 undef, i32 undef, i32 undef>
  %shuffled = shufflevector <4 x i32> %sum, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %shuffled
}

; CHECK-LABEL: @fmul
; CHECK-NEXT:   %1 = fmul float 1.000000e+00, %a
; CHECK-NEXT:   %2 = insertelement <4 x float> undef, float %1, i64 0
; CHECK-NEXT:   %3 = shufflevector <4 x float> %2, <4 x float> undef, <2 x i32> zeroinitializer
; CHECK-NEXT:   ret <2 x float> %3
define <2 x float> @fmul(float %a, float %b) {
  %op1 = insertelement <4 x float> undef, float %a, i32 0
  %binop = fmul <4 x float> <float 1.0, float poison, float poison, float poison>, %op1
  %shuffled = shufflevector <4 x float> %binop, <4 x float> undef, <2 x i32> zeroinitializer
  ret <2 x float> %shuffled
}

; cmp not supported
; CHECK-LABEL: @cmp
; CHECK-NEXT:  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
define <4 x i1> @cmp(i32 %a) {
  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
  %binop = icmp eq <4 x i32> %op1, <i32 1, i32 undef, i32 undef, i32 undef>
  %shuffled = shufflevector <4 x i1> %binop, <4 x i1> undef, <4 x i32> zeroinitializer
  ret <4 x i1> %shuffled
}

; not supported any position of non-posion value except the first one
; CHECK-LABEL: @nonzeroelement
; CHECK-NEXT:  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
define <4 x i32> @nonzeroelement(i32 %a) {
  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
  %sum = add <4 x i32> %op1, <i32 poison, i32 poison, i32 poison, i32 1>
  %shuffled = shufflevector <4 x i32> %sum, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %shuffled
}

; not supported non-const second operand
; CHECK-LABEL: @nonconst
; CHECK-NEXT:  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
; CHECK-NEXT:  %op2 = insertelement <4 x i32> undef, i32 %b, i32 0
define <4 x i32> @nonconst(i32 %a, i32 %b) {
  %op1 = insertelement <4 x i32> undef, i32 %a, i32 0
  %op2 = insertelement <4 x i32> undef, i32 %b, i32 0
  %sum = add <4 x i32> %op1, %op2
  %shuffled = shufflevector <4 x i32> %sum, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %shuffled
}

; not supported for values across phi-nodes
define <4 x i32> @phi(i32 %a, i32 %b, i32 %c) {
  %cmp = icmp slt i32 %c, 0
  br i1 %cmp, label %block_a, label %block_b

  block_a:
    %incoming_a = insertelement <4 x i32> undef, i32 %a, i32 0
    br label %exit

  block_b:
    %incoming_b = insertelement <4 x i32> undef, i32 %b, i32 0
    br label %exit

  exit:
    %op1 = phi <4 x i32> [ %incoming_a, %block_a ], [ %incoming_b, %block_b ]
    %binop = add <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, %op1
    %shuffled = shufflevector <4 x i32> %binop, <4 x i32> poison, <4 x i32> zeroinitializer
    ret <4 x i32> %shuffled
}

