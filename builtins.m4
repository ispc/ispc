;;  Copyright (c) 2010-2011, Intel Corporation
;;  All rights reserved.
;;
;;  Redistribution and use in source and binary forms, with or without
;;  modification, are permitted provided that the following conditions are
;;  met:
;;
;;    * Redistributions of source code must retain the above copyright
;;      notice, this list of conditions and the following disclaimer.
;;
;;    * Redistributions in binary form must reproduce the above copyright
;;      notice, this list of conditions and the following disclaimer in the
;;      documentation and/or other materials provided with the distribution.
;;
;;    * Neither the name of Intel Corporation nor the names of its
;;      contributors may be used to endorse or promote products derived from
;;      this software without specific prior written permission.
;;
;;
;;   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
;;   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
;;   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
;;   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
;;   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
;;   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
;;   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
;;   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
;;   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;;   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;;   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  

;; This file provides a variety of macros used to generate LLVM bitcode
;; parametrized in various ways.  Implementations of the standard library
;; builtins for various targets can use macros from this file to simplify
;; generating code for their implementations of those builtins.

declare i1 @__is_compile_time_constant_uniform_int32(i32)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; Helper macro for calling various SSE instructions for scalar values
;; but where the instruction takes a vector parameter.
;; $1 : name of variable to put the final value in
;; $2 : vector width of the target
;; $3 : scalar type of the operand
;; $4 : SSE intrinsic name
;; $5 : variable name that has the scalar value
;; For example, the following call causes the variable %ret to have
;; the result of a call to sqrtss with the scalar value in %0
;;  sse_unary_scalar(ret, 4, float, @llvm.x86.sse.sqrt.ss, %0)

define(`sse_unary_scalar', `
  %$1_vec = insertelement <$2 x $3> undef, $3 $5, i32 0
  %$1_val = call <$2 x $3> $4(<$2 x $3> %$1_vec)
  %$1 = extractelement <$2 x $3> %$1_val, i32 0
')

;; Similar to `sse_unary_scalar', this helper macro is for calling binary
;; SSE instructions with scalar values, 
;; $1: name of variable to put the result in
;; $2: vector width of the target
;; $3: scalar type of the operand
;; $4 : SSE intrinsic name
;; $5 : variable name that has the first scalar operand
;; $6 : variable name that has the second scalar operand

define(`sse_binary_scalar', `
  %$1_veca = insertelement <$2 x $3> undef, $3 $5, i32 0
  %$1_vecb = insertelement <$2 x $3> undef, $3 $6, i32 0
  %$1_val = call <$2 x $3> $4(<$2 x $3> %$1_veca, <$2 x $3> %$1_vecb)
  %$1 = extractelement <$2 x $3> %$1_val, i32 0
')

;; Do a reduction over a 4-wide vector
;; $1: type of final scalar result
;; $2: 4-wide function that takes 2 4-wide operands and returns the 
;;     element-wise reduction
;; $3: scalar function that takes two scalar operands and returns
;;     the final reduction

define(`reduce4', `
  %v1 = shufflevector <4 x $1> %0, <4 x $1> undef,
                      <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %m1 = call <4 x $1> $2(<4 x $1> %v1, <4 x $1> %0)
  %m1a = extractelement <4 x $1> %m1, i32 0
  %m1b = extractelement <4 x $1> %m1, i32 1
  %m = call $1 $3($1 %m1a, $1 %m1b)
  ret $1 %m
'
)

;; Similar to `reduce4', do a reduction over an 8-wide vector
;; $1: type of final scalar result
;; $2: 8-wide function that takes 2 8-wide operands and returns the 
;;     element-wise reduction
;; $3: scalar function that takes two scalar operands and returns
;;     the final reduction

define(`reduce8', `
  %v1 = shufflevector <8 x $1> %0, <8 x $1> undef,
        <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %m1 = call <8 x $1> $2(<8 x $1> %v1, <8 x $1> %0)
  %v2 = shufflevector <8 x $1> %m1, <8 x $1> undef,
        <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %m2 = call <8 x $1> $2(<8 x $1> %v2, <8 x $1> %m1)
  %m2a = extractelement <8 x $1> %m2, i32 0
  %m2b = extractelement <8 x $1> %m2, i32 1
  %m = call $1 $3($1 %m2a, $1 %m2b)
  ret $1 %m
'
)

define(`reduce16', `
  %v1 = shufflevector <16 x $1> %0, <16 x $1> undef,
        <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                    i32 undef, i32 undef, i32 undef, i32 undef,
                    i32 undef, i32 undef, i32 undef, i32 undef>
  %m1 = call <16 x $1> $2(<16 x $1> %v1, <16 x $1> %0)
  %v2 = shufflevector <16 x $1> %m1, <16 x $1> undef,
        <16 x i32> <i32 4, i32 5, i32 6, i32 7,
                    i32 undef, i32 undef, i32 undef, i32 undef,
                    i32 undef, i32 undef, i32 undef, i32 undef,
                    i32 undef, i32 undef, i32 undef, i32 undef>
  %m2 = call <16 x $1> $2(<16 x $1> %v2, <16 x $1> %m1)
  %v3 = shufflevector <16 x $1> %m2, <16 x $1> undef,
        <16 x i32> <i32 2, i32 3, i32 undef, i32 undef,
                    i32 undef, i32 undef, i32 undef, i32 undef,
                    i32 undef, i32 undef, i32 undef, i32 undef,
                    i32 undef, i32 undef, i32 undef, i32 undef>
  %m3 = call <16 x $1> $2(<16 x $1> %v3, <16 x $1> %m2)

  %m3a = extractelement <16 x $1> %m3, i32 0
  %m3b = extractelement <16 x $1> %m3, i32 1
  %m = call $1 $3($1 %m3a, $1 %m3b)
  ret $1 %m
'
)

;; Do an reduction over an 8-wide vector, using a vector reduction function
;; that only takes 4-wide vectors
;; $1: type of final scalar result
;; $2: 4-wide function that takes 2 4-wide operands and returns the 
;;     element-wise reduction
;; $3: scalar function that takes two scalar operands and returns
;;     the final reduction

define(`reduce8by4', `
  %v1 = shufflevector <8 x $1> %0, <8 x $1> undef,
        <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v2 = shufflevector <8 x $1> %0, <8 x $1> undef,
        <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %m1 = call <4 x $1> $2(<4 x $1> %v1, <4 x $1> %v2)
  %v3 = shufflevector <4 x $1> %m1, <4 x $1> undef,
        <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %m2 = call <4 x $1> $2(<4 x $1> %v3, <4 x $1> %m1)
  %m2a = extractelement <4 x $1> %m2, i32 0
  %m2b = extractelement <4 x $1> %m2, i32 1
  %m = call $1 $3($1 %m2a, $1 %m2b)
  ret $1 %m
'
)


;; Apply a unary function to the 4-vector in %0, return the vector result.
;; $1: scalar type of result
;; $2: name of scalar function to call

define(`unary1to4', `
  %v_0 = extractelement <4 x $1> %0, i32 0
  %r_0 = call $1 $2($1 %v_0)
  %ret_0 = insertelement <4 x $1> undef, $1 %r_0, i32 0
  %v_1 = extractelement <4 x $1> %0, i32 1
  %r_1 = call $1 $2($1 %v_1)
  %ret_1 = insertelement <4 x $1> %ret_0, $1 %r_1, i32 1
  %v_2 = extractelement <4 x $1> %0, i32 2
  %r_2 = call $1 $2($1 %v_2)
  %ret_2 = insertelement <4 x $1> %ret_1, $1 %r_2, i32 2
  %v_3 = extractelement <4 x $1> %0, i32 3
  %r_3 = call $1 $2($1 %v_3)
  %ret_3 = insertelement <4 x $1> %ret_2, $1 %r_3, i32 3
  ret <4 x $1> %ret_3
')

;; Given a unary function that takes a 2-wide vector and a 4-wide vector
;; that we'd like to apply it to, extract 2 2-wide vectors from the 4-wide
;; vector, apply it, and return the corresponding 4-wide vector result
;; $1: name of variable into which the final result should go
;; $2: scalar type of the vector elements
;; $3: 2-wide unary vector function to apply
;; $4: 4-wide operand value

define(`unary2to4', `
  %$1_0 = shufflevector <4 x $2> $4, <4 x $2> undef, <2 x i32> <i32 0, i32 1>
  %v$1_0 = call <2 x $2> $3(<2 x $2> %$1_0)
  %$1_1 = shufflevector <4 x $2> $4, <4 x $2> undef, <2 x i32> <i32 2, i32 3>
  %v$1_1 = call <2 x $2> $3(<2 x $2> %$1_1)
  %$1 = shufflevector <2 x $2> %v$1_0, <2 x $2> %v$1_1, 
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
'
)

;; Similar to `unary2to4', this applies a 2-wide binary function to two 4-wide
;; vector operands
;; $1: name of variable into which the final result should go
;; $2: scalar type of the vector elements
;; $3: 2-wide binary vector function to apply
;; $4: First 4-wide operand value
;; $5: Second 4-wide operand value

define(`binary2to4', `
%$1_0a = shufflevector <4 x $2> $4, <4 x $2> undef, <2 x i32> <i32 0, i32 1>
%$1_0b = shufflevector <4 x $2> $5, <4 x $2> undef, <2 x i32> <i32 0, i32 1>
%v$1_0 = call <2 x $2> $3(<2 x $2> %$1_0a, <2 x $2> %$1_0b)
%$1_1a = shufflevector <4 x $2> $4, <4 x $2> undef, <2 x i32> <i32 2, i32 3>
%$1_1b = shufflevector <4 x $2> $5, <4 x $2> undef, <2 x i32> <i32 2, i32 3>
%v$1_1 = call <2 x $2> $3(<2 x $2> %$1_1a, <2 x $2> %$1_1b)
%$1 = shufflevector <2 x $2> %v$1_0, <2 x $2> %v$1_1, 
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
'
)

;; Similar to `unary2to4', this maps a 4-wide unary function to an 8-wide 
;; vector operand
;; $1: name of variable into which the final result should go
;; $2: scalar type of the vector elements
;; $3: 4-wide unary vector function to apply
;; $4: 8-wide operand value

define(`unary4to8', `
  %$1_0 = shufflevector <8 x $2> $4, <8 x $2> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v$1_0 = call <4 x $2> $3(<4 x $2> %$1_0)
  %$1_1 = shufflevector <8 x $2> $4, <8 x $2> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v$1_1 = call <4 x $2> $3(<4 x $2> %$1_1)
  %$1 = shufflevector <4 x $2> %v$1_0, <4 x $2> %v$1_1, 
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
'
)

define(`unary4to16', `
  %$1_0 = shufflevector <16 x $2> $4, <16 x $2> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %v$1_0 = call <4 x $2> $3(<4 x $2> %$1_0)
  %$1_1 = shufflevector <16 x $2> $4, <16 x $2> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %v$1_1 = call <4 x $2> $3(<4 x $2> %$1_1)
  %$1_2 = shufflevector <16 x $2> $4, <16 x $2> undef, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  %v$1_2 = call <4 x $2> $3(<4 x $2> %$1_2)
  %$1_3 = shufflevector <16 x $2> $4, <16 x $2> undef, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
  %v$1_3 = call <4 x $2> $3(<4 x $2> %$1_3)

  %$1a = shufflevector <4 x $2> %v$1_0, <4 x $2> %v$1_1, 
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %$1b = shufflevector <4 x $2> %v$1_2, <4 x $2> %v$1_3, 
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %$1 = shufflevector <8 x $2> %$1a, <8 x $2> %$1b,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
'
)

;; And so forth...
;; $1: name of variable into which the final result should go
;; $2: scalar type of the vector elements
;; $3: 8-wide unary vector function to apply
;; $4: 16-wide operand value

define(`unary8to16', `
  %$1_0 = shufflevector <16 x $2> $4, <16 x $2> undef,
             <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v$1_0 = call <8 x $2> $3(<8 x $2> %$1_0)
  %$1_1 = shufflevector <16 x $2> $4, <16 x $2> undef,
             <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %v$1_1 = call <8 x $2> $3(<8 x $2> %$1_1)
  %$1 = shufflevector <8 x $2> %v$1_0, <8 x $2> %v$1_1, 
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
'
)

;; And along the lines of `binary2to4', this maps a 4-wide binary function to
;; two 8-wide vector operands
;; $1: name of variable into which the final result should go
;; $2: scalar type of the vector elements
;; $3: 4-wide unary vector function to apply
;; $4: First 8-wide operand value
;; $5: Second 8-wide operand value

define(`binary4to8', `
%$1_0a = shufflevector <8 x $2> $4, <8 x $2> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%$1_0b = shufflevector <8 x $2> $5, <8 x $2> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%v$1_0 = call <4 x $2> $3(<4 x $2> %$1_0a, <4 x $2> %$1_0b)
%$1_1a = shufflevector <8 x $2> $4, <8 x $2> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
%$1_1b = shufflevector <8 x $2> $5, <8 x $2> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
%v$1_1 = call <4 x $2> $3(<4 x $2> %$1_1a, <4 x $2> %$1_1b)
%$1 = shufflevector <4 x $2> %v$1_0, <4 x $2> %v$1_1, 
         <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
'
)

define(`binary8to16', `
%$1_0a = shufflevector <16 x $2> $4, <16 x $2> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
%$1_0b = shufflevector <16 x $2> $5, <16 x $2> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
%v$1_0 = call <8 x $2> $3(<8 x $2> %$1_0a, <8 x $2> %$1_0b)
%$1_1a = shufflevector <16 x $2> $4, <16 x $2> undef,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
%$1_1b = shufflevector <16 x $2> $5, <16 x $2> undef,
          <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
%v$1_1 = call <8 x $2> $3(<8 x $2> %$1_1a, <8 x $2> %$1_1b)
%$1 = shufflevector <8 x $2> %v$1_0, <8 x $2> %v$1_1, 
         <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                     i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
'
)

define(`binary4to16', `
%$1_0a = shufflevector <16 x $2> $4, <16 x $2> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%$1_0b = shufflevector <16 x $2> $5, <16 x $2> undef,
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%r$1_0 = call <4 x $2> $3(<4 x $2> %$1_0a, <4 x $2> %$1_0b) 

%$1_1a = shufflevector <16 x $2> $4, <16 x $2> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
%$1_1b = shufflevector <16 x $2> $5, <16 x $2> undef,
          <4 x i32> <i32 4, i32 5, i32 6, i32 7>
%r$1_1 = call <4 x $2> $3(<4 x $2> %$1_1a, <4 x $2> %$1_1b) 

%$1_2a = shufflevector <16 x $2> $4, <16 x $2> undef,
          <4 x i32> <i32 8, i32 9, i32 10, i32 11>
%$1_2b = shufflevector <16 x $2> $5, <16 x $2> undef,
          <4 x i32> <i32 8, i32 9, i32 10, i32 11>
%r$1_2 = call <4 x $2> $3(<4 x $2> %$1_2a, <4 x $2> %$1_2b) 

%$1_3a = shufflevector <16 x $2> $4, <16 x $2> undef,
          <4 x i32> <i32 12, i32 13, i32 14, i32 15>
%$1_3b = shufflevector <16 x $2> $5, <16 x $2> undef,
          <4 x i32> <i32 12, i32 13, i32 14, i32 15>
%r$1_3 = call <4 x $2> $3(<4 x $2> %$1_3a, <4 x $2> %$1_3b)

%r$1_01 = shufflevector <4 x $2> %r$1_0, <4 x $2> %r$1_1, 
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
%r$1_23 = shufflevector <4 x $2> %r$1_2, <4 x $2> %r$1_3, 
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>

%$1 = shufflevector <8 x $2> %r$1_01, <8 x $2> %r$1_23, 
          <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                      i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
')

;; Maps a 2-wide unary function to an 8-wide vector operand, returning an 
;; 8-wide vector result
;; $1: name of variable into which the final result should go
;; $2: scalar type of the vector elements
;; $3: 2-wide unary vector function to apply
;; $4: 8-wide operand value

define(`unary2to8', `
  %$1_0 = shufflevector <8 x $2> $4, <8 x $2> undef, <2 x i32> <i32 0, i32 1>
  %v$1_0 = call <2 x $2> $3(<2 x $2> %$1_0)
  %$1_1 = shufflevector <8 x $2> $4, <8 x $2> undef, <2 x i32> <i32 2, i32 3>
  %v$1_1 = call <2 x $2> $3(<2 x $2> %$1_1)
  %$1_2 = shufflevector <8 x $2> $4, <8 x $2> undef, <2 x i32> <i32 4, i32 5>
  %v$1_2 = call <2 x $2> $3(<2 x $2> %$1_2)
  %$1_3 = shufflevector <8 x $2> $4, <8 x $2> undef, <2 x i32> <i32 6, i32 7>
  %v$1_3 = call <2 x $2> $3(<2 x $2> %$1_3)
  %$1a = shufflevector <2 x $2> %v$1_0, <2 x $2> %v$1_1, 
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %$1b = shufflevector <2 x $2> %v$1_2, <2 x $2> %v$1_3, 
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %$1 = shufflevector <4 x $2> %$1a, <4 x $2> %$1b,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>           
'
)

;; Maps an 2-wide binary function to two 8-wide vector operands
;; $1: name of variable into which the final result should go
;; $2: scalar type of the vector elements
;; $3: 2-wide unary vector function to apply
;; $4: First 8-wide operand value
;; $5: Second 8-wide operand value

define(`binary2to8', `
  %$1_0a = shufflevector <8 x $2> $4, <8 x $2> undef, <2 x i32> <i32 0, i32 1>
  %$1_0b = shufflevector <8 x $2> $5, <8 x $2> undef, <2 x i32> <i32 0, i32 1>
  %v$1_0 = call <2 x $2> $3(<2 x $2> %$1_0a, <2 x $2> %$1_0b)
  %$1_1a = shufflevector <8 x $2> $4, <8 x $2> undef, <2 x i32> <i32 2, i32 3>
  %$1_1b = shufflevector <8 x $2> $5, <8 x $2> undef, <2 x i32> <i32 2, i32 3>
  %v$1_1 = call <2 x $2> $3(<2 x $2> %$1_1a, <2 x $2> %$1_1b)
  %$1_2a = shufflevector <8 x $2> $4, <8 x $2> undef, <2 x i32> <i32 4, i32 5>
  %$1_2b = shufflevector <8 x $2> $5, <8 x $2> undef, <2 x i32> <i32 4, i32 5>
  %v$1_2 = call <2 x $2> $3(<2 x $2> %$1_2a, <2 x $2> %$1_2b)
  %$1_3a = shufflevector <8 x $2> $4, <8 x $2> undef, <2 x i32> <i32 6, i32 7>
  %$1_3b = shufflevector <8 x $2> $5, <8 x $2> undef, <2 x i32> <i32 6, i32 7>
  %v$1_3 = call <2 x $2> $3(<2 x $2> %$1_3a, <2 x $2> %$1_3b)

  %$1a = shufflevector <2 x $2> %v$1_0, <2 x $2> %v$1_1, 
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %$1b = shufflevector <2 x $2> %v$1_2, <2 x $2> %v$1_3, 
           <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %$1 = shufflevector <4 x $2> %$1a, <4 x $2> %$1b,
           <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>           
'
)

;; The unary SSE round intrinsic takes a second argument that encodes the
;; rounding mode.  This macro makes it easier to apply the 4-wide roundps
;; to 8-wide vector operands
;; $1: value to be rounded
;; $2: integer encoding of rounding mode
;; FIXME: this just has a ret statement at the end to return the result,
;; which is inconsistent with the macros above 

define(`round4to8', `
%v0 = shufflevector <8 x float> $1, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%v1 = shufflevector <8 x float> $1, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
%r0 = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %v0, i32 $2)
%r1 = call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %v1, i32 $2)
%ret = shufflevector <4 x float> %r0, <4 x float> %r1, 
         <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
ret <8 x float> %ret
'
)

define(`round8to16', `
%v0 = shufflevector <16 x float> $1, <16 x float> undef,
        <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
%v1 = shufflevector <16 x float> $1, <16 x float> undef,
        <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
%r0 = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %v0, i32 $2)
%r1 = call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %v1, i32 $2)
%ret = shufflevector <8 x float> %r0, <8 x float> %r1, 
         <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                     i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
ret <16 x float> %ret
'
)

define(`round4to8double', `
%v0 = shufflevector <8 x double> $1, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%v1 = shufflevector <8 x double> $1, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
%r0 = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %v0, i32 $2)
%r1 = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %v1, i32 $2)
%ret = shufflevector <4 x double> %r0, <4 x double> %r1, 
         <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
ret <8 x double> %ret
'
)

; and similarly for doubles...

define(`round2to4double', `
%v0 = shufflevector <4 x double> $1, <4 x double> undef, <2 x i32> <i32 0, i32 1>
%v1 = shufflevector <4 x double> $1, <4 x double> undef, <2 x i32> <i32 2, i32 3>
%r0 = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %v0, i32 $2)
%r1 = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %v1, i32 $2)
%ret = shufflevector <2 x double> %r0, <2 x double> %r1, 
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
ret <4 x double> %ret
'
)

define(`round2to8double', `
%v0 = shufflevector <8 x double> $1, <8 x double> undef, <2 x i32> <i32 0, i32 1>
%v1 = shufflevector <8 x double> $1, <8 x double> undef, <2 x i32> <i32 2, i32 3>
%v2 = shufflevector <8 x double> $1, <8 x double> undef, <2 x i32> <i32 4, i32 5>
%v3 = shufflevector <8 x double> $1, <8 x double> undef, <2 x i32> <i32 6, i32 7>
%r0 = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %v0, i32 $2)
%r1 = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %v1, i32 $2)
%r2 = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %v2, i32 $2)
%r3 = call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %v3, i32 $2)
%ret0 = shufflevector <2 x double> %r0, <2 x double> %r1, 
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%ret1 = shufflevector <2 x double> %r2, <2 x double> %r3, 
          <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%ret = shufflevector <4 x double> %ret0, <4 x double> %ret1,
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
ret <8 x double> %ret
'
)

define(`round4to16double', `
%v0 = shufflevector <16 x double> $1, <16 x double> undef,
         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
%v1 = shufflevector <16 x double> $1, <16 x double> undef,
         <4 x i32> <i32 4, i32 5, i32 6, i32 7>
%v2 = shufflevector <16 x double> $1, <16 x double> undef,
         <4 x i32> <i32 8, i32 9, i32 10, i32 11>
%v3 = shufflevector <16 x double> $1, <16 x double> undef,
         <4 x i32> <i32 12, i32 13, i32 14, i32 15>
%r0 = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %v0, i32 $2)
%r1 = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %v1, i32 $2)
%r2 = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %v2, i32 $2)
%r3 = call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %v3, i32 $2)
%ret0 = shufflevector <4 x double> %r0, <4 x double> %r1, 
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
%ret1 = shufflevector <4 x double> %r2, <4 x double> %r3, 
          <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
%ret = shufflevector <8 x double> %ret0, <8 x double> %ret1,
          <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                      i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
ret <16 x double> %ret
'
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; forloop macro

divert(`-1')
# forloop(var, from, to, stmt) - improved version:
#   works even if VAR is not a strict macro name
#   performs sanity check that FROM is larger than TO
#   allows complex numerical expressions in TO and FROM
define(`forloop', `ifelse(eval(`($3) >= ($2)'), `1',
  `pushdef(`$1', eval(`$2'))_$0(`$1',
    eval(`$3'), `$4')popdef(`$1')')')
define(`_forloop',
  `$3`'ifelse(indir(`$1'), `$2', `',
    `define(`$1', incr(indir(`$1')))$0($@)')')
divert`'dnl

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; stdlib_core
;;
;; This macro defines a bunch of helper routines that only depend on the
;; target's vector width, which it takes as its first parameter.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define(`shuffles', `
define internal <$1 x $2> @__broadcast_$3(<$1 x $2>, i32) nounwind readnone alwaysinline {
  %v = extractelement <$1 x $2> %0, i32 %1
  %r_0 = insertelement <$1 x $2> undef, $2 %v, i32 0
forloop(i, 1, eval($1-1), `  %r_`'i = insertelement <$1 x $2> %r_`'eval(i-1), $2 %v, i32 i
')
  ret <$1 x $2> %r_`'eval($1-1)
}

define internal <$1 x $2> @__rotate_$3(<$1 x $2>, i32) nounwind readnone alwaysinline {
  %isc = call i1 @__is_compile_time_constant_uniform_int32(i32 %1)
  br i1 %isc, label %is_const, label %not_const

is_const:
  ; though verbose, this turms into tight code if %1 is a constant
forloop(i, 0, eval($1-1), `  
  %delta_`'i = add i32 %1, i
  %delta_clamped_`'i = and i32 %delta_`'i, eval($1-1)
  %v_`'i = extractelement <$1 x $2> %0, i32 %delta_clamped_`'i')

  %ret_0 = insertelement <$1 x $2> undef, $2 %v_0, i32 0
forloop(i, 1, eval($1-1), `  %ret_`'i = insertelement <$1 x $2> %ret_`'eval(i-1), $2 %v_`'i, i32 i
')
  ret <$1 x $2> %ret_`'eval($1-1)

not_const:
  ; store two instances of the vector into memory
  %ptr = alloca <$1 x $2>, i32 2
  %ptr0 = getelementptr <$1 x $2> * %ptr, i32 0
  store <$1 x $2> %0, <$1 x $2> * %ptr0
  %ptr1 = getelementptr <$1 x $2> * %ptr, i32 1
  store <$1 x $2> %0, <$1 x $2> * %ptr1

  ; compute offset in [0,vectorwidth-1], then index into the doubled-up vector
  %offset = and i32 %1, eval($1-1)
  %ptr_as_elt_array = bitcast <$1 x $2> * %ptr to [eval(2*$1) x $2] *
  %load_ptr = getelementptr [eval(2*$1) x $2] * %ptr_as_elt_array, i32 0, i32 %offset
  %load_ptr_vec = bitcast $2 * %load_ptr to <$1 x $2> *
  %result = load <$1 x $2> * %load_ptr_vec, align $4
  ret <$1 x $2> %result
}

define internal <$1 x $2> @__shuffle_$3(<$1 x $2>, <$1 x i32>) nounwind readnone alwaysinline {
forloop(i, 0, eval($1-1), `  
  %index_`'i = extractelement <$1 x i32> %1, i32 i')
forloop(i, 0, eval($1-1), `  
  %v_`'i = extractelement <$1 x $2> %0, i32 %index_`'i')

  %ret_0 = insertelement <$1 x $2> undef, $2 %v_0, i32 0
forloop(i, 1, eval($1-1), `  %ret_`'i = insertelement <$1 x $2> %ret_`'eval(i-1), $2 %v_`'i, i32 i
')
  ret <$1 x $2> %ret_`'eval($1-1)
}

define internal <$1 x $2> @__shuffle2_$3(<$1 x $2>, <$1 x $2>, <$1 x i32>) nounwind readnone alwaysinline {
  %v2 = shufflevector <$1 x $2> %0, <$1 x $2> %1, <eval(2*$1) x i32> <
      forloop(i, 0, eval(2*$1-2), `i32 i, ') i32 eval(2*$1-1)
  >
forloop(i, 0, eval($1-1), `  
  %index_`'i = extractelement <$1 x i32> %2, i32 i')

  %isc = call i1 @__is_compile_time_constant_varying_int32(<$1 x i32> %2)
  br i1 %isc, label %is_const, label %not_const

is_const:
  ; extract from the requested lanes and insert into the result; LLVM turns
  ; this into good code in the end
forloop(i, 0, eval($1-1), `  
  %v_`'i = extractelement <eval(2*$1) x $2> %v2, i32 %index_`'i')

  %ret_0 = insertelement <$1 x $2> undef, $2 %v_0, i32 0
forloop(i, 1, eval($1-1), `  %ret_`'i = insertelement <$1 x $2> %ret_`'eval(i-1), $2 %v_`'i, i32 i
')
  ret <$1 x $2> %ret_`'eval($1-1)

not_const:
  ; otherwise store the two vectors onto the stack and then use the given
  ; permutation vector to get indices into that array...
  %ptr = alloca <eval(2*$1) x $2>
  store <eval(2*$1) x $2> %v2, <eval(2*$1) x $2> * %ptr
  %baseptr = bitcast <eval(2*$1) x $2> * %ptr to $2 *

  %ptr_0 = getelementptr $2 * %baseptr, i32 %index_0
  %val_0 = load $2 * %ptr_0
  %result_0 = insertelement <$1 x $2> undef, $2 %val_0, i32 0

forloop(i, 1, eval($1-1), `  
  %ptr_`'i = getelementptr $2 * %baseptr, i32 %index_`'i
  %val_`'i = load $2 * %ptr_`'i
  %result_`'i = insertelement <$1 x $2> %result_`'eval(i-1), $2 %val_`'i, i32 i
')

  ret <$1 x $2> %result_`'eval($1-1)
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; global_atomic
;; Defines the implementation of a function that handles the mapping from
;; an ispc atomic function to the underlying LLVM intrinsics.  Specifically,
;; the function handles loooping over the active lanes, calling the underlying
;; scalar atomic intrinsic for each one, and assembling the vector result.
;;
;; Takes four parameters:
;; $1: vector width of the target
;; $2: operation being performed (w.r.t. LLVM atomic intrinsic names)
;;     (add, sub...)
;; $3: return type of the LLVM atomic (e.g. i32)
;; $4: return type of the LLVM atomic type, in ispc naming paralance (e.g. int32)

define(`global_atomic', `

declare $3 @llvm.atomic.load.$2.$3.p0$3($3 * %ptr, $3 %delta)

define internal <$1 x $3> @__atomic_$2_$4_global($3 * %ptr, <$1 x $3> %val,
                                                 <$1 x i32> %mask) nounwind alwaysinline {
  %rptr = alloca <$1 x $3>
  %rptr32 = bitcast <$1 x $3> * %rptr to $3 *

  per_lane($1, <$1 x i32> %mask, `
   %v_LANE_ID = extractelement <$1 x $3> %val, i32 LANE
   %r_LANE_ID = call $3 @llvm.atomic.load.$2.$3.p0$3($3 * %ptr, $3 %v_LANE_ID)
   %rp_LANE_ID = getelementptr $3 * %rptr32, i32 LANE
   store $3 %r_LANE_ID, $3 * %rp_LANE_ID')

  %r = load <$1 x $3> * %rptr
  ret <$1 x $3> %r
}
')

;; Macro to declare the function that implements the swap atomic.  
;; Takes three parameters:
;; $1: vector width of the target
;; $2: llvm type of the vector elements (e.g. i32)
;; $3: ispc type of the elements (e.g. int32)

define(`global_swap', `

declare $2 @llvm.atomic.swap.$2.p0$2($2 * %ptr, $2 %val)

define internal <$1 x $2> @__atomic_swap_$3_global($2* %ptr, <$1 x $2> %val,
                                                   <$1 x i32> %mask) nounwind alwaysinline {
  %rptr = alloca <$1 x $2>
  %rptr32 = bitcast <$1 x $2> * %rptr to $2 *

  per_lane($1, <$1 x i32> %mask, `
   %val_LANE_ID = extractelement <$1 x $2> %val, i32 LANE
   %r_LANE_ID = call $2 @llvm.atomic.swap.$2.p0$2($2 * %ptr, $2 %val_LANE_ID)
   %rp_LANE_ID = getelementptr $2 * %rptr32, i32 LANE
   store $2 %r_LANE_ID, $2 * %rp_LANE_ID')

  %r = load <$1 x $2> * %rptr
  ret <$1 x $2> %r
}
')


;; Similarly, macro to declare the function that implements the compare/exchange
;; atomic.  Takes three parameters:
;; $1: vector width of the target
;; $2: llvm type of the vector elements (e.g. i32)
;; $3: ispc type of the elements (e.g. int32)

define(`global_atomic_exchange', `

declare $2 @llvm.atomic.cmp.swap.$2.p0$2($2 * %ptr, $2 %cmp, $2 %val)

define internal <$1 x $2> @__atomic_compare_exchange_$3_global($2* %ptr, <$1 x $2> %cmp,
                               <$1 x $2> %val, <$1 x i32> %mask) nounwind alwaysinline {
  %rptr = alloca <$1 x $2>
  %rptr32 = bitcast <$1 x $2> * %rptr to $2 *

  per_lane($1, <$1 x i32> %mask, `
   %cmp_LANE_ID = extractelement <$1 x $2> %cmp, i32 LANE
   %val_LANE_ID = extractelement <$1 x $2> %val, i32 LANE
   %r_LANE_ID = call $2 @llvm.atomic.cmp.swap.$2.p0$2($2 * %ptr, $2 %cmp_LANE_ID,
                                                         $2 %val_LANE_ID)
   %rp_LANE_ID = getelementptr $2 * %rptr32, i32 LANE
   store $2 %r_LANE_ID, $2 * %rp_LANE_ID')

  %r = load <$1 x $2> * %rptr
  ret <$1 x $2> %r
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetch definitions

; prefetch has a new parameter in LLVM3.0, to distinguish between instruction
; and data caches--the declaration is now:
; declare void @llvm.prefetch(i8* nocapture %ptr, i32 %readwrite, i32 %locality,
;                             i32 %cachetype)  (cachetype 1 == data cache)
; however, the version below seems to still work...

declare void @llvm.prefetch(i8* nocapture %ptr, i32 %readwrite, i32 %locality)

define(`prefetch_read', `
define internal void @__prefetch_read_1_$1($2 *) alwaysinline {
  %ptr8 = bitcast $2 * %0 to i8 *
  call void @llvm.prefetch(i8 * %ptr8, i32 0, i32 3)
  ret void
}
define internal void @__prefetch_read_2_$1($2 *) alwaysinline {
  %ptr8 = bitcast $2 * %0 to i8 *
  call void @llvm.prefetch(i8 * %ptr8, i32 0, i32 2)
  ret void
}
define internal void @__prefetch_read_3_$1($2 *) alwaysinline {
  %ptr8 = bitcast $2 * %0 to i8 *
  call void @llvm.prefetch(i8 * %ptr8, i32 0, i32 1)
  ret void
}
define internal void @__prefetch_read_nt_$1($2 *) alwaysinline {
  %ptr8 = bitcast $2 * %0 to i8 *
  call void @llvm.prefetch(i8 * %ptr8, i32 0, i32 0)
  ret void
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define(`stdlib_core', `

declare i8* @ISPCMalloc(i64, i32) nounwind
declare i8* @ISPCFree(i8*) nounwind
declare void @ISPCLaunch(i8*, i8*) nounwind
declare void @ISPCSync() nounwind
declare void @ISPCInstrument(i8*, i8*, i32, i32) nounwind

declare i1 @__is_compile_time_constant_mask(<$1 x i32> %mask)
declare i1 @__is_compile_time_constant_varying_int32(<$1 x i32>)

; This function declares placeholder masked store functions for the
;  front-end to use.
;
;  void __pseudo_masked_store_8 (uniform int8 *ptr, varying int8 values, mask)
;  void __pseudo_masked_store_16(uniform int16 *ptr, varying int16 values, mask)
;  void __pseudo_masked_store_32(uniform int32 *ptr, varying int32 values, mask)
;  void __pseudo_masked_store_64(uniform int64 *ptr, varying int64 values, mask)
;
;  These in turn are converted to native masked stores or to regular
;  stores (if the mask is all on) by the MaskedStoreOptPass optimization
;  pass.

declare void @__pseudo_masked_store_8(<$1 x i8> * nocapture, <$1 x i8>, <$1 x i32>)
declare void @__pseudo_masked_store_16(<$1 x i16> * nocapture, <$1 x i16>, <$1 x i32>)
declare void @__pseudo_masked_store_32(<$1 x i32> * nocapture, <$1 x i32>, <$1 x i32>)
declare void @__pseudo_masked_store_64(<$1 x i64> * nocapture, <$1 x i64>, <$1 x i32>)

; Declare the pseudo-gather functions.  When the ispc front-end needs
; to perform a gather, it generates a call to one of these functions,
; which have signatures:
;    
; varying int8  __pseudo_gather(varying int8 *, mask)
; varying int16 __pseudo_gather(varying int16 *, mask)
; varying int32 __pseudo_gather(varying int32 *, mask)
; varying int64 __pseudo_gather(varying int64 *, mask)
;
; These functions are never actually implemented; the
; GatherScatterFlattenOpt optimization pass finds them and then converts
; them to make calls to the following functions, which represent gathers
; from a common base pointer with offsets.  This approach allows the
; front-end to be relatively simple in how it emits address calculation
; for gathers.
;
; varying int8  __pseudo_gather_base_offsets_8(uniform int8 *base, 
;                                              int32 offsets, mask)
; varying int16 __pseudo_gather_base_offsets_16(uniform int16 *base, 
;                                               int32 offsets, mask)
; varying int32 __pseudo_gather_base_offsets_32(uniform int32 *base, 
;                                               int32 offsets, mask)
; varying int64 __pseudo_gather_base_offsets_64(uniform int64 *base, 
;                                               int64 offsets, mask)
;
; Then, the GSImprovementsPass optimizations finds these and either
; converts them to native gather functions or converts them to vector
; loads, if equivalent.

declare <$1 x i8>  @__pseudo_gather_8([$1 x i8 *], <$1 x i32>) nounwind readonly
declare <$1 x i16> @__pseudo_gather_16([$1 x i8 *], <$1 x i32>) nounwind readonly
declare <$1 x i32> @__pseudo_gather_32([$1 x i8 *], <$1 x i32>) nounwind readonly
declare <$1 x i64> @__pseudo_gather_64([$1 x i8 *], <$1 x i32>) nounwind readonly

declare <$1 x i8>  @__pseudo_gather_base_offsets_8(i8 *, <$1 x i32>, <$1 x i32>) nounwind readonly
declare <$1 x i16> @__pseudo_gather_base_offsets_16(i8 *, <$1 x i32>, <$1 x i32>) nounwind readonly
declare <$1 x i32> @__pseudo_gather_base_offsets_32(i8 *, <$1 x i32>, <$1 x i32>) nounwind readonly
declare <$1 x i64> @__pseudo_gather_base_offsets_64(i8 *, <$1 x i32>, <$1 x i32>) nounwind readonly

; Similarly to the pseudo-gathers defined above, we also declare undefined
; pseudo-scatter instructions with signatures:
;
; void __pseudo_scatter_8 (varying int8 *, varying int8 values, mask)
; void __pseudo_scatter_16(varying int16 *, varying int16 values, mask)
; void __pseudo_scatter_32(varying int32 *, varying int32 values, mask)
; void __pseudo_scatter_64(varying int64 *, varying int64 values, mask)
;
; The GatherScatterFlattenOpt optimization pass also finds these and
; transforms them to scatters like:
;
; void __pseudo_scatter_base_offsets_8(uniform int8 *base, 
;                 varying int32 offsets, varying int8 values, mask)
; void __pseudo_scatter_base_offsets_16(uniform int16 *base, 
;                 varying int32 offsets, varying int16 values, mask)
; void __pseudo_scatter_base_offsets_32(uniform int32 *base, 
;                 varying int32 offsets, varying int32 values, mask)
; void __pseudo_scatter_base_offsets_64(uniform int64 *base, 
;                 varying int32 offsets, varying int64 values, mask)
;
; And the GSImprovementsPass in turn converts these to actual native
; scatters or masked stores.  

declare void @__pseudo_scatter_8([$1 x i8 *], <$1 x i8>, <$1 x i32>) nounwind
declare void @__pseudo_scatter_16([$1 x i8 *], <$1 x i16>, <$1 x i32>) nounwind
declare void @__pseudo_scatter_32([$1 x i8 *], <$1 x i32>, <$1 x i32>) nounwind
declare void @__pseudo_scatter_64([$1 x i8 *], <$1 x i64>, <$1 x i32>) nounwind

declare void @__pseudo_scatter_base_offsets_8(i8 * nocapture, <$1 x i32>,
                                              <$1 x i8>, <$1 x i32>) nounwind
declare void @__pseudo_scatter_base_offsets_16(i8 * nocapture, <$1 x i32>,
                                               <$1 x i16>, <$1 x i32>) nounwind
declare void @__pseudo_scatter_base_offsets_32(i8 * nocapture, <$1 x i32>,
                                               <$1 x i32>, <$1 x i32>) nounwind
declare void @__pseudo_scatter_base_offsets_64(i8 * nocapture, <$1 x i32>,
                                               <$1 x i64>, <$1 x i32>) nounwind

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; vector ops

define internal i8 @__extract_int8(<$1 x i8>, i32) nounwind readnone alwaysinline {
  %extract = extractelement <$1 x i8> %0, i32 %1
  ret i8 %extract
}

define internal <$1 x i8> @__insert_int8(<$1 x i8>, i32, 
                                           i8) nounwind readnone alwaysinline {
  %insert = insertelement <$1 x i8> %0, i8 %2, i32 %1
  ret <$1 x i8> %insert
}

define internal i16 @__extract_int16(<$1 x i16>, i32) nounwind readnone alwaysinline {
  %extract = extractelement <$1 x i16> %0, i32 %1
  ret i16 %extract
}

define internal <$1 x i16> @__insert_int16(<$1 x i16>, i32, 
                                           i16) nounwind readnone alwaysinline {
  %insert = insertelement <$1 x i16> %0, i16 %2, i32 %1
  ret <$1 x i16> %insert
}

define internal i32 @__extract_int32(<$1 x i32>, i32) nounwind readnone alwaysinline {
  %extract = extractelement <$1 x i32> %0, i32 %1
  ret i32 %extract
}

define internal <$1 x i32> @__insert_int32(<$1 x i32>, i32, 
                                           i32) nounwind readnone alwaysinline {
  %insert = insertelement <$1 x i32> %0, i32 %2, i32 %1
  ret <$1 x i32> %insert
}

define internal i64 @__extract_int64(<$1 x i64>, i32) nounwind readnone alwaysinline {
  %extract = extractelement <$1 x i64> %0, i32 %1
  ret i64 %extract
}

define internal <$1 x i64> @__insert_int64(<$1 x i64>, i32, 
                                           i64) nounwind readnone alwaysinline {
  %insert = insertelement <$1 x i64> %0, i64 %2, i32 %1
  ret <$1 x i64> %insert
}

shuffles($1, i8, int8, 1)
shuffles($1, i16, int16, 2)
shuffles($1, float, float, 4)
shuffles($1, i32, int32, 4)
shuffles($1, double, double, 8)
shuffles($1, i64, int64, 8)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; various bitcasts from one type to another

define internal <$1 x i32> @__intbits_varying_float(<$1 x float>) nounwind readnone alwaysinline {
  %float_to_int_bitcast = bitcast <$1 x float> %0 to <$1 x i32>
  ret <$1 x i32> %float_to_int_bitcast
}

define internal i32 @__intbits_uniform_float(float) nounwind readnone alwaysinline {
  %float_to_int_bitcast = bitcast float %0 to i32
  ret i32 %float_to_int_bitcast
}

define internal <$1 x i64> @__intbits_varying_double(<$1 x double>) nounwind readnone alwaysinline {
  %double_to_int_bitcast = bitcast <$1 x double> %0 to <$1 x i64>
  ret <$1 x i64> %double_to_int_bitcast
}

define internal i64 @__intbits_uniform_double(double) nounwind readnone alwaysinline {
  %double_to_int_bitcast = bitcast double %0 to i64
  ret i64 %double_to_int_bitcast
}

define internal <$1 x float> @__floatbits_varying_int32(<$1 x i32>) nounwind readnone alwaysinline {
  %int_to_float_bitcast = bitcast <$1 x i32> %0 to <$1 x float>
  ret <$1 x float> %int_to_float_bitcast
}

define internal float @__floatbits_uniform_int32(i32) nounwind readnone alwaysinline {
  %int_to_float_bitcast = bitcast i32 %0 to float
  ret float %int_to_float_bitcast
}

define internal <$1 x double> @__doublebits_varying_int64(<$1 x i64>) nounwind readnone alwaysinline {
  %int_to_double_bitcast = bitcast <$1 x i64> %0 to <$1 x double>
  ret <$1 x double> %int_to_double_bitcast
}

define internal double @__doublebits_uniform_int64(i64) nounwind readnone alwaysinline {
  %int_to_double_bitcast = bitcast i64 %0 to double
  ret double %int_to_double_bitcast
}

define internal <$1 x float> @__undef_varying() nounwind readnone alwaysinline {
  ret <$1 x float> undef
}

define internal float @__undef_uniform() nounwind readnone alwaysinline {
  ret float undef
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sign extension

define internal i32 @__sext_uniform_bool(i1) nounwind readnone alwaysinline {
  %r = sext i1 %0 to i32
  ret i32 %r
}

define internal <$1 x i32> @__sext_varying_bool(<$1 x i32>) nounwind readnone alwaysinline {
  ret <$1 x i32> %0
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefetching

prefetch_read(uniform_bool, i1)
prefetch_read(uniform_int8, i8)
prefetch_read(uniform_int16, i16)
prefetch_read(uniform_int32, i32)
prefetch_read(uniform_int64, i64)
prefetch_read(uniform_float, float)
prefetch_read(uniform_double, double)

prefetch_read(varying_bool, <$1 x i32>)
prefetch_read(varying_int8, <$1 x i8>)
prefetch_read(varying_int16, <$1 x i16>)
prefetch_read(varying_int32, <$1 x i32>)
prefetch_read(varying_int64, <$1 x i64>)
prefetch_read(varying_float, <$1 x float>)
prefetch_read(varying_double, <$1 x double>)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; stdlib transcendentals
;;
;; These functions provide entrypoints that call out to the libm 
;; implementations of the transcendental functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

declare float @sinf(float) nounwind readnone
declare float @cosf(float) nounwind readnone
declare void @sincosf(float, float *, float *) nounwind readnone
declare float @tanf(float) nounwind readnone
declare float @atanf(float) nounwind readnone
declare float @atan2f(float, float) nounwind readnone
declare float @expf(float) nounwind readnone
declare float @logf(float) nounwind readnone
declare float @powf(float, float) nounwind readnone

define internal float @__stdlib_sinf(float) nounwind readnone alwaysinline {
  %r = call float @sinf(float %0)
  ret float %r
}

define internal float @__stdlib_cosf(float) nounwind readnone alwaysinline {
  %r = call float @cosf(float %0)
  ret float %r
}

define internal void @__stdlib_sincosf(float, float *, float *) nounwind readnone alwaysinline {
  call void @sincosf(float %0, float *%1, float *%2)
  ret void
}

define internal float @__stdlib_tanf(float) nounwind readnone alwaysinline {
  %r = call float @tanf(float %0)
  ret float %r
}

define internal float @__stdlib_atanf(float) nounwind readnone alwaysinline {
  %r = call float @atanf(float %0)
  ret float %r
}

define internal float @__stdlib_atan2f(float, float) nounwind readnone alwaysinline {
  %r = call float @atan2f(float %0, float %1)
  ret float %r
}

define internal float @__stdlib_logf(float) nounwind readnone alwaysinline {
  %r = call float @logf(float %0)
  ret float %r
}

define internal float @__stdlib_expf(float) nounwind readnone alwaysinline {
  %r = call float @expf(float %0)
  ret float %r
}

define internal float @__stdlib_powf(float, float) nounwind readnone alwaysinline {
  %r = call float @powf(float %0, float %1)
  ret float %r
}

declare double @sin(double) nounwind readnone
declare double @cos(double) nounwind readnone
declare void @sincos(double, double *, double *) nounwind readnone
declare double @tan(double) nounwind readnone
declare double @atan(double) nounwind readnone
declare double @atan2(double, double) nounwind readnone
declare double @exp(double) nounwind readnone
declare double @log(double) nounwind readnone
declare double @pow(double, double) nounwind readnone

define internal double @__stdlib_sin(double) nounwind readnone alwaysinline {
  %r = call double @sin(double %0)
  ret double %r
}

define internal double @__stdlib_cos(double) nounwind readnone alwaysinline {
  %r = call double @cos(double %0)
  ret double %r
}

define internal void @__stdlib_sincos(double, double *, double *) nounwind readnone alwaysinline {
  call void @sincos(double %0, double *%1, double *%2)
  ret void
}

define internal double @__stdlib_tan(double) nounwind readnone alwaysinline {
  %r = call double @tan(double %0)
  ret double %r
}

define internal double @__stdlib_atan(double) nounwind readnone alwaysinline {
  %r = call double @atan(double %0)
  ret double %r
}

define internal double @__stdlib_atan2(double, double) nounwind readnone alwaysinline {
  %r = call double @atan2(double %0, double %1)
  ret double %r
}

define internal double @__stdlib_log(double) nounwind readnone alwaysinline {
  %r = call double @log(double %0)
  ret double %r
}

define internal double @__stdlib_exp(double) nounwind readnone alwaysinline {
  %r = call double @exp(double %0)
  ret double %r
}

define internal double @__stdlib_pow(double, double) nounwind readnone alwaysinline {
  %r = call double @pow(double %0, double %1)
  ret double %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; atomics and memory barriers

declare void @llvm.memory.barrier(i1 %loadload, i1 %loadstore, i1 %storeload,
                                  i1 %storestore, i1 %device)

define internal void @__memory_barrier() nounwind readnone alwaysinline {
  ;; see http://llvm.org/bugs/show_bug.cgi?id=2829.  It seems like we
  ;; only get an MFENCE on x86 if "device" is true, but IMHO we should
  ;; in the case where the first 4 args are true but it is false.
  ;;  So we just always set that to true...
  call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 true)
  ret void
}

global_atomic($1, add, i32, int32)
global_atomic($1, sub, i32, int32)
global_atomic($1, and, i32, int32)
global_atomic($1, or, i32, int32)
global_atomic($1, xor, i32, int32)
global_atomic($1, min, i32, int32)
global_atomic($1, max, i32, int32)
global_atomic($1, umin, i32, uint32)
global_atomic($1, umax, i32, uint32)

global_atomic($1, add, i64, int64)
global_atomic($1, sub, i64, int64)
global_atomic($1, and, i64, int64)
global_atomic($1, or, i64, int64)
global_atomic($1, xor, i64, int64)
global_atomic($1, min, i64, int64)
global_atomic($1, max, i64, int64)
global_atomic($1, umin, i64, uint64)
global_atomic($1, umax, i64, uint64)

global_swap($1, i32, int32)
global_swap($1, i64, int64)

define internal <$1 x float> @__atomic_swap_float_global(float * %ptr, <$1 x float> %val,
                                                   <$1 x i32> %mask) nounwind alwaysinline {
  %iptr = bitcast float * %ptr to i32 *
  %ival = bitcast <$1 x float> %val to <$1 x i32>
  %iret = call <$1 x i32> @__atomic_swap_int32_global(i32 * %iptr, <$1 x i32> %ival, <$1 x i32> %mask)
  %ret = bitcast <$1 x i32> %iret to <$1 x float>
  ret <$1 x float> %ret
}

define internal <$1 x double> @__atomic_swap_double_global(double * %ptr, <$1 x double> %val,
                                                   <$1 x i32> %mask) nounwind alwaysinline {
  %iptr = bitcast double * %ptr to i64 *
  %ival = bitcast <$1 x double> %val to <$1 x i64>
  %iret = call <$1 x i64> @__atomic_swap_int64_global(i64 * %iptr, <$1 x i64> %ival, <$1 x i32> %mask)
  %ret = bitcast <$1 x i64> %iret to <$1 x double>
  ret <$1 x double> %ret
}

global_atomic_exchange($1, i32, int32)
global_atomic_exchange($1, i64, int64)

define internal <$1 x float> @__atomic_compare_exchange_float_global(float * %ptr,
                      <$1 x float> %cmp, <$1 x float> %val, <$1 x i32> %mask) nounwind alwaysinline {
  %iptr = bitcast float * %ptr to i32 *
  %icmp = bitcast <$1 x float> %cmp to <$1 x i32>
  %ival = bitcast <$1 x float> %val to <$1 x i32>
  %iret = call <$1 x i32> @__atomic_compare_exchange_int32_global(i32 * %iptr, <$1 x i32> %icmp,
                                                                  <$1 x i32> %ival, <$1 x i32> %mask)
  %ret = bitcast <$1 x i32> %iret to <$1 x float>
  ret <$1 x float> %ret
}

define internal <$1 x double> @__atomic_compare_exchange_double_global(double * %ptr,
                      <$1 x double> %cmp, <$1 x double> %val, <$1 x i32> %mask) nounwind alwaysinline {
  %iptr = bitcast double * %ptr to i64 *
  %icmp = bitcast <$1 x double> %cmp to <$1 x i64>
  %ival = bitcast <$1 x double> %val to <$1 x i64>
  %iret = call <$1 x i64> @__atomic_compare_exchange_int64_global(i64 * %iptr, <$1 x i64> %icmp,
                                                                  <$1 x i64> %ival, <$1 x i32> %mask)
  %ret = bitcast <$1 x i64> %iret to <$1 x double>
  ret <$1 x double> %ret
}
')


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 64-bit integer min and max functions

;; utility function used by int64minmax below.  This shouldn't be called by
;; target .ll files directly.
;; $1: target vector width
;; $2: {min,max} (used in constructing function names)
;; $3: {int64,uint64} (used in constructing function names)
;; $4: {slt,sgt} comparison operator to used

define(`i64minmax', `
define internal i64 @__$2_uniform_$3(i64, i64) nounwind alwaysinline readnone {
  %c = icmp $4 i64 %0, %1
  %r = select i1 %c, i64 %0, i64 %1
  ret i64 %r
}

define internal <$1 x i64> @__$2_varying_$3(<$1 x i64>, <$1 x i64>) nounwind alwaysinline readnone {
  %rptr = alloca <$1 x i64>
  %r64ptr = bitcast <$1 x i64> * %rptr to i64 *

  forloop(i, 0, eval($1-1), `
  %v0_`'i = extractelement <$1 x i64> %0, i32 i
  %v1_`'i = extractelement <$1 x i64> %1, i32 i
  %c_`'i = icmp $4 i64 %v0_`'i, %v1_`'i
  %v_`'i = select i1 %c_`'i, i64 %v0_`'i, i64 %v1_`'i
  %ptr_`'i = getelementptr i64 * %r64ptr, i32 i
  store i64 %v_`'i, i64 * %ptr_`'i
')                  

  %ret = load <$1 x i64> * %rptr
  ret <$1 x i64> %ret
}
')

;; this is the function that target .ll files should call; it just takes the target
;; vector width as a parameter

define(`int64minmax', `
i64minmax($1,min,int64,slt)
i64minmax($1,max,int64,sgt)
i64minmax($1,min,uint64,ult)
i64minmax($1,max,uint64,ugt)
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Emit code to safely load a scalar value and broadcast it across the
;; elements of a vector.  Parameters:
;; $1: target vector width
;; $2: element type for which to emit the function (i32, i64, ...)
;; $3: suffix for function name (32, 64, ...)


define(`load_and_broadcast', `
define <$1 x $2> @__load_and_broadcast_$3(i8 *, <$1 x i32> %mask) nounwind alwaysinline {
  ; must not load if the mask is all off; the address may be invalid
  %mm = call i32 @__movmsk(<$1 x i32> %mask)
  %any_on = icmp ne i32 %mm, 0
  br i1 %any_on, label %load, label %skip

load:
  %ptr = bitcast i8 * %0 to $2 *
  %val = load $2 * %ptr

  %ret0 = insertelement <$1 x $2> undef, $2 %val, i32 0
  forloop(i, 1, eval($1-1), `
  %ret`'i = insertelement <$1 x $2> %ret`'eval(i-1), $2 %val, i32 i')
  ret <$1 x $2> %ret`'eval($1-1)

skip:
  ret <$1 x $2> undef
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Emit general-purpose code to do a masked load for targets that dont have
;; an instruction to do that.  Parameters:
;; $1: target vector width
;; $2: element type for which to emit the function (i32, i64, ...)
;; $3: suffix for function name (32, 64, ...)
;; $4: alignment for elements of type $2 (4, 8, ...)

define(`load_masked', `
define <$1 x $2> @__load_masked_$3(i8 *, <$1 x i32> %mask) nounwind alwaysinline {
entry:
  %mm = call i32 @__movmsk(<$1 x i32> %mask)
  ; if the first lane and the last lane are on, then it is safe to do a vector load
  ; of the whole thing--what the lanes in the middle want turns out to not matter...
  %mm_and = and i32 %mm, eval(1 | (1<<($1-1)))
  %can_vload = icmp eq i32 %mm_and, eval(1 | (1<<($1-1)))
  ; if we are not able to do a singe vload, we will accumulate lanes in this memory..
  %retptr = alloca <$1 x $2>
  %retptr32 = bitcast <$1 x $2> * %retptr to $2 *
  br i1 %can_vload, label %load, label %loop

load: 
  %ptr = bitcast i8 * %0 to <$1 x $2> *
  %valall = load <$1 x $2> * %ptr, align $4
  ret <$1 x $2> %valall

loop:
  ; loop over the lanes and see if each one is on...
  %lane = phi i32 [ 0, %entry ], [ %next_lane, %lane_done ]
  %lanemask = shl i32 1, %lane
  %mask_and = and i32 %mm, %lanemask
  %do_lane = icmp ne i32 %mask_and, 0
  br i1 %do_lane, label %load_lane, label %lane_done

load_lane:
  ; yes!  do the load and store the result into the appropriate place in the
  ; allocaed memory above
  %ptr32 = bitcast i8 * %0 to $2 *
  %lane_ptr = getelementptr $2 * %ptr32, i32 %lane
  %val = load $2 * %lane_ptr
  %store_ptr = getelementptr $2 * %retptr32, i32 %lane
  store $2 %val, $2 * %store_ptr
  br label %lane_done

lane_done:
  %next_lane = add i32 %lane, 1
  %done = icmp eq i32 %lane, eval($1-1)
  br i1 %done, label %return, label %loop

return:
  %r = load <$1 x $2> * %retptr
  ret <$1 x $2> %r
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store
;; emit code to do masked store as a set of per-lane scalar stores
;; parameters:
;; $1: target vector width
;; $2: llvm type of elements
;; $3: suffix for function name

define(`gen_masked_store', `
define void @__masked_store_$3(<$1 x $2>* nocapture, <$1 x $2>, <$1 x i32>) nounwind alwaysinline {
  per_lane($1, <$1 x i32> %2, `
      %ptr_ID = getelementptr <$1 x $2> * %0, i32 0, i32 LANE
      %storeval_ID = extractelement <$1 x $2> %1, i32 LANE
      store $2 %storeval_ID, $2 * %ptr_ID')
  ret void
}
')

define(`masked_store_blend_8_16_by_4', `
define void @__masked_store_blend_8(<4 x i8>* nocapture, <4 x i8>,
                                    <4 x i32>) nounwind alwaysinline {
  %old = load <4 x i8> * %0
  %old32 = bitcast <4 x i8> %old to i32
  %new32 = bitcast <4 x i8> %1 to i32

  %mask8 = trunc <4 x i32> %2 to <4 x i8>
  %mask32 = bitcast <4 x i8> %mask8 to i32
  %notmask32 = xor i32 %mask32, -1

  %newmasked = and i32 %new32, %mask32
  %oldmasked = and i32 %old32, %notmask32
  %result = or i32 %newmasked, %oldmasked

  %resultvec = bitcast i32 %result to <4 x i8>
  store <4 x i8> %resultvec, <4 x i8> * %0
  ret void
}

define void @__masked_store_blend_16(<4 x i16>* nocapture, <4 x i16>,
                                     <4 x i32>) nounwind alwaysinline {
  %old = load <4 x i16> * %0
  %old64 = bitcast <4 x i16> %old to i64
  %new64 = bitcast <4 x i16> %1 to i64

  %mask16 = trunc <4 x i32> %2 to <4 x i16>
  %mask64 = bitcast <4 x i16> %mask16 to i64
  %notmask64 = xor i64 %mask64, -1

  %newmasked = and i64 %new64, %mask64
  %oldmasked = and i64 %old64, %notmask64
  %result = or i64 %newmasked, %oldmasked

  %resultvec = bitcast i64 %result to <4 x i16>
  store <4 x i16> %resultvec, <4 x i16> * %0
  ret void
}
')

define(`masked_store_blend_8_16_by_8', `
define void @__masked_store_blend_8(<8 x i8>* nocapture, <8 x i8>,
                                    <8 x i32>) nounwind alwaysinline {
  %old = load <8 x i8> * %0
  %old64 = bitcast <8 x i8> %old to i64
  %new64 = bitcast <8 x i8> %1 to i64

  %mask8 = trunc <8 x i32> %2 to <8 x i8>
  %mask64 = bitcast <8 x i8> %mask8 to i64
  %notmask64 = xor i64 %mask64, -1

  %newmasked = and i64 %new64, %mask64
  %oldmasked = and i64 %old64, %notmask64
  %result = or i64 %newmasked, %oldmasked

  %resultvec = bitcast i64 %result to <8 x i8>
  store <8 x i8> %resultvec, <8 x i8> * %0
  ret void
}

define void @__masked_store_blend_16(<8 x i16>* nocapture, <8 x i16>,
                                     <8 x i32>) nounwind alwaysinline {
  %old = load <8 x i16> * %0
  %old128 = bitcast <8 x i16> %old to i128
  %new128 = bitcast <8 x i16> %1 to i128

  %mask16 = trunc <8 x i32> %2 to <8 x i16>
  %mask128 = bitcast <8 x i16> %mask16 to i128
  %notmask128 = xor i128 %mask128, -1

  %newmasked = and i128 %new128, %mask128
  %oldmasked = and i128 %old128, %notmask128
  %result = or i128 %newmasked, %oldmasked

  %resultvec = bitcast i128 %result to <8 x i16>
  store <8 x i16> %resultvec, <8 x i16> * %0
  ret void
}
')

define(`masked_store_blend_8_16_by_16', `
define void @__masked_store_blend_8(<16 x i8>* nocapture, <16 x i8>,
                                    <16 x i32>) nounwind alwaysinline {
  %old = load <16 x i8> * %0
  %old128 = bitcast <16 x i8> %old to i128
  %new128 = bitcast <16 x i8> %1 to i128

  %mask8 = trunc <16 x i32> %2 to <16 x i8>
  %mask128 = bitcast <16 x i8> %mask8 to i128
  %notmask128 = xor i128 %mask128, -1

  %newmasked = and i128 %new128, %mask128
  %oldmasked = and i128 %old128, %notmask128
  %result = or i128 %newmasked, %oldmasked

  %resultvec = bitcast i128 %result to <16 x i8>
  store <16 x i8> %resultvec, <16 x i8> * %0
  ret void
}

define void @__masked_store_blend_16(<16 x i16>* nocapture, <16 x i16>,
                                     <16 x i32>) nounwind alwaysinline {
  %old = load <16 x i16> * %0
  %old256 = bitcast <16 x i16> %old to i256
  %new256 = bitcast <16 x i16> %1 to i256

  %mask16 = trunc <16 x i32> %2 to <16 x i16>
  %mask256 = bitcast <16 x i16> %mask16 to i256
  %notmask256 = xor i256 %mask256, -1

  %newmasked = and i256 %new256, %mask256
  %oldmasked = and i256 %old256, %notmask256
  %result = or i256 %newmasked, %oldmasked

  %resultvec = bitcast i256 %result to <16 x i16>
  store <16 x i16> %resultvec, <16 x i16> * %0
  ret void
}
')


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; packed load and store functions
;;
;; These define functions to emulate those nice packed load and packed store
;; instructions.  For packed store, given a pointer to destination array and 
;; an offset into the array, for each lane where the mask is on, the
;; corresponding value for that lane is stored into packed locations in the
;; destination array.  For packed load, each lane that has an active mask
;; loads a sequential value from the array.
;;
;; $1: vector width of the target
;;
;; FIXME: use the per_lane macro, defined below, to implement these!

define(`packed_load_and_store', `

define i32 @__packed_load_active([0 x i32] *, i32 %start_offset, <$1 x i32> * %val_ptr,
                                 <$1 x i32> %full_mask) nounwind alwaysinline {
entry:
  %mask = call i32 @__movmsk(<$1 x i32> %full_mask)
  %baseptr = bitcast [0 x i32] * %0 to i32 *
  %startptr = getelementptr i32 * %baseptr, i32 %start_offset
  %mask_known = call i1 @__is_compile_time_constant_mask(<$1 x i32> %full_mask)
  br i1 %mask_known, label %known_mask, label %unknown_mask

known_mask:
  %allon = icmp eq i32 %mask, eval((1 << $1) -1)
  br i1 %allon, label %all_on, label %not_all_on

all_on:
  ;; everyone wants to load, so just load an entire vector width in a single
  ;; vector load
  %vecptr = bitcast i32 *%startptr to <$1 x i32> *
  %vec_load = load <$1 x i32> *%vecptr, align 4
  store <$1 x i32> %vec_load, <$1 x i32> * %val_ptr, align 4
  ret i32 $1

not_all_on:
  %alloff = icmp eq i32 %mask, 0
  br i1 %alloff, label %all_off, label %unknown_mask

all_off:
  ;; no one wants to load
  ret i32 0

unknown_mask:
  br label %loop

loop:
  %lane = phi i32 [ 0, %unknown_mask ], [ %nextlane, %loopend ]
  %lanemask = phi i32 [ 1, %unknown_mask ], [ %nextlanemask, %loopend ]
  %offset = phi i32 [ 0, %unknown_mask ], [ %nextoffset, %loopend ]

  ; is the current lane on?
  %and = and i32 %mask, %lanemask
  %do_load = icmp eq i32 %and, %lanemask
  br i1 %do_load, label %load, label %loopend 

load:
  %loadptr = getelementptr i32 *%startptr, i32 %offset
  %loadval = load i32 *%loadptr
  %val_ptr_i32 = bitcast <$1 x i32> * %val_ptr to i32 *
  %storeptr = getelementptr i32 *%val_ptr_i32, i32 %lane
  store i32 %loadval, i32 *%storeptr
  %offset1 = add i32 %offset, 1
  br label %loopend

loopend:
  %nextoffset = phi i32 [ %offset1, %load ], [ %offset, %loop ]
  %nextlane = add i32 %lane, 1
  %nextlanemask = mul i32 %lanemask, 2

  ; are we done yet?
  %test = icmp ne i32 %nextlane, $1
  br i1 %test, label %loop, label %done

done:
  ret i32 %nextoffset
}

define i32 @__packed_store_active([0 x i32] *, i32 %start_offset, <$1 x i32> %vals,
                                  <$1 x i32> %full_mask) nounwind alwaysinline {
entry:
  %mask = call i32 @__movmsk(<$1 x i32> %full_mask)
  %baseptr = bitcast [0 x i32] * %0 to i32 *
  %startptr = getelementptr i32 * %baseptr, i32 %start_offset
  %mask_known = call i1 @__is_compile_time_constant_mask(<$1 x i32> %full_mask)
  br i1 %mask_known, label %known_mask, label %unknown_mask

known_mask:
  %allon = icmp eq i32 %mask, eval((1 << $1) -1)
  br i1 %allon, label %all_on, label %not_all_on

all_on:
  %vecptr = bitcast i32 *%startptr to <$1 x i32> *
  store <$1 x i32> %vals, <$1 x i32> * %vecptr, align 4
  ret i32 $1

not_all_on:
  %alloff = icmp eq i32 %mask, 0
  br i1 %alloff, label %all_off, label %unknown_mask

all_off:
  ret i32 0

unknown_mask:
  br label %loop

loop:
  %lane = phi i32 [ 0, %unknown_mask ], [ %nextlane, %loopend ]
  %lanemask = phi i32 [ 1, %unknown_mask ], [ %nextlanemask, %loopend ]
  %offset = phi i32 [ 0, %unknown_mask ], [ %nextoffset, %loopend ]

  ; is the current lane on?
  %and = and i32 %mask, %lanemask
  %do_store = icmp eq i32 %and, %lanemask
  br i1 %do_store, label %store, label %loopend 

store:
  %storeval = extractelement <$1 x i32> %vals, i32 %lane
  %storeptr = getelementptr i32 *%startptr, i32 %offset
  store i32 %storeval, i32 *%storeptr
  %offset1 = add i32 %offset, 1
  br label %loopend

loopend:
  %nextoffset = phi i32 [ %offset1, %store ], [ %offset, %loop ]
  %nextlane = add i32 %lane, 1
  %nextlanemask = mul i32 %lanemask, 2

  ; are we done yet?
  %test = icmp ne i32 %nextlane, $1
  br i1 %test, label %loop, label %done

done:
  ret i32 %nextoffset
}
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reduce_equal

; count leading zeros
declare i32 @llvm.cttz.i32(i32)

define(`reduce_equal_aux', `
define internal i1 @__reduce_equal_$3(<$1 x $2> %v, $2 * %samevalue,
                                      <$1 x i32> %mask) nounwind alwaysinline {
entry:
   %mm = call i32 @__movmsk(<$1 x i32> %mask)
   %allon = icmp eq i32 %mm, eval((1<<$1)-1)
   br i1 %allon, label %check_neighbors, label %domixed

domixed:
  ; the mask is mixed on/off.  First see if the lanes are all off
  %alloff = icmp eq i32 %mm, 0
  br i1 %alloff, label %doalloff, label %actuallymixed

doalloff:
  ret i1 false  ;; this seems safest

actuallymixed: 
  ; First, figure out which lane is the first active one
  %first = call i32 @llvm.cttz.i32(i32 %mm)
  %baseval = extractelement <$1 x $2> %v, i32 %first
  %basev1 = bitcast $2 %baseval to <1 x $2>
  ; get a vector that is that value smeared across all elements
  %basesmear = shufflevector <1 x $2> %basev1, <1 x $2> undef,
        <$1 x i32> < forloop(i, 0, eval($1-2), `i32 0, ') i32 0 >

  ; now to a blend of that vector with the original vector, such that the
  ; result will be the original value for the active lanes, and the value
  ; from the first active lane for the inactive lanes.  Given that, we can
  ; just unconditionally check if the lanes are all equal in check_neighbors
  ; below without worrying about inactive lanes...
  %ptr = alloca <$1 x $2>
  store <$1 x $2> %basesmear, <$1 x $2> * %ptr
  %castptr = bitcast <$1 x $2> * %ptr to <$1 x $4> *
  %castv = bitcast <$1 x $2> %v to <$1 x $4>
  call void @__masked_store_blend_$6(<$1 x $4> * %castptr, <$1 x $4> %castv, <$1 x i32> %mask)
  %blendvec = load <$1 x $2> * %ptr
  br label %check_neighbors

check_neighbors:
  %vec = phi <$1 x $2> [ %blendvec, %actuallymixed ], [ %v, %entry ]
  ifelse($6, `32', `
  ; For 32-bit elements, we rotate once and compare with the vector, which ends 
  ; up comparing each element to its neighbor on the right.  Then see if
  ; all of those values are true; if so, then all of the elements are equal..
  %castvec = bitcast <$1 x $2> %vec to <$1 x $4>
  %castvr = call <$1 x $4> @__rotate_int$6(<$1 x $4> %castvec, i32 1)
  %vr = bitcast <$1 x $4> %castvr to <$1 x $2>
  %eq = $5 eq <$1 x $2> %vec, %vr
  %eq32 = sext <$1 x i1> %eq to <$1 x i32>
  %eqmm = call i32 @__movmsk(<$1 x i32> %eq32)
  %alleq = icmp eq i32 %eqmm, eval((1<<$1)-1)
  br i1 %alleq, label %all_equal, label %not_all_equal
  ', `
  ; But for 64-bit elements, it turns out to be more efficient to just
  ; scalarize and do a individual pairwise comparisons and AND those
  ; all together..
  forloop(i, 0, eval($1-1), `
  %v`'i = extractelement <$1 x $2> %vec, i32 i')

  forloop(i, 0, eval($1-2), `
  %eq`'i = $5 eq $2 %v`'i, %v`'eval(i+1)')

  %and0 = and i1 %eq0, %eq1
  forloop(i, 1, eval($1-3), `
  %and`'i = and i1 %and`'eval(i-1), %eq`'eval(i+1)')

  br i1 %and`'eval($1-3), label %all_equal, label %not_all_equal
  ')

all_equal:
  %the_value = extractelement <$1 x $2> %vec, i32 0
  store $2 %the_value, $2 * %samevalue
  ret i1 true

not_all_equal:
  ret i1 false
}
')

define(`reduce_equal', `
reduce_equal_aux($1, i32, int32, i32, icmp, 32)
reduce_equal_aux($1, float, float, i32, fcmp, 32)
reduce_equal_aux($1, i64, int64, i64, icmp, 64)
reduce_equal_aux($1, double, double, i64, fcmp, 64)
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; prefix sum stuff

; $1: vector width (e.g. 4)
; $2: vector element type (e.g. float)
; $3: bit width of vector element type (e.g. 32)
; $4: operator to apply (e.g. fadd)
; $5: identity element value (e.g. 0)
; $6: suffix for function (e.g. add_float)

define(`exclusive_scan', `
define internal <$1 x $2> @__exclusive_scan_$6(<$1 x $2> %v,
                                  <$1 x i32> %mask) nounwind alwaysinline {
  ; first, set the value of any off lanes to the identity value
  %ptr = alloca <$1 x $2>
  %idvec1 = bitcast $2 $5 to <1 x $2>
  %idvec = shufflevector <1 x $2> %idvec1, <1 x $2> undef,
      <$1 x i32> < forloop(i, 0, eval($1-2), `i32 0, ') i32 0 >
  store <$1 x $2> %idvec, <$1 x $2> * %ptr
  %ptr`'$3 = bitcast <$1 x $2> * %ptr to <$1 x i`'$3> *
  %vi = bitcast <$1 x $2> %v to <$1 x i`'$3>
  call void @__masked_store_blend_$3(<$1 x i`'$3> * %ptr`'$3, <$1 x i`'$3> %vi,
                                     <$1 x i32> %mask)
  %v_id = load <$1 x $2> * %ptr

  ; extract elements of the vector to use in computing the scan
  forloop(i, 0, eval($1-1), `
  %v`'i = extractelement <$1 x $2> %v_id, i32 i')

  ; and just compute the scan directly.
  ; 0th element is the identity (so nothing to do here),
  ; 1st element is identity (op) the 0th element of the original vector,
  ; each successive element is the previous element (op) the previous element
  ;  of the original vector
  %s1 = $4 $2 $5, %v0
  forloop(i, 2, eval($1-1), `
  %s`'i = $4 $2 %s`'eval(i-1), %v`'eval(i-1)')

  ; and fill in the result vector
  %r0 = insertelement <$1 x $2> undef, $2 $5, i32 0  ; 0th element gets identity
  forloop(i, 1, eval($1-1), `
  %r`'i = insertelement <$1 x $2> %r`'eval(i-1), $2 %s`'i, i32 i')

  ret <$1 x $2> %r`'eval($1-1)
}
')

define(`scans', `
exclusive_scan($1, i32, 32, add, 0, add_i32)
exclusive_scan($1, float, 32, fadd, zeroinitializer, add_float)
exclusive_scan($1, i64, 64, add, 0, add_i64)
exclusive_scan($1, double, 64, fadd, zeroinitializer, add_double)

exclusive_scan($1, i32, 32, and, -1, and_i32)
exclusive_scan($1, i64, 64, and, -1, and_i64)

exclusive_scan($1, i32, 32, or, 0, or_i32)
exclusive_scan($1, i64, 64, or, 0, or_i64)
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; per_lane
;;
;; The scary macro below encapsulates the 'scalarization' idiom--i.e. we have
;; some operation that we'd like to perform only for the lanes where the
;; mask is on
;; $1: vector width of the target
;; $2: variable that holds the mask
;; $3: block of code to run for each lane that is on
;;       Inside this code, any instances of the text "LANE" are replaced
;;       with an i32 value that represents the current lane number

; num lanes, mask, code block to do per lane
define(`per_lane', `
  br label %pl_entry

pl_entry:
  %pl_mask = call i32 @__movmsk($2)
  %pl_mask_known = call i1 @__is_compile_time_constant_mask($2)
  br i1 %pl_mask_known, label %pl_known_mask, label %pl_unknown_mask

pl_known_mask:
  ;; the mask is known at compile time; see if it is something we can
  ;; handle more efficiently
  %pl_is_allon = icmp eq i32 %pl_mask, eval((1<<$1)-1)
  br i1 %pl_is_allon, label %pl_all_on, label %pl_not_all_on

pl_all_on:
  ;; the mask is all on--just expand the code for each lane sequentially
  forloop(i, 0, eval($1-1), 
          `patsubst(`$3', `ID\|LANE', i)')
  br label %pl_done

pl_not_all_on:
  ;; not all on--see if it is all off or mixed
  ;; for the mixed case, we just run the general case, though we could
  ;; try to be smart and just emit the code based on what it actually is,
  ;; for example by emitting the code straight-line without a loop and doing 
  ;; the lane tests explicitly, leaving later optimization passes to eliminate
  ;; the stuff that is definitely not needed.  Not clear if we will frequently 
  ;; encounter a mask that is known at compile-time but is not either all on or
  ;; all off...
  %pl_alloff = icmp eq i32 %pl_mask, 0
  br i1 %pl_alloff, label %pl_done, label %pl_unknown_mask

pl_unknown_mask:
  br label %pl_loop

pl_loop:
  ;; Loop over each lane and see if we want to do the work for this lane
  %pl_lane = phi i32 [ 0, %pl_unknown_mask ], [ %pl_nextlane, %pl_loopend ]
  %pl_lanemask = phi i32 [ 1, %pl_unknown_mask ], [ %pl_nextlanemask, %pl_loopend ]

  ; is the current lane on?  if so, goto do work, otherwise to end of loop
  %pl_and = and i32 %pl_mask, %pl_lanemask
  %pl_doit = icmp eq i32 %pl_and, %pl_lanemask
  br i1 %pl_doit, label %pl_dolane, label %pl_loopend 

pl_dolane:
  ;; If so, substitute in the code from the caller and replace the LANE
  ;; stuff with the current lane number
  patsubst(`patsubst(`$3', `LANE_ID', `_id')', `LANE', `%pl_lane')
  br label %pl_loopend

pl_loopend:
  %pl_nextlane = add i32 %pl_lane, 1
  %pl_nextlanemask = mul i32 %pl_lanemask, 2

  ; are we done yet?
  %pl_test = icmp ne i32 %pl_nextlane, $1
  br i1 %pl_test, label %pl_loop, label %pl_done

pl_done:
')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather
;;
;; $1: vector width of the target
;; $2: scalar type for which to generate functions to do gathers

; vec width, type
define(`gen_gather', `
;; Define the utility function to do the gather operation for a single element
;; of the type
define internal <$1 x $2> @__gather_elt_$2(i8 * %ptr, <$1 x i32> %offsets, <$1 x $2> %ret,
                                           i32 %lane) nounwind readonly alwaysinline {
  ; compute address for this one from the base
  %offset32 = extractelement <$1 x i32> %offsets, i32 %lane
  %ptroffset = getelementptr i8 * %ptr, i32 %offset32
  %ptrcast = bitcast i8 * %ptroffset to $2 *

  ; load value and insert into returned value
  %val = load $2 *%ptrcast
  %updatedret = insertelement <$1 x $2> %ret, $2 %val, i32 %lane
  ret <$1 x $2> %updatedret
}


define <$1 x $2> @__gather_base_offsets_$2(i8 * %ptr, <$1 x i32> %offsets,
                                           <$1 x i32> %vecmask) nounwind readonly alwaysinline {
entry:
  %mask = call i32 @__movmsk(<$1 x i32> %vecmask)

  %maskKnown = call i1 @__is_compile_time_constant_mask(<$1 x i32> %vecmask)
  br i1 %maskKnown, label %known_mask, label %unknown_mask

known_mask:
  %alloff = icmp eq i32 %mask, 0
  br i1 %alloff, label %gather_all_off, label %unknown_mask

gather_all_off:
  ret <$1 x $2> undef

unknown_mask:
  ; We can be clever and avoid the per-lane stuff for gathers if we are willing
  ; to require that the 0th element of the array being gathered from is always
  ; legal to read from (and we do indeed require that, given the benefits!) 
  ;
  ; Set the offset to zero for lanes that are off
  %offsetsPtr = alloca <$1 x i32>
  store <$1 x i32> zeroinitializer, <$1 x i32> * %offsetsPtr
  call void @__masked_store_blend_32(<$1 x i32> * %offsetsPtr, <$1 x i32> %offsets, 
                                     <$1 x i32> %vecmask)
  %newOffsets = load <$1 x i32> * %offsetsPtr

  %ret0 = call <$1 x $2> @__gather_elt_$2(i8 * %ptr, <$1 x i32> %newOffsets,
                                          <$1 x $2> undef, i32 0)
  forloop(lane, 1, eval($1-1), 
          `patsubst(patsubst(`%retLANE = call <$1 x $2> @__gather_elt_$2(i8 * %ptr, 
                                <$1 x i32> %newOffsets, <$1 x $2> %retPREV, i32 LANE)
                    ', `LANE', lane), `PREV', eval(lane-1))')
  ret <$1 x $2> %ret`'eval($1-1)
}
'
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gen_scatter
;; Emit a function declaration for a scalarized scatter.
;;
;; $1: target vector width
;; $2: scalar type for which we want to generate code to scatter

define(`gen_scatter', `
;; Define the function that descripes the work to do to scatter a single
;; value
define internal void @__scatter_elt_$2(i64 %ptr64, <$1 x i32> %offsets, <$1 x $2> %values,
                                       i32 %lane) nounwind alwaysinline {
  %offset32 = extractelement <$1 x i32> %offsets, i32 %lane
  %offset64 = zext i32 %offset32 to i64
  %ptrdelta = add i64 %ptr64, %offset64
  %ptr = inttoptr i64 %ptrdelta to $2 *
  %storeval = extractelement <$1 x $2> %values, i32 %lane
  store $2 %storeval, $2 * %ptr
  ret void
}

define void @__scatter_base_offsets_$2(i8* %base, <$1 x i32> %offsets, <$1 x $2> %values,
                                       <$1 x i32> %mask) nounwind alwaysinline {
  ;; And use the `per_lane' macro to do all of the per-lane work for scatter...
  %ptr64 = ptrtoint i8 * %base to i64
  per_lane($1, <$1 x i32> %mask, `
      call void @__scatter_elt_$2(i64 %ptr64, <$1 x i32> %offsets, <$1 x $2> %values, i32 LANE)')
  ret void
}
'
)
