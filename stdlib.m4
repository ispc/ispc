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


define(`stdlib_core', `

declare i1 @__is_compile_time_constant_mask(<$1 x i32> %mask)
declare i1 @__is_compile_time_constant_varying_int32(<$1 x i32>)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; vector ops

define internal float @__extract(<$1 x float>, i32) nounwind readnone alwaysinline {
  %extract = extractelement <$1 x float> %0, i32 %1
  ret float %extract
}

define internal <$1 x float> @__insert(<$1 x float>, i32, 
                                       float) nounwind readnone alwaysinline {
  %insert = insertelement <$1 x float> %0, float %2, i32 %1
  ret <$1 x float> %insert
}

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

define internal float @__stdlib_sin(float) nounwind readnone alwaysinline {
  %r = call float @sinf(float %0)
  ret float %r
}

define internal float @__stdlib_cos(float) nounwind readnone alwaysinline {
  %r = call float @cosf(float %0)
  ret float %r
}

define internal void @__stdlib_sincos(float, float *, float *) nounwind readnone alwaysinline {
  call void @sincosf(float %0, float *%1, float *%2)
  ret void
}

define internal float @__stdlib_tan(float) nounwind readnone alwaysinline {
  %r = call float @tanf(float %0)
  ret float %r
}

define internal float @__stdlib_atan(float) nounwind readnone alwaysinline {
  %r = call float @atanf(float %0)
  ret float %r
}

define internal float @__stdlib_atan2(float, float) nounwind readnone alwaysinline {
  %r = call float @atan2f(float %0, float %1)
  ret float %r
}

define internal float @__stdlib_log(float) nounwind readnone alwaysinline {
  %r = call float @logf(float %0)
  ret float %r
}

define internal float @__stdlib_exp(float) nounwind readnone alwaysinline {
  %r = call float @expf(float %0)
  ret float %r
}

define internal float @__stdlib_pow(float, float) nounwind readnone alwaysinline {
  %r = call float @powf(float %0, float %1)
  ret float %r
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

global_atomic_exchange($1, i32, int32)
global_atomic_exchange($1, i64, int64)

')

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Definitions of 8 and 16-bit load and store functions
;;
;; The `int8_16' macro defines functions related to loading and storing 8 and
;; 16-bit values in memory, converting to and from i32.  (This is a workaround
;; to be able to use in-memory values of types in ispc programs, since the
;; compiler doesn't yet support 8 and 16-bit datatypes...
;;
;; Arguments to pass to `int8_16':
;; $1: vector width of the target

define(`int8_16', `
define internal <$1 x i32> @__load_uint8([0 x i32] *, i32 %offset,
                                         <$1 x i32> %mask) nounwind alwaysinline {
  %mm = call i32 @__movmsk(<$1 x i32> %mask)
  %any = icmp ne i32 %mm, 0
  br i1 %any, label %doload, label %skip

doload:  
  %ptr8 = bitcast [0 x i32] *%0 to i8 *
  %ptr = getelementptr i8 * %ptr8, i32 %offset
  %ptr64 = bitcast i8 * %ptr to i`'eval(8*$1) *
  %val = load i`'eval(8*$1) * %ptr64, align 1

  %vval = bitcast i`'eval(8*$1) %val to <$1 x i8>
  ; unsigned, so zero-extend to i32... 
  %ret = zext <$1 x i8> %vval to <$1 x i32>
  ret <$1 x i32> %ret

skip:
  ret <$1 x i32> undef
}


define internal <$1 x i32> @__load_int8([0 x i32] *, i32 %offset,
                                        <$1 x i32> %mask) nounwind alwaysinline {
  %mm = call i32 @__movmsk(<$1 x i32> %mask)
  %any = icmp ne i32 %mm, 0
  br i1 %any, label %doload, label %skip

doload:  
  %ptr8 = bitcast [0 x i32] *%0 to i8 *
  %ptr = getelementptr i8 * %ptr8, i32 %offset
  %ptr64 = bitcast i8 * %ptr to i`'eval(8*$1) *
  %val = load i`'eval(8*$1) * %ptr64, align 1

  %vval = bitcast i`'eval(8*$1) %val to <$1 x i8>
  ; signed, so sign-extend to i32... 
  %ret = sext <$1 x i8> %vval to <$1 x i32>
  ret <$1 x i32> %ret

skip:
  ret <$1 x i32> undef
}


define internal <$1 x i32> @__load_uint16([0 x i32] *, i32 %offset,
                                          <$1 x i32> %mask) nounwind alwaysinline {
  %mm = call i32 @__movmsk(<$1 x i32> %mask)
  %any = icmp ne i32 %mm, 0
  br i1 %any, label %doload, label %skip

doload:  
  %ptr16 = bitcast [0 x i32] *%0 to i16 *
  %ptr = getelementptr i16 * %ptr16, i32 %offset
  %ptr64 = bitcast i16 * %ptr to i`'eval(16*$1) *
  %val = load i`'eval(16*$1) * %ptr64, align 2

  %vval = bitcast i`'eval(16*$1) %val to <$1 x i16>
  ; unsigned, so use zero-extend...
  %ret = zext <$1 x i16> %vval to <$1 x i32>
  ret <$1 x i32> %ret

skip:
  ret <$1 x i32> undef
}


define internal <$1 x i32> @__load_int16([0 x i32] *, i32 %offset,
                                         <$1 x i32> %mask) nounwind alwaysinline {
  %mm = call i32 @__movmsk(<$1 x i32> %mask)
  %any = icmp ne i32 %mm, 0
  br i1 %any, label %doload, label %skip

doload:  
  %ptr16 = bitcast [0 x i32] *%0 to i16 *
  %ptr = getelementptr i16 * %ptr16, i32 %offset
  %ptr64 = bitcast i16 * %ptr to i`'eval(16*$1) *
  %val = load i`'eval(16*$1) * %ptr64, align 2

  %vval = bitcast i`'eval(16*$1) %val to <$1 x i16>
  ; signed, so use sign-extend...
  %ret = sext <$1 x i16> %vval to <$1 x i32>
  ret <$1 x i32> %ret

skip:
  ret <$1 x i32> undef
}


define internal void @__store_int8([0 x i32] *, i32 %offset, <$1 x i32> %val32,
                                   <$1 x i32> %mask) nounwind alwaysinline {
  %mm = call i32 @__movmsk(<$1 x i32> %mask)
  %any = icmp ne i32 %mm, 0
  br i1 %any, label %dostore, label %skip

dostore:  
  %val = trunc <$1 x i32> %val32 to <$1 x i8>
  %val64 = bitcast <$1 x i8> %val to i`'eval(8*$1)

  %mask8 = trunc <$1 x i32> %mask to <$1 x i8>
  %mask64 = bitcast <$1 x i8> %mask8 to i`'eval(8*$1)
  %notmask = xor i`'eval(8*$1) %mask64, -1

  %ptr8 = bitcast [0 x i32] *%0 to i8 *
  %ptr = getelementptr i8 * %ptr8, i32 %offset
  %ptr64 = bitcast i8 * %ptr to i`'eval(8*$1) *

  ;; load the old value, use logical ops to blend based on the mask, then
  ;; store the result back
  %old = load i`'eval(8*$1) * %ptr64, align 1
  %oldmasked = and i`'eval(8*$1) %old, %notmask
  %newmasked = and i`'eval(8*$1) %val64, %mask64
  %final = or i`'eval(8*$1) %oldmasked, %newmasked
  store i`'eval(8*$1) %final, i`'eval(8*$1) * %ptr64, align 1

  ret void

skip:
  ret void
}

define internal void @__store_int16([0 x i32] *, i32 %offset, <$1 x i32> %val32,
                                    <$1 x i32> %mask) nounwind alwaysinline {
  %mm = call i32 @__movmsk(<$1 x i32> %mask)
  %any = icmp ne i32 %mm, 0
  br i1 %any, label %dostore, label %skip

dostore:
  %val = trunc <$1 x i32> %val32 to <$1 x i16>
  %val64 = bitcast <$1 x i16> %val to i`'eval(16*$1)

  %mask8 = trunc <$1 x i32> %mask to <$1 x i16>
  %mask64 = bitcast <$1 x i16> %mask8 to i`'eval(16*$1)
  %notmask = xor i`'eval(16*$1) %mask64, -1

  %ptr16 = bitcast [0 x i32] *%0 to i16 *
  %ptr = getelementptr i16 * %ptr16, i32 %offset
  %ptr64 = bitcast i16 * %ptr to i`'eval(16*$1) *

  ;; as above, use mask to do blending with logical ops...
  %old = load i`'eval(16*$1) * %ptr64, align 2
  %oldmasked = and i`'eval(16*$1) %old, %notmask
  %newmasked = and i`'eval(16*$1) %val64, %mask64
  %final = or i`'eval(16*$1) %oldmasked, %newmasked
  store i`'eval(16*$1) %final, i`'eval(16*$1) * %ptr64, align 2

  ret void

skip:
  ret void
}
'
)

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
define internal <$1 x $2> @__gather_elt_$2(i64 %ptr64, <$1 x i32> %offsets, <$1 x $2> %ret,
                                           i32 %lane) nounwind readonly alwaysinline {
  ; compute address for this one from the base
  %offset32 = extractelement <$1 x i32> %offsets, i32 %lane
  %offset64 = zext i32 %offset32 to i64
  %ptrdelta = add i64 %ptr64, %offset64
  %ptr = inttoptr i64 %ptrdelta to $2 *

  ; load value and insert into returned value
  %val = load $2 *%ptr
  %updatedret = insertelement <$1 x $2> %ret, $2 %val, i32 %lane
  ret <$1 x $2> %updatedret
}


define <$1 x $2> @__gather_base_offsets_$2(i8*, <$1 x i32> %offsets,
                                           <$1 x i32> %vecmask) nounwind readonly alwaysinline {
entry:
  %mask = call i32 @__movmsk(<$1 x i32> %vecmask)
  %ptr64 = ptrtoint i8 * %0 to i64

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

  %ret0 = call <$1 x $2> @__gather_elt_$2(i64 %ptr64, <$1 x i32> %newOffsets,
                                          <$1 x $2> undef, i32 0)
  forloop(lane, 1, eval($1-1), 
          `patsubst(patsubst(`%retLANE = call <$1 x $2> @__gather_elt_$2(i64 %ptr64, 
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
