;;  Copyright (c) 2013-2015, Intel Corporation
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


;; svml macro

;; svml_stubs : stubs for svml calls
;; $1 - type ("float" or "double")
;; $2 - svml internal function suffix ("f" for float, "d" for double)
;; $3 - vector width
define(`svml_stubs',`
  declare <$3 x $1> @__svml_sin$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_asin$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__svml_cos$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare void @__svml_sincos$2(<$3 x $1>, <$3 x $1> *, <$3 x $1> *) nounwind readnone alwaysinline 
  declare <$3 x $1> @__svml_tan$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__svml_atan$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__svml_atan2$2(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__svml_exp$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__svml_log$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__svml_pow$2(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline 
')

;; svml_declare : declaration of __svml_* intrinsics 
;; $1 - type ("float" or "double")
;; $2 - __svml_* intrinsic function suffix 
;;      float:  "f4"(sse) "f8"(avx) "f16"(avx512)
;;      double:  "2"(sse)  "4"(avx)   "8"(avx512)
;; $3 - vector width
define(`svml_declare',`
  declare <$3 x $1> @__svml_sin$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_asin$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_cos$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_sincos$2(<$3 x $1> *, <$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_tan$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_atan$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_atan2$2(<$3 x $1>, <$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_exp$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_log$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_pow$2(<$3 x $1>, <$3 x $1>) nounwind readnone
');

;; defintition of __svml_* internal functions
;; $1 - type ("float" or "double")
;; $2 - __svml_* intrinsic function suffix 
;;      float:  "f4"(sse) "f8"(avx) "f16"(avx512)
;;      double:  "2"(sse)  "4"(avx)   "8"(avx512)
;; $3 - vector width
;; $4 - svml internal function suffix ("f" for float, "d" for double)
define(`svml_define',`
  define <$3 x $1> @__svml_sin$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_sin$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }
  define <$3 x $1> @__svml_asin$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_asin$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__svml_cos$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_cos$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define void @__svml_sincos$4(<$3 x $1>, <$3 x $1> *, <$3 x $1> *) nounwind readnone alwaysinline {
    %s = call <$3 x $1> @__svml_sincos$2(<$3 x $1> * %2, <$3 x $1> %0)
    store <$3 x $1> %s, <$3 x $1> * %1
    ret void
  }

  define <$3 x $1> @__svml_tan$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_tan$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__svml_atan$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_atan$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__svml_atan2$4(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_atan2$2(<$3 x $1> %0, <$3 x $1> %1)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__svml_exp$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_exp$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__svml_log$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_log$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__svml_pow$4(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_pow$2(<$3 x $1> %0, <$3 x $1> %1)
    ret <$3 x $1> %ret
  }
')


;; svml_define_x : defintition of __svml_* internal functions operation on extended width
;; $1 - type ("float" or "double")
;; $2 - __svml_* intrinsic function suffix 
;;      float:  "f4"(sse) "f8"(avx) "f16"(avx512)
;;      double:  "2"(sse)  "4"(avx)   "8"(avx512)
;; $3 - vector width
;; $4 - svml internal function suffix ("f" for float, "d" for double)
;; $5 - extended width, must be at least twice the native vector width
;;      contigent on existing of unary$3to$5 and binary$3to$5 macros

;; *todo*: in sincos call use __svml_sincos[f][2,4,8,16] call, e.g.
;;define void @__svml_sincosf(<8 x float>, <8 x float> *,
;;                                    <8 x float> *) nounwind readnone alwaysinline {
;;  ; call svml_sincosf4 two times with the two 4-wide sub-vectors
;;  %a = shufflevector <8 x float> %0, <8 x float> undef,
;;         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
;;  %b = shufflevector <8 x float> %0, <8 x float> undef,
;;         <4 x i32> <i32 4, i32 5, i32 6, i32 7>
;;
;;  %cospa = alloca <4 x float>
;;  %sa = call <4 x float> @__svml_sincosf4(<4 x float> * %cospa, <4 x float> %a)
;;
;;  %cospb = alloca <4 x float>
;;  %sb = call <4 x float> @__svml_sincosf4(<4 x float> * %cospb, <4 x float> %b)
;;
;;  %sin = shufflevector <4 x float> %sa, <4 x float> %sb,
;;         <8 x i32> <i32 0, i32 1, i32 2, i32 3,
;;                    i32 4, i32 5, i32 6, i32 7>
;;  store <8 x float> %sin, <8 x float> * %1
;;
;;  %cosa = load <4 x float> * %cospa
;;  %cosb = load <4 x float> * %cospb
;;  %cos = shufflevector <4 x float> %cosa, <4 x float> %cosb,
;;         <8 x i32> <i32 0, i32 1, i32 2, i32 3,
;;                    i32 4, i32 5, i32 6, i32 7>
;;  store <8 x float> %cos, <8 x float> * %2
;;
;;  ret void
;;}
define(`svml_define_x',`
  define <$5 x $1> @__svml_sin$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_sin$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__svml_asin$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_asin$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__svml_cos$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_cos$2, %0)
    ret <$5 x $1> %ret
  }
  define void @__svml_sincos$4(<$5 x $1>,<$5 x $1>*,<$5 x $1>*) nounwind readnone alwaysinline 
  {
    %s = call <$5 x $1> @__svml_sin$4(<$5 x $1> %0)
    %c = call <$5 x $1> @__svml_cos$4(<$5 x $1> %0)
    store <$5 x $1> %s, <$5 x $1> * %1
    store <$5 x $1> %c, <$5 x $1> * %2
    ret void
  }
  define <$5 x $1> @__svml_tan$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_tan$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__svml_atan$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_atan$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__svml_atan2$4(<$5 x $1>,<$5 x $1>) nounwind readnone alwaysinline {
    binary$3to$5(ret, $1, @__svml_atan2$2, %0, %1)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__svml_exp$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_exp$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__svml_log$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_log$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__svml_pow$4(<$5 x $1>,<$5 x $1>) nounwind readnone alwaysinline {
    binary$3to$5(ret, $1, @__svml_pow$2, %0, %1)
    ret <$5 x $1> %ret
  }
')

