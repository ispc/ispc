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


;; acml macro

;; acml_stubs : stubs for acml calls
;; $1 - type ("float" or "double")
;; $2 - acml internal function suffix ("f" for float, "d" for double)
;; $3 - vector width
define(`acml_stubs',`
  declare <$3 x $1> @__acml_sin$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__acml_asin$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__acml_cos$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare void @__acml_sincos$2(<$3 x $1>, <$3 x $1> *, <$3 x $1> *) nounwind readnone alwaysinline 
  declare <$3 x $1> @__acml_tan$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__acml_atan$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__acml_atan2$2(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__acml_exp$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__acml_log$2(<$3 x $1>) nounwind readnone alwaysinline 
  declare <$3 x $1> @__acml_pow$2(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline 
')

;; acml_declare : declaration of __acml_* intrinsics 
;; $1 - type ("float" or "double")
;; $2 - __acml_* intrinsic function suffix 
;;      float:  "f4"(sse) "f8"(avx) "f16"(avx512)
;;      double:  "2"(sse)  "4"(avx)   "8"(avx512)
;; $3 - vector width
define(`acml_declare',`
;;  declare <$3 x $1> @___acml_sin$2(<$3 x $1>) nounwind readnone
;;  declare <$3 x $1> @___acml_asin$2(<$3 x $1>) nounwind readnone
;;  declare <$3 x $1> @___acml_cos$2(<$3 x $1>) nounwind readnone
;;  declare void @___acml_sincos$2(<$3 x $1>,  <$3 x $1> *, <$3 x $1>*) nounwind readnone
;;  declare <$3 x $1> @___acml_tan$2(<$3 x $1>) nounwind readnone
;;  declare <$3 x $1> @___acml_atan$2(<$3 x $1>) nounwind readnone
;;  declare <$3 x $1> @___acml_atan2$2(<$3 x $1>, <$3 x $1>) nounwind readnone
;;  declare <$3 x $1> @___acml_exp$2(<$3 x $1>) nounwind readnone
;;  declare <$3 x $1> @___acml_log$2(<$3 x $1>) nounwind readnone
;;  declare <$3 x $1> @___acml_pow$2(<$3 x $1>, <$3 x $1>) nounwind readnone
');

;; defintition of __acml_* internal functions
;; $1 - type ("float" or "double")
;; $2 - __acml_* intrinsic function suffix 
;;      float:  "f4"(sse) "f8"(avx) "f16"(avx512)
;;      double:  "2"(sse)  "4"(avx)   "8"(avx512)
;; $3 - vector width
;; $4 - acml internal function suffix ("f" for float, "d" for double)
define(`acml_define',`
  define <$3 x $1> @__acml_sin$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_sin$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }
  define <$3 x $1> @__acml_asin$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_asin$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_cos$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_cos$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define void @__acml_sincos$4(<$3 x $1>, <$3 x $1> *, <$3 x $1> *) nounwind readnone alwaysinline {
    call void @___acml_sincos$2(<$3 x $1> %0, <$3 x $1> * %1, <$3 x $1> * %2)
    ret void
  }

  define <$3 x $1> @__acml_tan$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_tan$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_atan$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_atan$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_atan2$4(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_atan2$2(<$3 x $1> %0, <$3 x $1> %1)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_exp$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_exp$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_log$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_log$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_pow$4(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @___acml_pow$2(<$3 x $1> %0, <$3 x $1> %1)
    ret <$3 x $1> %ret
  }
')


;; acml_define_x : defintition of __acml_* internal functions operation on extended width
;; $1 - type ("float" or "double")
;; $2 - __acml_* intrinsic function suffix 
;;      float:  "f4"(sse) "f8"(avx) "f16"(avx512)
;;      double:  "2"(sse)  "4"(avx)   "8"(avx512)
;; $3 - vector width
;; $4 - acml internal function suffix ("f" for float, "d" for double)
;; $5 - extended width, must be at least twice the native vector width
;;      contigent on existing of unary$3to$5 and binary$3to$5 macros

;; *todo*: in sincos call use __acml_sincos[f][2,4,8,16] call, e.g.
;;define void @__acml_sincosf(<8 x float>, <8 x float> *,
;;                                    <8 x float> *) nounwind readnone alwaysinline {
;;  ; call acml_sincosf4 two times with the two 4-wide sub-vectors
;;  %a = shufflevector <8 x float> %0, <8 x float> undef,
;;         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
;;  %b = shufflevector <8 x float> %0, <8 x float> undef,
;;         <4 x i32> <i32 4, i32 5, i32 6, i32 7>
;;
;;  %cospa = alloca <4 x float>
;;  %sa = call <4 x float> @__acml_sincosf4(<4 x float> * %cospa, <4 x float> %a)
;;
;;  %cospb = alloca <4 x float>
;;  %sb = call <4 x float> @__acml_sincosf4(<4 x float> * %cospb, <4 x float> %b)
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
define(`acml_define_x',`
  define <$5 x $1> @__acml_sin$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @___acml_sin$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_asin$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @___acml_asin$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_cos$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @___acml_cos$2, %0)
    ret <$5 x $1> %ret
  }
  define void @__acml_sincos$4(<$5 x $1>,<$5 x $1>*,<$5 x $1>*) nounwind readnone alwaysinline 
  {
;;    call void @___acml_sincos$2(<$5 x $1> %0, <$5 x $1> * %1, <$5 x $1> * %2)
    %s = call <$5 x $1> @__acml_sin$4(<$5 x $1> %0)
    %c = call <$5 x $1> @__acml_cos$4(<$5 x $1> %0)
    store <$5 x $1> %s, <$5 x $1> * %1
    store <$5 x $1> %c, <$5 x $1> * %2
    ret void
  }
  define <$5 x $1> @__acml_tan$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @___acml_tan$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_atan$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @___acml_atan$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_atan2$4(<$5 x $1>,<$5 x $1>) nounwind readnone alwaysinline {
    binary$3to$5(ret, $1, @___acml_atan2$2, %0, %1)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_exp$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @___acml_exp$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_log$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @___acml_log$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_pow$4(<$5 x $1>,<$5 x $1>) nounwind readnone alwaysinline {
    binary$3to$5(ret, $1, @___acml_pow$2, %0, %1)
    ret <$5 x $1> %ret
  }
')


;;;;;;;;;;;;
;; Genearted from acml_lib via
;;   clang -std=c++11 -mavx -O3 acml_dispatch.cpp  -emit-llvm -c &&  llvm-dis acml_dispatch.bc
;; and copied here, with attribute change
;; attributes #0 = { alwaysinline nounwind readnone}
;; attributes #1 = { alwaysinline nounwind readnone}
;;;;;;;;;;;;;;;

define(`acml_dispatch',`
define <4 x float> @___acml_sinf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_sinf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_sinf(<4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @___acml_cosf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_cosf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_cosf(<4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @___acml_tanf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_tanf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_tanf(<4 x float>) #1

; Function Attrs: uwtable
define void @___acml_sincosf4(<4 x float> %x, <4 x float>* %ys, <4 x float>* %yc) #0 {
  tail call void @amd_vrs4_sincosf(<4 x float> %x, <4 x float>* %ys, <4 x float>* %yc)
  ret void
}

declare void @amd_vrs4_sincosf(<4 x float>, <4 x float>*, <4 x float>*) #1

; Function Attrs: uwtable
define <4 x float> @___acml_expf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_expf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_expf(<4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @___acml_logf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_logf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_logf(<4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @___acml_powf4(<4 x float> %x, <4 x float> %y) #0 {
  %1 = tail call <4 x float> @amd_vrs4_powf(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_powf(<4 x float>, <4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @___acml_asinf4(<4 x float> %x) #0 {
  %1 = extractelement <4 x float> %x, i32 0
  %2 = tail call float @amd_asinf(float %1)
  %3 = extractelement <4 x float> %x, i32 1
  %4 = tail call float @amd_asinf(float %3)
  %5 = insertelement <4 x float> undef, float %4, i32 1
  %6 = extractelement <4 x float> %x, i32 2
  %7 = tail call float @amd_asinf(float %6)
  %8 = insertelement <4 x float> %5, float %7, i32 0
  %9 = extractelement <4 x float> %x, i32 3
  %10 = tail call float @amd_asinf(float %9)
  %11 = insertelement <4 x float> %8, float %10, i32 3
  ret <4 x float> %11
}

declare float @amd_asinf(float) #1

; Function Attrs: uwtable
define <4 x float> @___acml_acosf4(<4 x float> %x) #0 {
  %1 = extractelement <4 x float> %x, i32 0
  %2 = tail call float @amd_acosf(float %1)
  %3 = extractelement <4 x float> %x, i32 1
  %4 = tail call float @amd_acosf(float %3)
  %5 = insertelement <4 x float> undef, float %4, i32 1
  %6 = extractelement <4 x float> %x, i32 2
  %7 = tail call float @amd_acosf(float %6)
  %8 = insertelement <4 x float> %5, float %7, i32 0
  %9 = extractelement <4 x float> %x, i32 3
  %10 = tail call float @amd_acosf(float %9)
  %11 = insertelement <4 x float> %8, float %10, i32 3
  ret <4 x float> %11
}

declare float @amd_acosf(float) #1

; Function Attrs: uwtable
define <4 x float> @___acml_atanf4(<4 x float> %x) #0 {
  %1 = extractelement <4 x float> %x, i32 0
  %2 = tail call float @amd_atanf(float %1)
  %3 = extractelement <4 x float> %x, i32 1
  %4 = tail call float @amd_atanf(float %3)
  %5 = insertelement <4 x float> undef, float %4, i32 1
  %6 = extractelement <4 x float> %x, i32 2
  %7 = tail call float @amd_atanf(float %6)
  %8 = insertelement <4 x float> %5, float %7, i32 0
  %9 = extractelement <4 x float> %x, i32 3
  %10 = tail call float @amd_atanf(float %9)
  %11 = insertelement <4 x float> %8, float %10, i32 3
  ret <4 x float> %11
}

declare float @amd_atanf(float) #1

; Function Attrs: uwtable
define <4 x float> @___acml_atan2f4(<4 x float> %x, <4 x float> %y) #0 {
  %1 = extractelement <4 x float> %x, i32 0
  %2 = extractelement <4 x float> %y, i32 0
  %3 = tail call float @amd_atan2f(float %1, float %2)
  %4 = insertelement <4 x float> undef, float %3, i32 0
  %5 = extractelement <4 x float> %x, i32 1
  %6 = extractelement <4 x float> %y, i32 1
  %7 = tail call float @amd_atan2f(float %5, float %6)
  %8 = insertelement <4 x float> %4, float %7, i32 1
  %9 = extractelement <4 x float> %x, i32 2
  %10 = extractelement <4 x float> %y, i32 2
  %11 = tail call float @amd_atan2f(float %9, float %10)
  %12 = insertelement <4 x float> %8, float %11, i32 2
  %13 = extractelement <4 x float> %x, i32 3
  %14 = extractelement <4 x float> %y, i32 3
  %15 = tail call float @amd_atan2f(float %13, float %14)
  %16 = insertelement <4 x float> %12, float %15, i32 3
  ret <4 x float> %16
}

declare float @amd_atan2f(float, float) #1

; Function Attrs: uwtable
define <2 x double> @___acml_sin2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_sin(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_sin(<2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @___acml_cos2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_cos(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_cos(<2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @___acml_tan2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_tan(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_tan(<2 x double>) #1

; Function Attrs: uwtable
define void @___acml_sincos2(<2 x double> %x, <2 x double>* %ys, <2 x double>* %yc) #0 {
  tail call void @amd_vrd2_sincos(<2 x double> %x, <2 x double>* %ys, <2 x double>* %yc)
  ret void
}

declare void @amd_vrd2_sincos(<2 x double>, <2 x double>*, <2 x double>*) #1

; Function Attrs: uwtable
define <2 x double> @___acml_exp2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_exp(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_exp(<2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @___acml_log2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_log(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_log(<2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @___acml_pow2(<2 x double> %x, <2 x double> %y) #0 {
  %1 = tail call <2 x double> @amd_vrd2_pow(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_pow(<2 x double>, <2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @___acml_asin2(<2 x double> %x) #0 {
  %1 = extractelement <2 x double> %x, i32 0
  %2 = tail call double @amd_asin(double %1)
  %3 = insertelement <2 x double> undef, double %2, i32 0
  %4 = extractelement <2 x double> %x, i32 1
  %5 = tail call double @amd_asin(double %4)
  %6 = insertelement <2 x double> %3, double %5, i32 1
  ret <2 x double> %6
}

declare double @amd_asin(double) #1

; Function Attrs: uwtable
define <2 x double> @___acml_acos2(<2 x double> %x) #0 {
  %1 = extractelement <2 x double> %x, i32 0
  %2 = tail call double @amd_acos(double %1)
  %3 = insertelement <2 x double> undef, double %2, i32 0
  %4 = extractelement <2 x double> %x, i32 1
  %5 = tail call double @amd_acos(double %4)
  %6 = insertelement <2 x double> %3, double %5, i32 1
  ret <2 x double> %6
}

declare double @amd_acos(double) #1

; Function Attrs: uwtable
define <2 x double> @___acml_atan2(<2 x double> %x) #0 {
  %1 = extractelement <2 x double> %x, i32 0
  %2 = tail call double @amd_atan(double %1)
  %3 = insertelement <2 x double> undef, double %2, i32 0
  %4 = extractelement <2 x double> %x, i32 1
  %5 = tail call double @amd_atan(double %4)
  %6 = insertelement <2 x double> %3, double %5, i32 1
  ret <2 x double> %6
}

declare double @amd_atan(double) #1

; Function Attrs: uwtable
define <2 x double> @___acml_atan22(<2 x double> %x, <2 x double> %y) #0 {
  %1 = extractelement <2 x double> %x, i32 0
  %2 = extractelement <2 x double> %y, i32 0
  %3 = tail call double @amd_atan2(double %1, double %2)
  %4 = insertelement <2 x double> undef, double %3, i32 0
  %5 = extractelement <2 x double> %x, i32 1
  %6 = extractelement <2 x double> %y, i32 1
  %7 = tail call double @amd_atan2(double %5, double %6)
  %8 = insertelement <2 x double> %4, double %7, i32 1
  ret <2 x double> %8
}

declare double @amd_atan2(double, double) #1

; Function Attrs: uwtable
define <8 x float> @___acml_sinf8(<8 x float> %x) #0 {
  %1 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %2 = tail call <4 x float> @amd_vrs4_sinf(<4 x float> %1)
  %3 = shufflevector <4 x float> %2, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %4 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %5 = tail call <4 x float> @amd_vrs4_sinf(<4 x float> %4)
  %6 = shufflevector <4 x float> %5, <4 x float> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3>
  %7 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %6, <8 x float> %3
  ret <8 x float> %7
}

; Function Attrs: uwtable
define <8 x float> @___acml_cosf8(<8 x float> %x) #0 {
  %1 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %2 = tail call <4 x float> @amd_vrs4_cosf(<4 x float> %1)
  %3 = shufflevector <4 x float> %2, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %4 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %5 = tail call <4 x float> @amd_vrs4_cosf(<4 x float> %4)
  %6 = shufflevector <4 x float> %5, <4 x float> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3>
  %7 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %6, <8 x float> %3
  ret <8 x float> %7
}

; Function Attrs: uwtable
define <8 x float> @___acml_tanf8(<8 x float> %x) #0 {
  %1 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %2 = tail call <4 x float> @amd_vrs4_tanf(<4 x float> %1)
  %3 = shufflevector <4 x float> %2, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %4 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %5 = tail call <4 x float> @amd_vrs4_tanf(<4 x float> %4)
  %6 = shufflevector <4 x float> %5, <4 x float> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3>
  %7 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %6, <8 x float> %3
  ret <8 x float> %7
}

; Function Attrs: uwtable
define <8 x float> @___acml_asinf8(<8 x float> %x) #0 {
  %1 = extractelement <8 x float> %x, i32 0
  %2 = tail call float @amd_asinf(float %1)
  %3 = extractelement <8 x float> %x, i32 1
  %4 = tail call float @amd_asinf(float %3)
  %5 = insertelement <8 x float> undef, float %4, i32 1
  %6 = extractelement <8 x float> %x, i32 2
  %7 = tail call float @amd_asinf(float %6)
  %8 = insertelement <8 x float> %5, float %7, i32 0
  %9 = extractelement <8 x float> %x, i32 3
  %10 = tail call float @amd_asinf(float %9)
  %11 = insertelement <8 x float> %8, float %10, i32 3
  %12 = extractelement <8 x float> %x, i32 4
  %13 = tail call float @amd_asinf(float %12)
  %14 = extractelement <8 x float> %x, i32 5
  %15 = tail call float @amd_asinf(float %14)
  %16 = insertelement <8 x float> undef, float %15, i32 5
  %17 = extractelement <8 x float> %x, i32 6
  %18 = tail call float @amd_asinf(float %17)
  %19 = insertelement <8 x float> %16, float %18, i32 4
  %20 = extractelement <8 x float> %x, i32 7
  %21 = tail call float @amd_asinf(float %20)
  %22 = insertelement <8 x float> %19, float %21, i32 7
  %23 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %22, <8 x float> %11
  ret <8 x float> %23
}

; Function Attrs: uwtable
define <8 x float> @___acml_acosf8(<8 x float> %x) #0 {
  %1 = extractelement <8 x float> %x, i32 0
  %2 = tail call float @amd_acosf(float %1)
  %3 = extractelement <8 x float> %x, i32 1
  %4 = tail call float @amd_acosf(float %3)
  %5 = insertelement <8 x float> undef, float %4, i32 1
  %6 = extractelement <8 x float> %x, i32 2
  %7 = tail call float @amd_acosf(float %6)
  %8 = insertelement <8 x float> %5, float %7, i32 0
  %9 = extractelement <8 x float> %x, i32 3
  %10 = tail call float @amd_acosf(float %9)
  %11 = insertelement <8 x float> %8, float %10, i32 3
  %12 = extractelement <8 x float> %x, i32 4
  %13 = tail call float @amd_acosf(float %12)
  %14 = extractelement <8 x float> %x, i32 5
  %15 = tail call float @amd_acosf(float %14)
  %16 = insertelement <8 x float> undef, float %15, i32 5
  %17 = extractelement <8 x float> %x, i32 6
  %18 = tail call float @amd_acosf(float %17)
  %19 = insertelement <8 x float> %16, float %18, i32 4
  %20 = extractelement <8 x float> %x, i32 7
  %21 = tail call float @amd_acosf(float %20)
  %22 = insertelement <8 x float> %19, float %21, i32 7
  %23 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %22, <8 x float> %11
  ret <8 x float> %23
}

; Function Attrs: uwtable
define <8 x float> @___acml_atanf8(<8 x float> %x) #0 {
  %1 = extractelement <8 x float> %x, i32 0
  %2 = tail call float @amd_atanf(float %1)
  %3 = extractelement <8 x float> %x, i32 1
  %4 = tail call float @amd_atanf(float %3)
  %5 = insertelement <8 x float> undef, float %4, i32 1
  %6 = extractelement <8 x float> %x, i32 2
  %7 = tail call float @amd_atanf(float %6)
  %8 = insertelement <8 x float> %5, float %7, i32 0
  %9 = extractelement <8 x float> %x, i32 3
  %10 = tail call float @amd_atanf(float %9)
  %11 = insertelement <8 x float> %8, float %10, i32 3
  %12 = extractelement <8 x float> %x, i32 4
  %13 = tail call float @amd_atanf(float %12)
  %14 = extractelement <8 x float> %x, i32 5
  %15 = tail call float @amd_atanf(float %14)
  %16 = insertelement <8 x float> undef, float %15, i32 5
  %17 = extractelement <8 x float> %x, i32 6
  %18 = tail call float @amd_atanf(float %17)
  %19 = insertelement <8 x float> %16, float %18, i32 4
  %20 = extractelement <8 x float> %x, i32 7
  %21 = tail call float @amd_atanf(float %20)
  %22 = insertelement <8 x float> %19, float %21, i32 7
  %23 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %22, <8 x float> %11
  ret <8 x float> %23
}

; Function Attrs: uwtable
define <8 x float> @___acml_expf8(<8 x float> %x) #0 {
  %1 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %2 = tail call <4 x float> @amd_vrs4_expf(<4 x float> %1)
  %3 = shufflevector <4 x float> %2, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %4 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %5 = tail call <4 x float> @amd_vrs4_expf(<4 x float> %4)
  %6 = shufflevector <4 x float> %5, <4 x float> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3>
  %7 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %6, <8 x float> %3
  ret <8 x float> %7
}

; Function Attrs: uwtable
define <8 x float> @___acml_logf8(<8 x float> %x) #0 {
  %1 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %2 = tail call <4 x float> @amd_vrs4_logf(<4 x float> %1)
  %3 = shufflevector <4 x float> %2, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %4 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %5 = tail call <4 x float> @amd_vrs4_logf(<4 x float> %4)
  %6 = shufflevector <4 x float> %5, <4 x float> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3>
  %7 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %6, <8 x float> %3
  ret <8 x float> %7
}

; Function Attrs: uwtable
define <8 x float> @___acml_powf8(<8 x float> %x, <8 x float> %y) #0 {
  %1 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %2 = shufflevector <8 x float> %y, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = tail call <4 x float> @amd_vrs4_powf(<4 x float> %1, <4 x float> %2)
  %4 = shufflevector <4 x float> %3, <4 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %5 = shufflevector <8 x float> %x, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %6 = shufflevector <8 x float> %y, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %7 = tail call <4 x float> @amd_vrs4_powf(<4 x float> %5, <4 x float> %6)
  %8 = shufflevector <4 x float> %7, <4 x float> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3>
  %9 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %8, <8 x float> %4
  ret <8 x float> %9
}

; Function Attrs: uwtable
define <8 x float> @___acml_atan2f8(<8 x float> %x, <8 x float> %y) #0 {
  %1 = extractelement <8 x float> %x, i32 0
  %2 = extractelement <8 x float> %y, i32 0
  %3 = tail call float @amd_atan2f(float %1, float %2)
  %4 = insertelement <8 x float> undef, float %3, i32 0
  %5 = extractelement <8 x float> %x, i32 1
  %6 = extractelement <8 x float> %y, i32 1
  %7 = tail call float @amd_atan2f(float %5, float %6)
  %8 = insertelement <8 x float> %4, float %7, i32 1
  %9 = extractelement <8 x float> %x, i32 2
  %10 = extractelement <8 x float> %y, i32 2
  %11 = tail call float @amd_atan2f(float %9, float %10)
  %12 = insertelement <8 x float> %8, float %11, i32 2
  %13 = extractelement <8 x float> %x, i32 3
  %14 = extractelement <8 x float> %y, i32 3
  %15 = tail call float @amd_atan2f(float %13, float %14)
  %16 = insertelement <8 x float> %12, float %15, i32 3
  %17 = extractelement <8 x float> %x, i32 4
  %18 = extractelement <8 x float> %y, i32 4
  %19 = tail call float @amd_atan2f(float %17, float %18)
  %20 = insertelement <8 x float> undef, float %19, i32 4
  %21 = extractelement <8 x float> %x, i32 5
  %22 = extractelement <8 x float> %y, i32 5
  %23 = tail call float @amd_atan2f(float %21, float %22)
  %24 = insertelement <8 x float> %20, float %23, i32 5
  %25 = extractelement <8 x float> %x, i32 6
  %26 = extractelement <8 x float> %y, i32 6
  %27 = tail call float @amd_atan2f(float %25, float %26)
  %28 = insertelement <8 x float> %24, float %27, i32 6
  %29 = extractelement <8 x float> %x, i32 7
  %30 = extractelement <8 x float> %y, i32 7
  %31 = tail call float @amd_atan2f(float %29, float %30)
  %32 = insertelement <8 x float> %28, float %31, i32 7
  %33 = select <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 true, i1 true, i1 true, i1 true>, <8 x float> %32, <8 x float> %16
  ret <8 x float> %33
}

; Function Attrs: uwtable
define void @___acml_sincosf8(<8 x float> %x, <8 x float>* %ys, <8 x float>* %yc) #0 {
  %1 = bitcast <8 x float>* %ys to <4 x float>*
  %2 = bitcast <8 x float>* %yc to <4 x float>*
  tail call void @amd_vrs4_sincosf(<4 x float> undef, <4 x float>* %1, <4 x float>* %2)
  %3 = getelementptr inbounds <8 x float>* %ys, i64 2
  %4 = bitcast <8 x float>* %3 to <4 x float>*
  %5 = getelementptr inbounds <8 x float>* %yc, i64 2
  %6 = bitcast <8 x float>* %5 to <4 x float>*
  tail call void @amd_vrs4_sincosf(<4 x float> undef, <4 x float>* %4, <4 x float>* %6)
  ret void
}

; Function Attrs: uwtable
define <4 x double> @___acml_sin4(<4 x double> %x) #0 {
  %1 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %2 = tail call <2 x double> @amd_vrd2_sin(<2 x double> %1)
  %3 = shufflevector <2 x double> %2, <2 x double> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %4 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %5 = tail call <2 x double> @amd_vrd2_sin(<2 x double> %4)
  %6 = shufflevector <2 x double> %5, <2 x double> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %7 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %6, <4 x double> %3
  ret <4 x double> %7
}

; Function Attrs: uwtable
define <4 x double> @___acml_cos4(<4 x double> %x) #0 {
  %1 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %2 = tail call <2 x double> @amd_vrd2_cos(<2 x double> %1)
  %3 = shufflevector <2 x double> %2, <2 x double> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %4 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %5 = tail call <2 x double> @amd_vrd2_cos(<2 x double> %4)
  %6 = shufflevector <2 x double> %5, <2 x double> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %7 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %6, <4 x double> %3
  ret <4 x double> %7
}

; Function Attrs: uwtable
define <4 x double> @___acml_tan4(<4 x double> %x) #0 {
  %1 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %2 = tail call <2 x double> @amd_vrd2_tan(<2 x double> %1)
  %3 = shufflevector <2 x double> %2, <2 x double> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %4 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %5 = tail call <2 x double> @amd_vrd2_tan(<2 x double> %4)
  %6 = shufflevector <2 x double> %5, <2 x double> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %7 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %6, <4 x double> %3
  ret <4 x double> %7
}

; Function Attrs: uwtable
define <4 x double> @___acml_asin4(<4 x double> %x) #0 {
  %1 = extractelement <4 x double> %x, i32 0
  %2 = tail call double @amd_asin(double %1)
  %3 = insertelement <4 x double> undef, double %2, i32 0
  %4 = extractelement <4 x double> %x, i32 1
  %5 = tail call double @amd_asin(double %4)
  %6 = insertelement <4 x double> %3, double %5, i32 1
  %7 = extractelement <4 x double> %x, i32 2
  %8 = tail call double @amd_asin(double %7)
  %9 = insertelement <4 x double> undef, double %8, i32 2
  %10 = extractelement <4 x double> %x, i32 3
  %11 = tail call double @amd_asin(double %10)
  %12 = insertelement <4 x double> %9, double %11, i32 3
  %13 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %12, <4 x double> %6
  ret <4 x double> %13
}

; Function Attrs: uwtable
define <4 x double> @___acml_acos4(<4 x double> %x) #0 {
  %1 = extractelement <4 x double> %x, i32 0
  %2 = tail call double @amd_acos(double %1)
  %3 = insertelement <4 x double> undef, double %2, i32 0
  %4 = extractelement <4 x double> %x, i32 1
  %5 = tail call double @amd_acos(double %4)
  %6 = insertelement <4 x double> %3, double %5, i32 1
  %7 = extractelement <4 x double> %x, i32 2
  %8 = tail call double @amd_acos(double %7)
  %9 = insertelement <4 x double> undef, double %8, i32 2
  %10 = extractelement <4 x double> %x, i32 3
  %11 = tail call double @amd_acos(double %10)
  %12 = insertelement <4 x double> %9, double %11, i32 3
  %13 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %12, <4 x double> %6
  ret <4 x double> %13
}

; Function Attrs: uwtable
define <4 x double> @___acml_atan4(<4 x double> %x) #0 {
  %1 = extractelement <4 x double> %x, i32 0
  %2 = tail call double @amd_atan(double %1)
  %3 = insertelement <4 x double> undef, double %2, i32 0
  %4 = extractelement <4 x double> %x, i32 1
  %5 = tail call double @amd_atan(double %4)
  %6 = insertelement <4 x double> %3, double %5, i32 1
  %7 = extractelement <4 x double> %x, i32 2
  %8 = tail call double @amd_atan(double %7)
  %9 = insertelement <4 x double> undef, double %8, i32 2
  %10 = extractelement <4 x double> %x, i32 3
  %11 = tail call double @amd_atan(double %10)
  %12 = insertelement <4 x double> %9, double %11, i32 3
  %13 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %12, <4 x double> %6
  ret <4 x double> %13
}

; Function Attrs: uwtable
define <4 x double> @___acml_exp4(<4 x double> %x) #0 {
  %1 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %2 = tail call <2 x double> @amd_vrd2_exp(<2 x double> %1)
  %3 = shufflevector <2 x double> %2, <2 x double> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %4 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %5 = tail call <2 x double> @amd_vrd2_exp(<2 x double> %4)
  %6 = shufflevector <2 x double> %5, <2 x double> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %7 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %6, <4 x double> %3
  ret <4 x double> %7
}

; Function Attrs: uwtable
define <4 x double> @___acml_log4(<4 x double> %x) #0 {
  %1 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %2 = tail call <2 x double> @amd_vrd2_log(<2 x double> %1)
  %3 = shufflevector <2 x double> %2, <2 x double> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %4 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %5 = tail call <2 x double> @amd_vrd2_log(<2 x double> %4)
  %6 = shufflevector <2 x double> %5, <2 x double> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %7 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %6, <4 x double> %3
  ret <4 x double> %7
}

; Function Attrs: uwtable
define <4 x double> @___acml_pow4(<4 x double> %x, <4 x double> %y) #0 {
  %1 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %2 = shufflevector <4 x double> %y, <4 x double> undef, <2 x i32> <i32 0, i32 1>
  %3 = tail call <2 x double> @amd_vrd2_pow(<2 x double> %1, <2 x double> %2)
  %4 = shufflevector <2 x double> %3, <2 x double> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %5 = shufflevector <4 x double> %x, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %6 = shufflevector <4 x double> %y, <4 x double> undef, <2 x i32> <i32 2, i32 3>
  %7 = tail call <2 x double> @amd_vrd2_pow(<2 x double> %5, <2 x double> %6)
  %8 = shufflevector <2 x double> %7, <2 x double> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %9 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %8, <4 x double> %4
  ret <4 x double> %9
}

; Function Attrs: uwtable
define <4 x double> @___acml_atan24(<4 x double> %x, <4 x double> %y) #0 {
  %1 = extractelement <4 x double> %x, i32 0
  %2 = extractelement <4 x double> %y, i32 0
  %3 = tail call double @amd_atan2(double %1, double %2)
  %4 = insertelement <4 x double> undef, double %3, i32 0
  %5 = extractelement <4 x double> %x, i32 1
  %6 = extractelement <4 x double> %y, i32 1
  %7 = tail call double @amd_atan2(double %5, double %6)
  %8 = insertelement <4 x double> %4, double %7, i32 1
  %9 = extractelement <4 x double> %x, i32 2
  %10 = extractelement <4 x double> %y, i32 2
  %11 = tail call double @amd_atan2(double %9, double %10)
  %12 = insertelement <4 x double> undef, double %11, i32 2
  %13 = extractelement <4 x double> %x, i32 3
  %14 = extractelement <4 x double> %y, i32 3
  %15 = tail call double @amd_atan2(double %13, double %14)
  %16 = insertelement <4 x double> %12, double %15, i32 3
  %17 = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x double> %16, <4 x double> %8
  ret <4 x double> %17
}

; Function Attrs: uwtable
define void @___acml_sincos4(<4 x double> %x, <4 x double>* %ys, <4 x double>* %yc) #0 {
  %1 = bitcast <4 x double>* %ys to <2 x double>*
  %2 = bitcast <4 x double>* %yc to <2 x double>*
  tail call void @amd_vrd2_sincos(<2 x double> undef, <2 x double>* %1, <2 x double>* %2)
  %3 = getelementptr inbounds <4 x double>* %ys, i64 1
  %4 = bitcast <4 x double>* %3 to <2 x double>*
  %5 = getelementptr inbounds <4 x double>* %yc, i64 1
  %6 = bitcast <4 x double>* %5 to <2 x double>*
  tail call void @amd_vrd2_sincos(<2 x double> undef, <2 x double>* %4, <2 x double>* %6)
  ret void
}

attributes #0 = { alwaysinline nounwind readnone}
attributes #1 = { alwaysinline nounwind readnone}
')

