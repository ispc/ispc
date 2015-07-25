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
  declare <$3 x $1> @__acml_sin$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__acml_asin$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__acml_cos$2(<$3 x $1>) nounwind readnone
  declare void @__acml_sincos$2(<$3 x $1>,  <$3 x $1> *, <$3 x $1>*) nounwind readnone
  declare <$3 x $1> @__acml_tan$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__acml_atan$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__acml_atan2$2(<$3 x $1>, <$3 x $1>) nounwind readnone
  declare <$3 x $1> @__acml_exp$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__acml_log$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__acml_pow$2(<$3 x $1>, <$3 x $1>) nounwind readnone
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
    %ret = call <$3 x $1> @__acml_sin$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }
  define <$3 x $1> @__acml_asin$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__acml_asin$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_cos$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__acml_cos$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define void @__acml_sincos$4(<$3 x $1>, <$3 x $1> *, <$3 x $1> *) nounwind readnone alwaysinline {
    call void @__acml_sincos$2(<$3 x $1> %0, <$3 x $1> * %1, <$3 x $1> * %2)
    ret void
  }

  define <$3 x $1> @__acml_tan$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__acml_tan$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_atan$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__acml_atan$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_atan2$4(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__acml_atan2$2(<$3 x $1> %0, <$3 x $1> %1)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_exp$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__acml_exp$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_log$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__acml_log$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__acml_pow$4(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__acml_pow$2(<$3 x $1> %0, <$3 x $1> %1)
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
    unary$3to$5(ret, $1, @__acml_sin$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_asin$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__acml_asin$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_cos$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__acml_cos$2, %0)
    ret <$5 x $1> %ret
  }
  define void @__acml_sincos$4(<$5 x $1>,<$5 x $1>*,<$5 x $1>*) nounwind readnone alwaysinline 
  {
;;    call void @__acml_sincos$2(<$5 x $1> %0, <$5 x $1> * %1, <$5 x $1> * %2)
    %s = call <$5 x $1> @__acml_sin$4(<$5 x $1> %0)
    %c = call <$5 x $1> @__acml_cos$4(<$5 x $1> %0)
    store <$5 x $1> %s, <$5 x $1> * %1
    store <$5 x $1> %c, <$5 x $1> * %2
    ret void
  }
  define <$5 x $1> @__acml_tan$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__acml_tan$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_atan$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__acml_atan$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_atan2$4(<$5 x $1>,<$5 x $1>) nounwind readnone alwaysinline {
    binary$3to$5(ret, $1, @__acml_atan2$2, %0, %1)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_exp$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__acml_exp$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_log$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__acml_log$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__acml_pow$4(<$5 x $1>,<$5 x $1>) nounwind readnone alwaysinline {
    binary$3to$5(ret, $1, @__acml_pow$2, %0, %1)
    ret <$5 x $1> %ret
  }
')

; Function Attrs: uwtable
define <4 x float> @__acml_sinf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_sinf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_sinf(<4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @__acml_cosf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_cosf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_cosf(<4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @__acml_tanf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_tanf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_tanf(<4 x float>) #1

; Function Attrs: uwtable
define void @__acml_sincosf4(<4 x float> %x, <4 x float>* %ys, <4 x float>* %yc) #0 {
  tail call void @amd_vrs4_sincosf(<4 x float> %x, <4 x float>* %ys, <4 x float>* %yc)
  ret void
}

declare void @amd_vrs4_sincosf(<4 x float>, <4 x float>*, <4 x float>*) #1

; Function Attrs: uwtable
define <4 x float> @__acml_expf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_expf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_expf(<4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @__acml_logf4(<4 x float> %x) #0 {
  %1 = tail call <4 x float> @amd_vrs4_logf(<4 x float> %x)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_logf(<4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @__acml_powf4(<4 x float> %x, <4 x float> %y) #0 {
  %1 = tail call <4 x float> @amd_vrs4_powf(<4 x float> %x, <4 x float> %y)
  ret <4 x float> %1
}

declare <4 x float> @amd_vrs4_powf(<4 x float>, <4 x float>) #1

; Function Attrs: uwtable
define <4 x float> @__acml_asinf4(<4 x float> %x) #0 {
  %1 = extractelement <4 x float> %x, i32 0
  %2 = extractelement <4 x float> %x, i32 1
  %3 = extractelement <4 x float> %x, i32 2
  %4 = extractelement <4 x float> %x, i32 3
  %5 = tail call float @amd_asinf(float %1)
  %6 = tail call float @amd_asinf(float %2)
  %7 = insertelement <4 x float> undef, float %6, i32 1
  %8 = tail call float @amd_asinf(float %3)
  %9 = insertelement <4 x float> %7, float %8, i32 0
  %10 = tail call float @amd_asinf(float %4)
  %11 = insertelement <4 x float> %9, float %10, i32 3
  ret <4 x float> %11
}

declare float @amd_asinf(float) #1

; Function Attrs: uwtable
define <4 x float> @__acml_acosf4(<4 x float> %x) #0 {
  %1 = extractelement <4 x float> %x, i32 0
  %2 = extractelement <4 x float> %x, i32 1
  %3 = extractelement <4 x float> %x, i32 2
  %4 = extractelement <4 x float> %x, i32 3
  %5 = tail call float @amd_acosf(float %1)
  %6 = tail call float @amd_acosf(float %2)
  %7 = insertelement <4 x float> undef, float %6, i32 1
  %8 = tail call float @amd_acosf(float %3)
  %9 = insertelement <4 x float> %7, float %8, i32 0
  %10 = tail call float @amd_acosf(float %4)
  %11 = insertelement <4 x float> %9, float %10, i32 3
  ret <4 x float> %11
}

declare float @amd_acosf(float) #1

; Function Attrs: uwtable
define <4 x float> @__acml_atanf4(<4 x float> %x) #0 {
  %1 = extractelement <4 x float> %x, i32 0
  %2 = extractelement <4 x float> %x, i32 1
  %3 = extractelement <4 x float> %x, i32 2
  %4 = extractelement <4 x float> %x, i32 3
  %5 = tail call float @amd_atanf(float %1)
  %6 = tail call float @amd_atanf(float %2)
  %7 = insertelement <4 x float> undef, float %6, i32 1
  %8 = tail call float @amd_atanf(float %3)
  %9 = insertelement <4 x float> %7, float %8, i32 0
  %10 = tail call float @amd_atanf(float %4)
  %11 = insertelement <4 x float> %9, float %10, i32 3
  ret <4 x float> %11
}

declare float @amd_atanf(float) #1

; Function Attrs: uwtable
define <4 x float> @__acml_atan2f4(<4 x float> %x, <4 x float> %y) #0 {
  %1 = extractelement <4 x float> %x, i32 0
  %2 = extractelement <4 x float> %y, i32 0
  %3 = tail call float @amd_atan2f(float %1, float %2)
  %4 = insertelement <4 x float> undef, float %3, i32 0
  %5 = tail call float @amd_atan2f(float %1, float %2)
  %6 = insertelement <4 x float> %4, float %5, i32 1
  %7 = tail call float @amd_atan2f(float %1, float %2)
  %8 = insertelement <4 x float> %6, float %7, i32 2
  %9 = tail call float @amd_atan2f(float %1, float %2)
  %10 = insertelement <4 x float> %8, float %9, i32 3
  ret <4 x float> %10
}

declare float @amd_atan2f(float, float) #1

; Function Attrs: uwtable
define <2 x double> @__acml_sind2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_sin(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_sin(<2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @__acml_cosd2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_cos(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_cos(<2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @__acml_tand2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_tan(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_tan(<2 x double>) #1

; Function Attrs: uwtable
define void @__acml_sincosd2(<2 x double> %x, <2 x double>* %ys, <2 x double>* %yc) #0 {
  tail call void @amd_vrd2_sincos(<2 x double> %x, <2 x double>* %ys, <2 x double>* %yc)
  ret void
}

declare void @amd_vrd2_sincos(<2 x double>, <2 x double>*, <2 x double>*) #1

; Function Attrs: uwtable
define <2 x double> @__acml_expd2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_exp(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_exp(<2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @__acml_logd2(<2 x double> %x) #0 {
  %1 = tail call <2 x double> @amd_vrd2_log(<2 x double> %x)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_log(<2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @__acml_powd2(<2 x double> %x, <2 x double> %y) #0 {
  %1 = tail call <2 x double> @amd_vrd2_pow(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %1
}

declare <2 x double> @amd_vrd2_pow(<2 x double>, <2 x double>) #1

; Function Attrs: uwtable
define <2 x double> @__acml_asind2(<2 x double> %x) #0 {
  %1 = extractelement <2 x double> %x, i32 0
  %2 = extractelement <2 x double> %x, i32 1
  %3 = tail call double @amd_asin(double %1)
  %4 = insertelement <2 x double> undef, double %3, i32 0
  %5 = tail call double @amd_asin(double %2)
  %6 = insertelement <2 x double> %4, double %5, i32 1
  ret <2 x double> %6
}

declare double @amd_asin(double) #1

; Function Attrs: uwtable
define <2 x double> @__acml_acosd2(<2 x double> %x) #0 {
  %1 = extractelement <2 x double> %x, i32 0
  %2 = extractelement <2 x double> %x, i32 1
  %3 = tail call double @amd_acos(double %1)
  %4 = insertelement <2 x double> undef, double %3, i32 0
  %5 = tail call double @amd_acos(double %2)
  %6 = insertelement <2 x double> %4, double %5, i32 1
  ret <2 x double> %6
}

declare double @amd_acos(double) #1

; Function Attrs: uwtable
define <2 x double> @__acml_atand2(<2 x double> %x) #0 {
  %1 = extractelement <2 x double> %x, i32 0
  %2 = extractelement <2 x double> %x, i32 1
  %3 = tail call double @amd_atan(double %1)
  %4 = insertelement <2 x double> undef, double %3, i32 0
  %5 = tail call double @amd_atan(double %2)
  %6 = insertelement <2 x double> %4, double %5, i32 1
  ret <2 x double> %6
}

declare double @amd_atan(double) #1

; Function Attrs: uwtable
define <2 x double> @__acml_atan2d2(<2 x double> %x, <2 x double> %y) #0 {
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

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

