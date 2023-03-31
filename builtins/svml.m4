;;  Copyright (c) 2013-2023, Intel Corporation
;;
;;  SPDX-License-Identifier: BSD-3-Clause


;; svml macro

;; svml_stubs : stubs for svml calls
;; $1 - type ("float" or "double")
;; $2 - svml internal function suffix ("f" for float, "d" for double)
;; $3 - vector width
define(`svml_stubs',`
  declare <$3 x $1> @__svml_sin$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_asin$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_cos$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_acos$2(<$3 x $1>) nounwind readnone alwaysinline
  declare void @__svml_sincos$2(<$3 x $1>, i8 *, i8 *) nounwind alwaysinline
  declare <$3 x $1> @__svml_tan$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_atan$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_atan2$2(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_exp$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_log$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_pow$2(<$3 x $1>, <$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_sqrt$2(<$3 x $1>) nounwind readnone alwaysinline
  declare <$3 x $1> @__svml_invsqrt$2(<$3 x $1>) nounwind readnone alwaysinline
')

;; svml_declare : declaration of __svml_* intrinsics
;; $1 - type ("float" or "double")
;; $2 - __svml_* intrinsic function suffix
;;      float:  "f4"(sse) "f8"(avx) "f16"(avx512)
;;      double:  "2"(sse)  "4"(avx)   "8"(avx512)
;; $3 - vector width
define(`svml_declare',`
  %struct.__svml_sincos_ret$2 = type { <$3 x $1>, <$3 x $1> }
  declare <$3 x $1> @__svml_sin$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_asin$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_cos$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_acos$2(<$3 x $1>) nounwind readnone
  declare %struct.__svml_sincos_ret$2 @__svml_sincos$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_tan$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_atan$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_atan2$2(<$3 x $1>, <$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_exp$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_log$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_pow$2(<$3 x $1>, <$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_sqrt$2(<$3 x $1>) nounwind readnone
  declare <$3 x $1> @__svml_invsqrt$2(<$3 x $1>) nounwind readnone
');

;; defintition of __svml_* internal functions
;; $1 - type ("float" or "double")
;; $2 - __svml_* intrinsic function suffix
;;      float:  "f4"(xmm) "f8"(ymm) "f16"(zmm)
;;      double:  "2"(xmm)  "4"(ymm)   "8"(zmm)
;; $3 - vector width
;; $4 - svml internal function suffix ("f" for float, "d" for double)
define(`svml_define',`
  svml_declare($1, $2, $3)

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

  define <$3 x $1> @__svml_acos$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_acos$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define void @__svml_sincos$4(<$3 x $1>, i8 *, i8 *) nounwind alwaysinline {
    %ret = call %struct.__svml_sincos_ret$2 @__svml_sincos$2(<$3 x $1> %0)
    %sin = extractvalue %struct.__svml_sincos_ret$2 %ret, 0
    %cos = extractvalue %struct.__svml_sincos_ret$2 %ret, 1
    %ptr1 = bitcast i8* %1 to <$3 x $1>*
    %ptr2 = bitcast i8* %2 to <$3 x $1>*
    store <$3 x $1> %sin, <$3 x $1> * %ptr1
    store <$3 x $1> %cos, <$3 x $1> * %ptr2
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

  define <$3 x $1> @__svml_sqrt$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_sqrt$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }

  define <$3 x $1> @__svml_invsqrt$4(<$3 x $1>) nounwind readnone alwaysinline {
    %ret = call <$3 x $1> @__svml_invsqrt$2(<$3 x $1> %0)
    ret <$3 x $1> %ret
  }
')


;; svml_define_x : defintition of __svml_* internal functions operation on extended width
;; $1 - type ("float" or "double")
;; $2 - __svml_* intrinsic function suffix
;;      float:  "f4"(xmm) "f8"(ymm) "f16"(zmm)
;;      double:  "2"(xmm)  "4"(ymm)   "8"(zmm)
;; $3 - vector width
;; $4 - svml internal function suffix ("f" for float, "d" for double)
;; $5 - extended width, must be at least twice the native vector width
;;      contigent on existing of unary$3to$5 and binary$3to$5 macros

;; *todo*: in sincos call use __svml_sincos[f][2,4,8,16] call, e.g.
;;define void @__svml_sincosf(<8 x float>, i8 *,
;;                                    i8 *) nounwind alwaysinline {
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
;;  %ptr1 = bitcast i8 * %1 <8 x float> *
;;  store <8 x float> %sin, <8 x float> * %ptr1
;;
;;  %cosa = load <4 x float> * %cospa
;;  %cosb = load <4 x float> * %cospb
;;  %cos = shufflevector <4 x float> %cosa, <4 x float> %cosb,
;;         <8 x i32> <i32 0, i32 1, i32 2, i32 3,
;;                    i32 4, i32 5, i32 6, i32 7>
;;  %ptr2 = bitcast i8 * %2 <8 x float> *
;;  store <8 x float> %cos, <8 x float> * %ptr2
;;
;;  ret void
;;}
define(`svml_define_x',`
  svml_declare($1, $2, $3)

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
  define <$5 x $1> @__svml_acos$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_acos$2, %0)
    ret <$5 x $1> %ret
  }
  define void @__svml_sincos$4(<$5 x $1>,i8*,i8*) nounwind alwaysinline
  {
    %ptr1 = bitcast i8* %1 to <$5 x $1>*
    %ptr2 = bitcast i8* %2 to <$5 x $1>*
    %s = call <$5 x $1> @__svml_sin$4(<$5 x $1> %0)
    %c = call <$5 x $1> @__svml_cos$4(<$5 x $1> %0)
    store <$5 x $1> %s, <$5 x $1> * %ptr1
    store <$5 x $1> %c, <$5 x $1> * %ptr2
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
  define <$5 x $1> @__svml_sqrt$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_sqrt$2, %0)
    ret <$5 x $1> %ret
  }
  define <$5 x $1> @__svml_invsqrt$4(<$5 x $1>) nounwind readnone alwaysinline {
    unary$3to$5(ret, $1, @__svml_invsqrt$2, %0)
    ret <$5 x $1> %ret
  }
')

;; Based on RUNTIME macro (32 or 64) set SVML_SUFFIX to first or second argument.
;; $1 - 32 bit suffix
;; $2 - 64 bit suffix
define(`svml_set_suffix', `
  ifelse(RUNTIME, `32', `define(SVML_SUFFIX, _$1)',
         RUNTIME, `64', `define(SVML_SUFFIX, _$2)',`
    errprint(`ERROR: svml() call cannot handle runtime: 'RUNTIME)
    m4exit(`1')
  ')
')

;; svml() - define SVML implementation for float and double types.
;; The function requires WIDTH macro to be defined in the calling context.
;; $1 - ISA, either SSE2, SSE4, AVX1, AVX2, AVX512KNL, or AVX512SKX.
;;
;; A handy command to verify SVML implementation across all targets that support it:
;; alloy.py -r --only="current stability -O2 -O1 -O0" --only-targets="sse avx" --ispc-flags="--math-lib=svml" --compiler=icpx --time
;;
;; SVML has generic versions of all supported functions, which dispatches to ISA-specific versions.
;; The drawback is that dispatch trampoline takes a few cycles.
;; We use specific versions when possbile, i.e. when there's a full set of functions we need with a certain suffix.
define(`svml', `
  ifelse($1, `SSE2', `
    ;; there is no one-size-fits-all suffix for SSE2
    ;; so using generic version, which dispateches to the right one.
    ifelse(WIDTH, `4', `
      svml_define(float,f4,4,f)
      svml_define_x(double,2,2,d,4)
    ',
    WIDTH, `8', `
      svml_define_x(float,f4,4,f,8)
      svml_define_x(double,2,2,d,8)
    ', `
      errprint(`ERROR: svml() call cannot handle width: 'WIDTH` for ISA: '$1)
      m4exit(`1')
    ')
  ',
  $1, `SSE4', `
    ;; due to incremental nature of SSE2/SSE3/SSSE3/SSE4.1 there is no one-size-fits-all suffix,
    ;; so using generic version, which dispateches to the right one.
    ifelse(WIDTH, `4', `
      svml_define(float,f4,4,f)
      svml_define_x(double,2,2,d,4)
    ',
    WIDTH, `8', `
      svml_define_x(float,f4,4,f,8)
      svml_define_x(double,2,2,d,8)
    ',
    WIDTH, `16', `
      svml_define_x(float,f4,4,f,16)
      svml_define_x(double,2,2,d,16)
    ', `
      errprint(`ERROR: svml() call cannot handle width: 'WIDTH` for ISA: '$1)
      m4exit(`1')
    ')
  ',
  $1, `AVX1', `
    svml_set_suffix(`g9',`e9')
    ;; note, avx1-i32x4 is an alias for sse4-i32x4
    ifelse(WIDTH, `4', `
      svml_define(float,f4,4,f)
      svml_define(double,4,4,d)
    ',
    WIDTH, `8', `
      svml_define(float,f8`'SVML_SUFFIX,8,f)
      svml_define_x(double,4`'SVML_SUFFIX,4,d,8)
    ',
    WIDTH, `16', `
      svml_define_x(float,f8`'SVML_SUFFIX,8,f,16)
      svml_define_x(double,4`'SVML_SUFFIX,4,d,16)
    ', `
      errprint(`ERROR: svml() call cannot handle width: 'WIDTH` for ISA: '$1)
      m4exit(`1')
    ')
  ',
  $1, `AVX2', `
    svml_set_suffix(`s9',`l9')
    ifelse(WIDTH, `4', `
      svml_define(float,f4`'SVML_SUFFIX,4,f)
      svml_define(double,4`'SVML_SUFFIX,4,d)
    ',
    WIDTH, `8', `
      svml_define(float,f8`'SVML_SUFFIX,8,f)
      svml_define_x(double,4`'SVML_SUFFIX,4,d,8)
    ',
    WIDTH, `16', `
      svml_define_x(float,f8`'SVML_SUFFIX,8,f,16)
      svml_define_x(double,4`'SVML_SUFFIX,4,d,16)
    ',
    WIDTH, `32', `
      svml_define_x(float,f8`'SVML_SUFFIX,8,f,32)
      svml_define_x(double,4`'SVML_SUFFIX,4,d,32)
    ', `
      errprint(`ERROR: svml() call cannot handle width: 'WIDTH` for ISA: '$1)
      m4exit(`1')
    ')
  ',
  $1, `AVX512SKX', `
    svml_set_suffix(`x0',`z0')
    ifelse(WIDTH, `4', `
      svml_define(float,f4`'SVML_SUFFIX,4,f)
      svml_define(double,4`'SVML_SUFFIX,4,d)
    ',
    WIDTH, `8', `
      svml_define(float,f8`'SVML_SUFFIX,8,f)
      svml_define_x(double,4`'SVML_SUFFIX,4,d,8) ;; avoid zmm, so double pumping
    ',
    WIDTH, `16', `
      svml_define(float,f16`'SVML_SUFFIX,16,f)
      svml_define_x(double,8`'SVML_SUFFIX,8,d,16)
    ',
    WIDTH, `32', `
      svml_define_x(float,f16`'SVML_SUFFIX,16,f,32)
      svml_define_x(double,8`'SVML_SUFFIX,8,d,32)
    ',
    WIDTH, `64', `
      svml_define_x(float,f16`'SVML_SUFFIX,16,f,64)
      svml_define_x(double,8`'SVML_SUFFIX,8,d,64)
    ', `
      errprint(`ERROR: svml() call cannot handle width: 'WIDTH` for ISA: '$1)
      m4exit(`1')
    ')
  ',
  $1, `AVX512KNL', `
    svml_set_suffix(`a3',`b3')
    ifelse(WIDTH, `16', `
      svml_define(float,f16`'SVML_SUFFIX,16,f)
      svml_define_x(double,8`'SVML_SUFFIX,8,d,16)
    ', `
      errprint(`ERROR: svml() call cannot handle width: 'WIDTH` for ISA: '$1)
      m4exit(`1')
    ')
  ', `
    errprint(`ERROR: First svml() parameter is not properly defined: '$1)
    m4exit(`1')
  ')
')

