;; svml

;; stub
define(`svmlf_stubs',`
  declare <$1 x float> @__svml_sinf(<$1 x float>) nounwind readnone alwaysinline
  declare <$1 x float> @__svml_asinf(<$1 x float>) nounwind readnone alwaysinline 
  declare <$1 x float> @__svml_cosf(<$1 x float>) nounwind readnone alwaysinline 
  declare void @__svml_sincosf(<$1 x float>, <$1 x float> *, <$1 x float> *) nounwind readnone alwaysinline 
  declare <$1 x float> @__svml_tanf(<$1 x float>) nounwind readnone alwaysinline 
  declare <$1 x float> @__svml_atanf(<$1 x float>) nounwind readnone alwaysinline 
  declare <$1 x float> @__svml_atan2f(<$1 x float>, <$1 x float>) nounwind readnone alwaysinline 
  declare <$1 x float> @__svml_expf(<$1 x float>) nounwind readnone alwaysinline 
  declare <$1 x float> @__svml_logf(<$1 x float>) nounwind readnone alwaysinline 
  declare <$1 x float> @__svml_powf(<$1 x float>, <$1 x float>) nounwind readnone alwaysinline 
')

define(`svmld_stubs',`
  declare <$1 x double> @__svml_sind(<$1 x double>) nounwind readnone alwaysinline 
  declare <$1 x double> @__svml_asind(<$1 x double>) nounwind readnone alwaysinline 
  declare <$1 x double> @__svml_cosd(<$1 x double>) nounwind readnone alwaysinline 
  declare void @__svml_sincosd(<$1 x double>, <$1 x double> *, <$1 x double> *) nounwind readnone alwaysinline 
  declare <$1 x double> @__svml_tand(<$1 x double>) nounwind readnone alwaysinline 
  declare <$1 x double> @__svml_atand(<$1 x double>) nounwind readnone alwaysinline 
  declare <$1 x double> @__svml_atan2d(<$1 x double>, <$1 x double>) nounwind readnone alwaysinline 
  declare <$1 x double> @__svml_expd(<$1 x double>) nounwind readnone alwaysinline 
  declare <$1 x double> @__svml_logd(<$1 x double>) nounwind readnone alwaysinline 
  declare <$1 x double> @__svml_powd(<$1 x double>, <$1 x double>) nounwind readnone alwaysinline 
')

;; single precision
define(`svmlf_declare',`
  declare <$1 x float> @__svml_sinf$1(<$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_asinf$1(<$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_cosf$1(<$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_sincosf$1(<$1 x float> *, <$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_tanf$1(<$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_atanf$1(<$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_atan2f$1(<$1 x float>, <$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_expf$1(<$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_logf$1(<$1 x float>) nounwind readnone
  declare <$1 x float> @__svml_powf$1(<$1 x float>, <$1 x float>) nounwind readnone
');



define(`svmlf_define',`
  define <$1 x float> @__svml_sinf(<$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_sinf$1(<$1 x float> %0)
    ret <$1 x float> %ret
  }
  define <$1 x float> @__svml_asinf(<$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_asinf$1(<$1 x float> %0)
    ret <$1 x float> %ret
  }

  define <$1 x float> @__svml_cosf(<$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_cosf$1(<$1 x float> %0)
    ret <$1 x float> %ret
  }

  define void @__svml_sincosf(<$1 x float>, <$1 x float> *, <$1 x float> *) nounwind readnone alwaysinline {
    %s = call <$1 x float> @__svml_sincosf$1(<$1 x float> * %2, <$1 x float> %0)
    store <$1 x float> %s, <$1 x float> * %1
    ret void
  }

  define <$1 x float> @__svml_tanf(<$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_tanf$1(<$1 x float> %0)
    ret <$1 x float> %ret
  }

  define <$1 x float> @__svml_atanf(<$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_atanf$1(<$1 x float> %0)
    ret <$1 x float> %ret
  }

  define <$1 x float> @__svml_atan2f(<$1 x float>, <$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_atan2f$1(<$1 x float> %0, <$1 x float> %1)
    ret <$1 x float> %ret
  }

  define <$1 x float> @__svml_expf(<$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_expf$1(<$1 x float> %0)
    ret <$1 x float> %ret
  }

  define <$1 x float> @__svml_logf(<$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_logf$1(<$1 x float> %0)
    ret <$1 x float> %ret
  }

  define <$1 x float> @__svml_powf(<$1 x float>, <$1 x float>) nounwind readnone alwaysinline {
    %ret = call <$1 x float> @__svml_powf$1(<$1 x float> %0, <$1 x float> %1)
    ret <$1 x float> %ret
  }
')

;; double precision
define(`svmld_declare',`
  declare <$1 x double> @__svml_sin$1(<$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_asin$1(<$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_cos$1(<$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_sincos$1(<$1 x double> *, <$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_tan$1(<$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_atan$1(<$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_atan2$1(<$1 x double>, <$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_exp$1(<$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_log$1(<$1 x double>) nounwind readnone
  declare <$1 x double> @__svml_pow$1(<$1 x double>, <$1 x double>) nounwind readnone
')

define(`svmld_define',`
  define <$1 x double> @__svml_sind(<$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_sin$1(<$1 x double> %0)
    ret <$1 x double> %ret
  }
  define <$1 x double> @__svml_asind(<$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_asin$1(<$1 x double> %0)
    ret <$1 x double> %ret
  }


  define <$1 x double> @__svml_cosd(<$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_cos$1(<$1 x double> %0)
    ret <$1 x double> %ret
  }

  define void @__svml_sincosd(<$1 x double>, <$1 x double> *, <$1 x double> *) nounwind readnone alwaysinline {
    %s = call <$1 x double> @__svml_sincos$1(<$1 x double> * %2, <$1 x double> %0)
    store <$1 x double> %s, <$1 x double> * %1
    ret void
  }

  define <$1 x double> @__svml_tand(<$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_tan$1(<$1 x double> %0)
    ret <$1 x double> %ret
  }

  define <$1 x double> @__svml_atand(<$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_atan$1(<$1 x double> %0)
    ret <$1 x double> %ret
  }

  define <$1 x double> @__svml_atan2d(<$1 x double>, <$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_atan2$1(<$1 x double> %0, <$1 x double> %1)
    ret <$1 x double> %ret
  }

  define <$1 x double> @__svml_expd(<$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_exp$1(<$1 x double> %0)
    ret <$1 x double> %ret
  }

  define <$1 x double> @__svml_logd(<$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_log$1(<$1 x double> %0)
    ret <$1 x double> %ret
  }

  define <$1 x double> @__svml_powd(<$1 x double>, <$1 x double>) nounwind readnone alwaysinline {
    %ret = call <$1 x double> @__svml_pow$1(<$1 x double> %0, <$1 x double> %1)
    ret <$1 x double> %ret
  }
')

;; need to implement smvld for 2xvectorWidth ...:w

define(`svmld2_define',`
  define <$1 x double> @__svml_sinxx(<$1 x double>) nounwind readnone alwaysinline {
    %v0 = shufflevector <$1 x double> %0, <$1 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
    %v1 = shufflevector <$1 x double> %0, <$1 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
    %ret0 = call <$2 x double> @__svml_sin$2(<$2 x double> %v0)
    %ret1 = call <$2 x double> @__svml_sin$2(<$2 x double> %v1)
    %ret  = shufflevector <$2 x double> %ret0, <$2 x double> %ret1, <$1 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
    ret <$1 x double> %ret
  }
')
