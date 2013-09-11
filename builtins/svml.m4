;; svml

;; stubs
define(`svml_stubs',`
  declare <$2 x $1> @__svml_sin$3(<$2 x $1>) nounwind readnone alwaysinline
  declare <$2 x $1> @__svml_asin$3(<$2 x $1>) nounwind readnone alwaysinline 
  declare <$2 x $1> @__svml_cos$3(<$2 x $1>) nounwind readnone alwaysinline 
  declare void @__svml_sincos$3(<$2 x $1>, <$2 x $1> *, <$2 x $1> *) nounwind readnone alwaysinline 
  declare <$2 x $1> @__svml_tan$3(<$2 x $1>) nounwind readnone alwaysinline 
  declare <$2 x $1> @__svml_atan$3(<$2 x $1>) nounwind readnone alwaysinline 
  declare <$2 x $1> @__svml_atan2$3(<$2 x $1>, <$2 x $1>) nounwind readnone alwaysinline 
  declare <$2 x $1> @__svml_exp$3(<$2 x $1>) nounwind readnone alwaysinline 
  declare <$2 x $1> @__svml_log$3(<$2 x $1>) nounwind readnone alwaysinline 
  declare <$2 x $1> @__svml_pow$3(<$2 x $1>, <$2 x $1>) nounwind readnone alwaysinline 
')

;; decalre __svml calls
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

;; define native __svml calls
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


;; define x2 __svml calls
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
  declare void @__svml_sincos$4(<$5 x $1>,<$5 x $1>*,<$5 x $1>*) nounwind readnone alwaysinline 
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

