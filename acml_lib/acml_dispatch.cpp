#include "amdlibm.h"
#include <immintrin.h>
/*
 
 * single precision:
        "__acml_sinf",
        "__acml_cosf",
        "__acml_tanf",
        "__acml_sincosf",
        
        "__acml_expf",
        "__acml_logf",
        "__acml_powf",

        "__acml_asinf",   
        "__acml_acosf",  
        "__acml_atanf", 
        "__acml_atan2f",


 * double precision:
        "__acml_sind",
        "__acml_cosd",
        "__acml_tand",
        "__acml_sincosd",
        
        "__acml_expd",
        "__acml_logd",
        "__acml_powd",

        "__acml_asind"
        "__acml_acosd"
        "__acml_atand"
        "__acml_atan2d"


*/

/* SSE */
extern "C"
{
  union v2d
  {
    __m128d vec;
    struct {double x,y;};
  };
  union v4f
  {
    __m128 vec;
    struct {float x,y,z,w;};
  };


  // Source: https://github.com/searchivarius/BlogCode/tree/master/2014/5/14
#define MM_EXTRACT_DOUBLE(v,i) _mm_cvtsd_f64(_mm_shuffle_pd(v, v, _MM_SHUFFLE2(0, i)))
#define MM_EXTRACT_FLOAT(v,i)  _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, i)))

  // single precision implementation

  __m128 __acml_sinf4(__m128 x) { return amd_vrs4_sinf(x); }
  __m128 __acml_cosf4(__m128 x) { return amd_vrs4_cosf(x); }
  __m128 __acml_tanf4(__m128 x) { return amd_vrs4_tanf(x); }
  void __acml_sincosf4(__m128 x, __m128 *ys, __m128 *yc) { amd_vrs4_sincosf(x, ys, yc); }
  
  __m128 __acml_expf4(__m128 x) { return amd_vrs4_expf(x); }
  __m128 __acml_logf4(__m128 x) { return amd_vrs4_logf(x); }
  __m128 __acml_powf4(__m128 x, __m128 y) { return amd_vrs4_powf(x,y); }

  __m128 __acml_asinf4(__m128 x)
  {
    const float x0 = MM_EXTRACT_FLOAT(x,0);
    const float x1 = MM_EXTRACT_FLOAT(x,1);
    const float x2 = MM_EXTRACT_FLOAT(x,2);
    const float x3 = MM_EXTRACT_FLOAT(x,3);
    v4f res;
    res.x = amd_asinf(x0);
    res.y = amd_asinf(x1);
    res.x = amd_asinf(x2);
    res.w = amd_asinf(x3);
    return res.vec;
  }
  __m128 __acml_acosf4(__m128 x)
  {
    const float x0 = MM_EXTRACT_FLOAT(x,0);
    const float x1 = MM_EXTRACT_FLOAT(x,1);
    const float x2 = MM_EXTRACT_FLOAT(x,2);
    const float x3 = MM_EXTRACT_FLOAT(x,3);
    v4f res;
    res.x = amd_acosf(x0);
    res.y = amd_acosf(x1);
    res.x = amd_acosf(x2);
    res.w = amd_acosf(x3);
    return res.vec;
  }
  __m128 __acml_atanf4(__m128 x)
  {
    const float x0 = MM_EXTRACT_FLOAT(x,0);
    const float x1 = MM_EXTRACT_FLOAT(x,1);
    const float x2 = MM_EXTRACT_FLOAT(x,2);
    const float x3 = MM_EXTRACT_FLOAT(x,3);
    v4f res;
    res.x = amd_atanf(x0);
    res.y = amd_atanf(x1);
    res.x = amd_atanf(x2);
    res.w = amd_atanf(x3);
    return res.vec;
  }
  __m128 __acml_atan2f4(__m128 x, __m128 y)
  {
    v4f res;
    const double x0 = MM_EXTRACT_FLOAT(x,0);
    const double y0 = MM_EXTRACT_FLOAT(y,0);
    res.x = amd_atan2f(x0,y0);
    const double x1 = MM_EXTRACT_FLOAT(x,1);
    const double y1 = MM_EXTRACT_FLOAT(y,1);
    res.y = amd_atan2f(x0,y0);
    const double x2 = MM_EXTRACT_FLOAT(x,2);
    const double y2 = MM_EXTRACT_FLOAT(y,2);
    res.z = amd_atan2f(x0,y0);
    const double x3 = MM_EXTRACT_FLOAT(x,3);
    const double y3 = MM_EXTRACT_FLOAT(y,3);
    res.w = amd_atan2f(x0,y0);
    return res.vec;
  }
  
    // double precision implementation
  
  __m128d __acml_sin2(__m128d x) { return amd_vrd2_sin(x); }
  __m128d __acml_cos2(__m128d x) { return amd_vrd2_cos(x); }
  __m128d __acml_tan2(__m128d x) { return amd_vrd2_tan(x); }
  void __acml_sincos2(__m128d x, __m128d *ys, __m128d *yc) { amd_vrd2_sincos(x, ys, yc); }
  
  __m128d __acml_exp2(__m128d x) { return amd_vrd2_exp(x); }
  __m128d __acml_log2(__m128d x) { return amd_vrd2_log(x); }
  __m128d __acml_pow2(__m128d x, __m128d y) { return amd_vrd2_pow(x,y); }

  __m128d __acml_asin2(__m128d x)
  {
    const double x0 = MM_EXTRACT_DOUBLE(x,0);
    const double x1 = MM_EXTRACT_DOUBLE(x,1);
    v2d res;
    res.x = amd_asin(x0);
    res.y = amd_asin(x1);
    return res.vec;
  }
  __m128d __acml_acos2(__m128d x)
  {
    const double x0 = MM_EXTRACT_DOUBLE(x,0);
    const double x1 = MM_EXTRACT_DOUBLE(x,1);
    v2d res;
    res.x = amd_acos(x0);
    res.y = amd_acos(x1);
    return res.vec;
  }
  __m128d __acml_atan2(__m128d x)
  {
    const double x0 = MM_EXTRACT_DOUBLE(x,0);
    const double x1 = MM_EXTRACT_DOUBLE(x,1);
    v2d res;
    res.x = amd_atan(x0);
    res.y = amd_atan(x1);
    return res.vec;
  }
  __m128d __acml_atan22(__m128d x, __m128d y)
  {
    v2d res;
    const double x0 = MM_EXTRACT_DOUBLE(x,0);
    const double y0 = MM_EXTRACT_DOUBLE(y,0);
    res.x = amd_atan2(x0,y0);
    const double x1 = MM_EXTRACT_DOUBLE(x,1);
    const double y1 = MM_EXTRACT_DOUBLE(y,1);
    res.y = amd_atan2(x1,y1);
    return res.vec;
  }
#undef MM_EXTRACT_DOUBLE
#undef MM_EXTRACT_SINGLE
}

/* AVX */
extern "C"
{
  union v4d
  {
    __m256d vec;
    struct {__m128d x, y;};
  };
  union v8f
  {
    __m256 vec;
    struct {__m128 x,y;};
  };


  // single precision implementation
  
#define GENCALL(op1,op2)  \
__m256 __acml_##op1(__m256 x) { \
  v8f in{x},out; \
  out.x = __acml_##op2(in.x); \
  out.y = __acml_##op2(in.y); \
  return out.vec; \
}
#define GENCALL2(op1,op2)  \
__m256 __acml_##op1(__m256 x, __m256 y) { \
  v8f in0{x},in1{y},out; \
  out.x = __acml_##op2(in0.x,in1.x); \
  out.y = __acml_##op2(in0.y,in1.y); \
  return out.vec; \
}

GENCALL(sinf8,sinf4)
GENCALL(cosf8,cosf4)
GENCALL(tanf8,tanf4)
GENCALL(asinf8,asinf4)
GENCALL(acosf8,acosf4)
GENCALL(atanf8,atanf4)
GENCALL(expf8,expf4)
GENCALL(logf8,logf4)
GENCALL2(powf8,powf4)
void __acml_sincosf8(__m256 x, __m256 *ys, __m256 *yc) 
{ 
  __m128* ys_f = (__m128*)ys;
  __m128* yc_f = (__m128*)yc;
  v8f out;
  __acml_sincosf4(out.x, ys_f, yc_f); 
  __acml_sincosf4(out.y, ys_f+4, yc_f+4); 
}
#undef GENCALL
#undef GENCALL2

  // double precision implementation
  
#define GENCALL(op1,op2)  \
__m256d __acml_##op1(__m256d x) { \
  v4d in{x},out; \
  out.x = __acml_##op2(in.x); \
  out.y = __acml_##op2(in.y); \
  return out.vec; \
}
#define GENCALL2(op1,op2)  \
__m256d __acml_##op1(__m256d x, __m256d y) { \
  v4d in0{x},in1{y},out; \
  out.x = __acml_##op2(in0.x,in1.x); \
  out.y = __acml_##op2(in0.y,in1.y); \
  return out.vec; \
}

GENCALL(sin4,sin2)
GENCALL(cos4,cos2)
GENCALL(tan4,tan2)
GENCALL(asin4,asin2)
GENCALL(acos4,acos2)
GENCALL(atan4,atan2)
GENCALL(exp4,exp2)
GENCALL(log4,log2)
GENCALL2(pow4,pow2)

void __acml_sincos4(__m256d x, __m256d *ys, __m256d *yc) 
{ 
  __m128d* ys_f = (__m128d*)ys;
  __m128d* yc_f = (__m128d*)yc;
  v4d out;
  __acml_sincos2(out.x, ys_f, yc_f); 
  __acml_sincos2(out.y, ys_f+2, yc_f+2); 
}
#undef GENCALL
#undef GENCALL2


}
