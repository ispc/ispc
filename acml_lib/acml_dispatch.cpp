#include "amdlibm.h"
#include <immintrin.h>
#include <cmath>
/*
 
 * single precision:
        "___acml_sinf",
        "___acml_cosf",
        "___acml_tanf",
        "___acml_sincosf",
        
        "___acml_expf",
        "___acml_logf",
        "___acml_powf",

        "___acml_asinf",   
        "___acml_acosf",  
        "___acml_atanf", 
        "___acml_atan2f",


 * double precision:
        "___acml_sind",
        "___acml_cosd",
        "___acml_tand",
        "___acml_sincosd",
        
        "___acml_expd",
        "___acml_logd",
        "___acml_powd",

        "___acml_asind"
        "___acml_acosd"
        "___acml_atand"
        "___acml_atan2d"


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



  // single precision implementation

  __m128 ___acml_sinf4(__m128 x) { return amd_vrs4_sinf(x); }
  __m128 ___acml_cosf4(__m128 x) { return amd_vrs4_cosf(x); }
  __m128 ___acml_tanf4(__m128 x) { return amd_vrs4_tanf(x); }
  void ___acml_sincosf4(__m128 x, __m128 *ys, __m128 *yc) { amd_vrs4_sincosf(x, ys, yc); }
  
  __m128 ___acml_expf4(__m128 x) { return amd_vrs4_expf(x); }
  __m128 ___acml_logf4(__m128 x) { return amd_vrs4_logf(x); }
  __m128 ___acml_powf4(__m128 x, __m128 y) { return amd_vrs4_powf(x,y); }

  __m128 ___acml_asinf4(__m128 x)
  {
    v4f in{x}, res;
    res.x = amd_asinf(in.x);
    res.y = amd_asinf(in.y);
    res.x = amd_asinf(in.z);
    res.w = amd_asinf(in.w);
    return res.vec;
  }
  __m128 ___acml_acosf4(__m128 x)
  {
    v4f in{x}, res;
    res.x = amd_acosf(in.x);
    res.y = amd_acosf(in.y);
    res.x = amd_acosf(in.z);
    res.w = amd_acosf(in.w);
    return res.vec;
  }
  __m128 ___acml_atanf4(__m128 x)
  {
    v4f in{x}, res;
    res.x = amd_atanf(in.x);
    res.y = amd_atanf(in.y);
    res.x = amd_atanf(in.z);
    res.w = amd_atanf(in.w);
    return res.vec;
  }
  __m128 ___acml_atan2f4(__m128 x, __m128 y)
  {
    v4f in0{x}, in1{y}, res;
    res.x = amd_atan2f(in0.x,in1.x);
    res.y = amd_atan2f(in0.y,in1.y);
    res.z = amd_atan2f(in0.z,in1.z);
    res.w = amd_atan2f(in0.w,in1.w);
    return res.vec;
  }
  
    // double precision implementation
  
  __m128d ___acml_sin2(__m128d x) { return amd_vrd2_sin(x); }
  __m128d ___acml_cos2(__m128d x) { return amd_vrd2_cos(x); }
  __m128d ___acml_tan2(__m128d x) { return amd_vrd2_tan(x); }
  void ___acml_sincos2(__m128d x, __m128d *ys, __m128d *yc) { amd_vrd2_sincos(x, ys, yc); }
  
  __m128d ___acml_exp2(__m128d x) { return amd_vrd2_exp(x); }
  __m128d ___acml_log2(__m128d x) { return amd_vrd2_log(x); }
  __m128d ___acml_pow2(__m128d x, __m128d y) { return amd_vrd2_pow(x,y); }

  __m128d ___acml_asin2(__m128d x)
  {
    v2d in{x}, res;
    res.x = amd_asin(in.x);
    res.y = amd_asin(in.y);
    return res.vec;
  }
  __m128d ___acml_acos2(__m128d x)
  {
    v2d in{x}, res;
    res.x = amd_acos(in.x);
    res.y = amd_acos(in.y);
    return res.vec;
  }
  __m128d ___acml_atan2(__m128d x)
  {
    v2d in{x}, res;
    res.x = amd_atan(in.x);
    res.y = amd_atan(in.y);
    return res.vec;
  }
  __m128d ___acml_atan22(__m128d x, __m128d y)
  {
    v2d in0{x},in1{y},res;
    res.x = amd_atan2(in0.x,in1.x);
    res.y = amd_atan2(in0.y,in1.y);
    return res.vec;
  }
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
__m256 ___acml_##op1(__m256 x) { \
  v8f in{x},out; \
  out.x = ___acml_##op2(in.x); \
  out.y = ___acml_##op2(in.y); \
  return out.vec; \
}
#define GENCALL2(op1,op2)  \
__m256 ___acml_##op1(__m256 x, __m256 y) { \
  v8f in0{x},in1{y},out; \
  out.x = ___acml_##op2(in0.x,in1.x); \
  out.y = ___acml_##op2(in0.y,in1.y); \
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
GENCALL2(atan2f8,atan2f4)
void ___acml_sincosf8(__m256 x, __m256 *ys, __m256 *yc) 
{ 
  __m128* ys_f = (__m128*)ys;
  __m128* yc_f = (__m128*)yc;
  v8f out;
  ___acml_sincosf4(out.x, ys_f, yc_f); 
  ___acml_sincosf4(out.y, ys_f+4, yc_f+4); 
}
#undef GENCALL
#undef GENCALL2

  // double precision implementation
  
#define GENCALL(op1,op2)  \
__m256d ___acml_##op1(__m256d x) { \
  v4d in{x},out; \
  out.x = ___acml_##op2(in.x); \
  out.y = ___acml_##op2(in.y); \
  return out.vec; \
}
#define GENCALL2(op1,op2)  \
__m256d ___acml_##op1(__m256d x, __m256d y) { \
  v4d in0{x},in1{y},out; \
  out.x = ___acml_##op2(in0.x,in1.x); \
  out.y = ___acml_##op2(in0.y,in1.y); \
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
GENCALL2(atan24,atan22)

void ___acml_sincos4(__m256d x, __m256d *ys, __m256d *yc) 
{ 
  __m128d* ys_f = (__m128d*)ys;
  __m128d* yc_f = (__m128d*)yc;
  v4d out;
  ___acml_sincos2(out.x, ys_f, yc_f); 
  ___acml_sincos2(out.y, ys_f+2, yc_f+2); 
}
#undef GENCALL
#undef GENCALL2


}
