/**
  Copyright (c) 2010-2012, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <immintrin.h>
#include <zmmintrin.h>

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#define PRE_ALIGN(x)  /*__declspec(align(x))*/
#define POST_ALIGN(x)  
#define roundf(x) (floorf(x + .5f))
#define round(x) (floor(x + .5))
#else
#define FORCEINLINE __forceinline
#define PRE_ALIGN(x)
#define POST_ALIGN(x)  __attribute__ ((aligned(x)))
#endif

#define KNC 1
#if 0
extern "C" 
{
  int printf(const unsigned char *, ...);
  int puts(unsigned char *);
  unsigned int putchar(unsigned int);
  int fflush(void *);
  uint8_t *memcpy(uint8_t *, uint8_t *, uint64_t);
  uint8_t *memset(uint8_t *, uint8_t, uint64_t);
  void memset_pattern16(void *, const void *, uint64_t);
}
#endif

typedef float __vec1_f;
typedef double __vec1_d;
typedef int8_t __vec1_i8;
typedef int16_t __vec1_i16;
typedef int32_t __vec1_i32;
typedef int64_t __vec1_i64;

struct __vec8_i1 {
    __vec8_i1() { }
    __vec8_i1(const __mmask8 &vv) : v(vv) { }
    __vec8_i1(bool v0, bool v1, bool v2, bool v3,
              bool v4, bool v5, bool v6, bool v7) {
        v = ((v0 & 1) |
             ((v1 & 1) << 1) |
             ((v2 & 1) << 2) |
             ((v3 & 1) << 3) |
             ((v4 & 1) << 4) |
             ((v5 & 1) << 5) |
             ((v6 & 1) << 6) |
             ((v7 & 1) << 7) );
    }
             
    __mmask8 v;
    FORCEINLINE operator __mmask8() const { return v; }//0xFF & v; }
};


template <typename T>
struct vec8 {
    vec8() { }
    vec8(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {
        data[0] = v0;        data[1] = v1;        data[2] = v2;        data[3] = v3;
        data[4] = v4;        data[5] = v5;        data[6] = v6;        data[7] = v7;
    }
    T data[8]; 
    FORCEINLINE const T& operator[](const int i) const { return data[i]; }
    FORCEINLINE       T& operator[](const int i)       { return data[i]; }
};

/****************/

struct PRE_ALIGN(32) __vec8_i32  
{
#ifdef __ZMM64BIT__
  __m512i _data;
  FORCEINLINE __vec8_i32(const __m512i &in) : _data(in) {}
  FORCEINLINE operator __m512i() const   { return _data; }
#else /* __ZMM64BIT__ */
  typedef int32_t  _v8si  __attribute__((vector_size(32)));
  _v8si _data;
  FORCEINLINE __vec8_i32(const __m512i &in) 
  {
    _mm512_mask_extpackstorelo_epi32((__m512i*)&_data,  0xFF, in, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  }
  FORCEINLINE operator __m512i() const   
  { 
    return _mm512_extloadunpacklo_epi32(_mm512_setzero_epi32(), (uint8_t*)&_data, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE); 
  }
#endif /* __ZMM64BIT__ */
  
  __vec8_i32() { }
  FORCEINLINE __vec8_i32(int32_t v0, int32_t v1, int32_t v2, int32_t v3, 
      int32_t v4, int32_t v5, int32_t v6, int32_t v7) 
  {
    const __m512i v  = _mm512_set_16to16_pi(0,0,0,0,0,0,0,0, v7, v6, v5, v4, v3, v2, v1, v0);
    *this = __vec8_i32(v);
  }

  FORCEINLINE const int32_t& operator[](const int i) const {  return ((int32_t*)this)[i]; }
  FORCEINLINE       int32_t& operator[](const int i)       {  return ((int32_t*)this)[i]; }
} POST_ALIGN(32);

PRE_ALIGN(32) struct __vec8_f 
{
#ifdef __ZMM64BIT__
  __m512 _data;
  FORCEINLINE __vec8_f(const __m512 &in) : _data(in) {}
  FORCEINLINE operator __m512() const   { return _data; }
#else /* __ZMM64BIT__ */
  typedef float  _v8sf  __attribute__((vector_size(32)));
  _v8sf _data;
  FORCEINLINE __vec8_f(const __m512 &in) 
  {
    _mm512_mask_extpackstorelo_ps((__m512*)&_data,  0xFF, in, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
  }
  FORCEINLINE operator __m512() const   
  { 
    return _mm512_extloadunpacklo_ps(_mm512_setzero_ps(), (uint8_t*)&_data, _MM_UPCONV_PS_NONE, _MM_HINT_NONE); 
  }
#endif /* __ZMM64BIT__ */
  FORCEINLINE __vec8_f() { }
  FORCEINLINE __vec8_f(float v0, float v1, float v2, float v3, 
                       float v4, float v5, float v6, float v7) 
  {
    const __m512 v  = _mm512_set_16to16_ps(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, v7, v6, v5, v4, v3, v2, v1, v0);
    *this = __vec8_f(v);
  }

  FORCEINLINE const float& operator[](const int i) const {  return ((float*)this)[i]; }
  FORCEINLINE       float& operator[](const int i)       {  return ((float*)this)[i]; }
} POST_ALIGN(32);

struct PRE_ALIGN(64) __vec8_d 
{
    __m512d v;
    FORCEINLINE __vec8_d() : v(_mm512_undefined_pd()) {}
    FORCEINLINE __vec8_d(const __m512d _v) : v(_v) {}
    FORCEINLINE __vec8_d(const __vec8_d &o) : v(o.v) {}
    FORCEINLINE __vec8_d& operator =(const __vec8_d &o) { v=o.v; return *this; }
    FORCEINLINE operator __m512d() const { return v; }
    FORCEINLINE __vec8_d(double v00, double v01, double v02, double v03, 
                          double v04, double v05, double v06, double v07) :
        v ( _mm512_set_8to8_pd(v07, v06, v05, v04, v03, v02, v01, v00) ) {}
    FORCEINLINE const double& operator[](const int i) const {  return ((double*)this)[i]; }
    FORCEINLINE       double& operator[](const int i)       {  return ((double*)this)[i]; }
} POST_ALIGN(64);

/****************/

PRE_ALIGN(64) struct __vec8_i64  : public vec8<int64_t> { 
    __vec8_i64() { }
    __vec8_i64(int64_t v0, int64_t v1, int64_t v2, int64_t v3, 
               int64_t v4, int64_t v5, int64_t v6, int64_t v7) 
        : vec8<int64_t>(v0, v1, v2, v3, v4, v5, v6, v7) { }
} POST_ALIGN(64);

PRE_ALIGN(16) struct __vec8_i8   : public vec8<int8_t> { 
    __vec8_i8() { }
    __vec8_i8(int8_t v0, int8_t v1, int8_t v2, int8_t v3, 
               int8_t v4, int8_t v5, int8_t v6, int8_t v7)
        : vec8<int8_t>(v0, v1, v2, v3, v4, v5, v6, v7) { }
} POST_ALIGN(16);

PRE_ALIGN(32) struct __vec8_i16  : public vec8<int16_t> { 
    __vec8_i16() { }
    __vec8_i16(int16_t v0, int16_t v1, int16_t v2, int16_t v3, 
                int16_t v4, int16_t v5, int16_t v6, int16_t v7) 
        : vec8<int16_t>(v0, v1, v2, v3, v4, v5, v6, v7) { }
} POST_ALIGN(32);

static inline int32_t __extract_element(__vec8_i32, int);


///////////////////////////////////////////////////////////////////////////
// macros...

#define UNARY_OP(TYPE, NAME, OP)            \
static FORCEINLINE TYPE NAME(TYPE v) {      \
    TYPE ret;                               \
    for (int i = 0; i < 8; ++i)            \
        ret[i] = OP(v[i]);              \
    return ret;                             \
}

#define BINARY_OP(TYPE, NAME, OP)                               \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                  \
    TYPE ret;                                                   \
   for (int i = 0; i < 8; ++i)                                 \
       ret[i] = a[i] OP b[i];                             \
   return ret;                                                   \
}

#define BINARY_OP_CAST(TYPE, CAST, NAME, OP)                        \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   for (int i = 0; i < 8; ++i)                                     \
       ret[i] = (CAST)(a[i]) OP (CAST)(b[i]);                 \
   return ret;                                                      \
}

#define BINARY_OP_FUNC(TYPE, NAME, FUNC)                            \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   for (int i = 0; i < 8; ++i)                                     \
       ret[i] = FUNC(a[i], b[i]);                             \
   return ret;                                                      \
}

#define CMP_OP(TYPE, SUFFIX, CAST, NAME, OP)                        \
static FORCEINLINE __vec8_i1 NAME##_##SUFFIX(TYPE a, TYPE b) {     \
   __vec8_i1 ret;                                                  \
   ret.v = 0;                                                       \
   for (int i = 0; i < 8; ++i)                                     \
       ret.v |= ((CAST)(a[i]) OP (CAST)(b[i])) << i;            \
   return ret;                                                      \
}                                                                   \
static FORCEINLINE __vec8_i1 NAME##_##SUFFIX##_and_mask(TYPE a, TYPE b,       \
                                              __vec8_i1 mask) {    \
   __vec8_i1 ret;                                                  \
   ret.v = 0;                                                       \
   for (int i = 0; i < 8; ++i)                                     \
       ret.v |= ((CAST)(a[i]) OP (CAST)(b[i])) << i;            \
   ret.v &= mask.v;                                                 \
   return ret;                                                      \
}

#define INSERT_EXTRACT(VTYPE, STYPE)                                  \
static FORCEINLINE STYPE __extract_element(VTYPE v, int index) {      \
    return ((STYPE *)&v)[index];                                      \
}                                                                     \
static FORCEINLINE void __insert_element(VTYPE *v, int index, STYPE val) { \
    ((STYPE *)v)[index] = val;                                        \
}

#define LOAD_STORE(VTYPE, STYPE)                       \
template <int ALIGN>                                   \
static FORCEINLINE VTYPE __load(const VTYPE *p) {      \
    STYPE *ptr = (STYPE *)p;                           \
    VTYPE ret;                                         \
    for (int i = 0; i < 8; ++i)                       \
        ret[i] = ptr[i];                             \
    return ret;                                        \
}                                                      \
template <int ALIGN>                                   \
static FORCEINLINE void __store(VTYPE *p, VTYPE v) {   \
    STYPE *ptr = (STYPE *)p;                           \
    for (int i = 0; i < 8; ++i)                       \
        ptr[i] = v[i];                               \
}

#define LOADS(VTYPE, STYPE)                       \
template <int ALIGN>                                   \
static FORCEINLINE VTYPE __load(const VTYPE *p) {      \
    STYPE *ptr = (STYPE *)p;                           \
    VTYPE ret;                                         \
    for (int i = 0; i < 8; ++i)                       \
        ret[i] = ptr[i];                             \
    return ret;                                        \
}                                                      \

#define STORES(VTYPE, STYPE)                       \
template <int ALIGN>                                   \
static FORCEINLINE void __store(VTYPE *p, VTYPE v) {   \
    STYPE *ptr = (STYPE *)p;                           \
    for (int i = 0; i < 8; ++i)                       \
        ptr[i] = v[i];                               \
}

#define REDUCE_ADD(TYPE, VTYPE, NAME)           \
static FORCEINLINE TYPE NAME(VTYPE v) {         \
     TYPE ret = v[0];                         \
     for (int i = 1; i < 8; ++i)               \
         ret = ret + v[i];                    \
     return ret;                                \
}

#define REDUCE_MINMAX(TYPE, VTYPE, NAME, OP)                    \
static FORCEINLINE TYPE NAME(VTYPE v) {                         \
    TYPE ret = v[0];                                          \
    for (int i = 1; i < 8; ++i)                                \
        ret = (ret OP (TYPE)v[i]) ? ret : (TYPE)v[i];       \
    return ret;                                                 \
}

#define SELECT(TYPE)                                                \
static FORCEINLINE TYPE __select(__vec8_i1 mask, TYPE a, TYPE b) { \
    TYPE ret;                                                       \
    for (int i = 0; i < 8; ++i)                                    \
        ret[i] = (mask.v & (1<<i)) ? a[i] : b[i];             \
    return ret;                                                     \
}                                                                   \
static FORCEINLINE TYPE __select(bool cond, TYPE a, TYPE b) {       \
    return cond ? a : b;                                            \
}

#define SHIFT_UNIFORM(TYPE, CAST, NAME, OP)                         \
static FORCEINLINE TYPE NAME(TYPE a, int32_t b) {                   \
   TYPE ret;                                                        \
   for (int i = 0; i < 8; ++i)                                     \
       ret[i] = (CAST)(a[i]) OP b;                              \
   return ret;                                                      \
}

#define SMEAR(VTYPE, NAME, STYPE)                                  \
template <class RetVecType> VTYPE __smear_##NAME(STYPE);           \
template <> FORCEINLINE VTYPE __smear_##NAME<VTYPE>(STYPE v) {     \
    VTYPE ret;                                                     \
    for (int i = 0; i < 8; ++i)                                   \
        ret[i] = v;                                              \
    return ret;                                                    \
}

#define SETZERO(VTYPE, NAME)                                       \
template <class RetVecType> VTYPE __setzero_##NAME();              \
template <> FORCEINLINE VTYPE __setzero_##NAME<VTYPE>() {          \
    VTYPE ret;                                                     \
    for (int i = 0; i < 8; ++i)                                   \
        ret[i] = 0;                                              \
    return ret;                                                    \
}

#define UNDEF(VTYPE, NAME)                                         \
template <class RetVecType> VTYPE __undef_##NAME();                \
template <> FORCEINLINE VTYPE __undef_##NAME<VTYPE>() {            \
    return VTYPE();                                                \
}

#define BROADCAST(VTYPE, NAME, STYPE)                 \
static FORCEINLINE VTYPE __broadcast_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 8; ++i)                      \
        ret[i] = v[index & 0x7];                  \
    return ret;                                       \
}                                                     \

#define ROTATE(VTYPE, NAME, STYPE)                    \
static FORCEINLINE VTYPE __rotate_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 8; ++i)                      \
        ret[i] = v[(i+index) & 0x7];              \
    return ret;                                       \
}                                                     \

#define SHUFFLES(VTYPE, NAME, STYPE)                 \
static FORCEINLINE VTYPE __shuffle_##NAME(VTYPE v, __vec8_i32 index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 8; ++i)                      \
        ret[i] = v[__extract_element(index, i) & 0x7];      \
    return ret;                                       \
}                                                     \
static FORCEINLINE VTYPE __shuffle2_##NAME(VTYPE v0, VTYPE v1, __vec8_i32 index) {     \
    VTYPE ret;                                        \
    for (int i = 0; i < 8; ++i) {                    \
        int ii = __extract_element(index, i) & 0xf;    \
        ret[i] = (ii < 8) ? v0[ii] : v1[ii-8];  \
    }                                                 \
    return ret;                                       \
}

#define SHUFFLE2(VTYPE, NAME, STYPE)                 \
static FORCEINLINE VTYPE __shuffle2_##NAME(VTYPE v0, VTYPE v1, __vec8_i32 index) {     \
    VTYPE ret;                                        \
    for (int i = 0; i < 8; ++i) {                    \
        int ii = __extract_element(index, i) & 0xf;    \
        ret[i] = (ii < 8) ? v0[ii] : v1[ii-8];  \
    }                                                 \
    return ret;                                       \
}

///////////////////////////////////////////////////////////////////////////

INSERT_EXTRACT(__vec1_i8, int8_t)
INSERT_EXTRACT(__vec1_i16, int16_t)
INSERT_EXTRACT(__vec1_i32, int32_t)
INSERT_EXTRACT(__vec1_i64, int64_t)
INSERT_EXTRACT(__vec1_f, float)
INSERT_EXTRACT(__vec1_d, double)

///////////////////////////////////////////////////////////////////////////
// mask ops

static FORCEINLINE __vec8_i1 __movmsk(__vec8_i1 mask) {
    return mask.v;
}

static FORCEINLINE bool __any(__vec8_i1 mask) {
    return (mask.v!=0);
}

static FORCEINLINE bool __all(__vec8_i1 mask) {
    return (mask.v==0xFF);
}

static FORCEINLINE bool __none(__vec8_i1 mask) {
    return (mask.v==0);
}

static FORCEINLINE __vec8_i1 __equal_i1(__vec8_i1 a, __vec8_i1 b) {
    return (a.v & b.v) | (~a.v & ~b.v);
}

static FORCEINLINE __vec8_i1 __and(__vec8_i1 a, __vec8_i1 b) {
    return  a.v & b.v;
}

static FORCEINLINE __vec8_i1 __xor(__vec8_i1 a, __vec8_i1 b) {
    return a.v ^ b.v;
}

static FORCEINLINE __vec8_i1 __or(__vec8_i1 a, __vec8_i1 b) {
    return  a.v | b.v;
}

static FORCEINLINE __vec8_i1 __not(__vec8_i1 v) {
    return ~v;
}

static FORCEINLINE __vec8_i1 __and_not1(__vec8_i1 a, __vec8_i1 b) {
    return  ~a.v & b.v;
}

static FORCEINLINE __vec8_i1 __and_not2(__vec8_i1 a, __vec8_i1 b) {
    return  a.v & ~b.v;
}

static FORCEINLINE __vec8_i1 __select(__vec8_i1 mask, __vec8_i1 a, 
                                       __vec8_i1 b) {
    return  (a.v & mask.v) | (b.v & ~mask.v);
}

static FORCEINLINE __vec8_i1 __select(bool cond, __vec8_i1 a, __vec8_i1 b) {
    return cond ? a : b;
}

static FORCEINLINE bool __extract_element(__vec8_i1 vec, int index) {
    return (vec.v & (1 << index)) ? true : false;
}

static FORCEINLINE void __insert_element(__vec8_i1 *vec, int index, 
                                         bool val) {
    if (val == false)
        vec->v &= ~(1 << index);
    else
        vec->v |= (1 << index);
}

template <int ALIGN> static FORCEINLINE __vec8_i1 __load(const __vec8_i1 *p) {
    uint8_t *ptr = (uint8_t *)p;
    __vec8_i1 r;
    r.v = *ptr;
    return r;
}

template <int ALIGN> static FORCEINLINE void __store(__vec8_i1 *p, __vec8_i1 v) {
    uint8_t *ptr = (uint8_t *)p;
    *ptr = v.v;
}

template <class RetVecType> RetVecType __smear_i1(int i);
template <> static FORCEINLINE __vec8_i1 __smear_i1<__vec8_i1>(int i) {
    return i?0xFF:0x0;
}

template <class RetVecType> RetVecType __setzero_i1();
template <> static FORCEINLINE __vec8_i1 __setzero_i1<__vec8_i1>() {
    return 0;
}

template <class RetVecType> __vec8_i1 __undef_i1();
template <> FORCEINLINE __vec8_i1 __undef_i1<__vec8_i1>() {
    return __vec8_i1();
}


///////////////////////////////////////////////////////////////////////////
// int8

BINARY_OP(__vec8_i8, __add, +)
BINARY_OP(__vec8_i8, __sub, -)
BINARY_OP(__vec8_i8, __mul, *)

BINARY_OP(__vec8_i8, __or, |)
BINARY_OP(__vec8_i8, __and, &)
BINARY_OP(__vec8_i8, __xor, ^)
BINARY_OP(__vec8_i8, __shl, <<)

BINARY_OP_CAST(__vec8_i8, uint8_t, __udiv, /)
BINARY_OP_CAST(__vec8_i8, int8_t,  __sdiv, /)

BINARY_OP_CAST(__vec8_i8, uint8_t, __urem, %)
BINARY_OP_CAST(__vec8_i8, int8_t,  __srem, %)
BINARY_OP_CAST(__vec8_i8, uint8_t, __lshr, >>)
BINARY_OP_CAST(__vec8_i8, int8_t,  __ashr, >>)

SHIFT_UNIFORM(__vec8_i8, uint8_t, __lshr, >>)
SHIFT_UNIFORM(__vec8_i8, int8_t, __ashr, >>)
SHIFT_UNIFORM(__vec8_i8, int8_t, __shl, <<)

CMP_OP(__vec8_i8, i8, int8_t,  __equal, ==)
CMP_OP(__vec8_i8, i8, int8_t,  __not_equal, !=)
CMP_OP(__vec8_i8, i8, uint8_t, __unsigned_less_equal, <=)
CMP_OP(__vec8_i8, i8, int8_t,  __signed_less_equal, <=)
CMP_OP(__vec8_i8, i8, uint8_t, __unsigned_greater_equal, >=)
CMP_OP(__vec8_i8, i8, int8_t,  __signed_greater_equal, >=)
CMP_OP(__vec8_i8, i8, uint8_t, __unsigned_less_than, <)
CMP_OP(__vec8_i8, i8, int8_t,  __signed_less_than, <)
CMP_OP(__vec8_i8, i8, uint8_t, __unsigned_greater_than, >)
CMP_OP(__vec8_i8, i8, int8_t,  __signed_greater_than, >)

SELECT(__vec8_i8)
INSERT_EXTRACT(__vec8_i8, int8_t)
SMEAR(__vec8_i8, i8, int8_t)
SETZERO(__vec8_i8, i8)
UNDEF(__vec8_i8, i8)
BROADCAST(__vec8_i8, i8, int8_t)
ROTATE(__vec8_i8, i8, int8_t)
SHUFFLES(__vec8_i8, i8, int8_t)
LOAD_STORE(__vec8_i8, int8_t)

///////////////////////////////////////////////////////////////////////////
// int16

BINARY_OP(__vec8_i16, __add, +)
BINARY_OP(__vec8_i16, __sub, -)
BINARY_OP(__vec8_i16, __mul, *)

BINARY_OP(__vec8_i16, __or, |)
BINARY_OP(__vec8_i16, __and, &)
BINARY_OP(__vec8_i16, __xor, ^)
BINARY_OP(__vec8_i16, __shl, <<)

BINARY_OP_CAST(__vec8_i16, uint16_t, __udiv, /)
BINARY_OP_CAST(__vec8_i16, int16_t,  __sdiv, /)

BINARY_OP_CAST(__vec8_i16, uint16_t, __urem, %)
BINARY_OP_CAST(__vec8_i16, int16_t,  __srem, %)
BINARY_OP_CAST(__vec8_i16, uint16_t, __lshr, >>)
BINARY_OP_CAST(__vec8_i16, int16_t,  __ashr, >>)

SHIFT_UNIFORM(__vec8_i16, uint16_t, __lshr, >>)
SHIFT_UNIFORM(__vec8_i16, int16_t, __ashr, >>)
SHIFT_UNIFORM(__vec8_i16, int16_t, __shl, <<)

CMP_OP(__vec8_i16, i16, int16_t,  __equal, ==)
CMP_OP(__vec8_i16, i16, int16_t,  __not_equal, !=)
CMP_OP(__vec8_i16, i16, uint16_t, __unsigned_less_equal, <=)
CMP_OP(__vec8_i16, i16, int16_t,  __signed_less_equal, <=)
CMP_OP(__vec8_i16, i16, uint16_t, __unsigned_greater_equal, >=)
CMP_OP(__vec8_i16, i16, int16_t,  __signed_greater_equal, >=)
CMP_OP(__vec8_i16, i16, uint16_t, __unsigned_less_than, <)
CMP_OP(__vec8_i16, i16, int16_t,  __signed_less_than, <)
CMP_OP(__vec8_i16, i16, uint16_t, __unsigned_greater_than, >)
CMP_OP(__vec8_i16, i16, int16_t,  __signed_greater_than, >)

SELECT(__vec8_i16)
INSERT_EXTRACT(__vec8_i16, int16_t)
SMEAR(__vec8_i16, i16, int16_t)
SETZERO(__vec8_i16, i16)
UNDEF(__vec8_i16, i16)
BROADCAST(__vec8_i16, i16, int16_t)
ROTATE(__vec8_i16, i16, int16_t)
SHUFFLES(__vec8_i16, i16, int16_t)
LOAD_STORE(__vec8_i16, int16_t)

#if 0 /* evghenii::int32 */
///////////////////////////////////////////////////////////////////////////
// int32

BINARY_OP(__vec8_i32, __add, +)
BINARY_OP(__vec8_i32, __sub, -)
BINARY_OP(__vec8_i32, __mul, *)

BINARY_OP(__vec8_i32, __or, |)
BINARY_OP(__vec8_i32, __and, &)
BINARY_OP(__vec8_i32, __xor, ^)
BINARY_OP(__vec8_i32, __shl, <<)

BINARY_OP_CAST(__vec8_i32, uint32_t, __udiv, /)
BINARY_OP_CAST(__vec8_i32, int32_t,  __sdiv, /)

BINARY_OP_CAST(__vec8_i32, uint32_t, __urem, %)
BINARY_OP_CAST(__vec8_i32, int32_t,  __srem, %)
BINARY_OP_CAST(__vec8_i32, uint32_t, __lshr, >>)
BINARY_OP_CAST(__vec8_i32, int32_t,  __ashr, >>)

SHIFT_UNIFORM(__vec8_i32, uint32_t, __lshr, >>)
SHIFT_UNIFORM(__vec8_i32, int32_t, __ashr, >>)
SHIFT_UNIFORM(__vec8_i32, int32_t, __shl, <<)

CMP_OP(__vec8_i32, i32, int32_t,  __equal, ==)
CMP_OP(__vec8_i32, i32, int32_t,  __not_equal, !=)
CMP_OP(__vec8_i32, i32, uint32_t, __unsigned_less_equal, <=)
CMP_OP(__vec8_i32, i32, int32_t,  __signed_less_equal, <=)
CMP_OP(__vec8_i32, i32, uint32_t, __unsigned_greater_equal, >=)
CMP_OP(__vec8_i32, i32, int32_t,  __signed_greater_equal, >=)
CMP_OP(__vec8_i32, i32, uint32_t, __unsigned_less_than, <)
CMP_OP(__vec8_i32, i32, int32_t,  __signed_less_than, <)
CMP_OP(__vec8_i32, i32, uint32_t, __unsigned_greater_than, >)
CMP_OP(__vec8_i32, i32, int32_t,  __signed_greater_than, >)

SELECT(__vec8_i32)
INSERT_EXTRACT(__vec8_i32, int32_t)
SMEAR(__vec8_i32, i32, int32_t)
SETZERO(__vec8_i32, i32)
UNDEF(__vec8_i32, i32)
BROADCAST(__vec8_i32, i32, int32_t)
ROTATE(__vec8_i32, i32, int32_t)
SHUFFLES(__vec8_i32, i32, int32_t)
LOAD_STORE(__vec8_i32, int32_t)

#else /* evghenii::int32 */
///////////////////////////////////////////////////////////////////////////
// int32
///////////////////////////////////////////////////////////////////////////

#define IZERO _mm512_setzero_epi32()
static FORCEINLINE __vec8_i32 __add(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_add_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __sub(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_sub_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __mul(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_mullo_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __udiv(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_div_epu32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __sdiv(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_div_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __urem(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_rem_epu32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __srem(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_rem_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __or(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_or_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __and(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_and_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __xor(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_xor_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __shl(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_sllv_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __lshr(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_srlv_epi32(IZERO,0xFF, a, b);
}

static FORCEINLINE __vec8_i32 __ashr(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_srav_epi32(IZERO,0xFF, a, b); 
}

static FORCEINLINE __vec8_i32 __shl(__vec8_i32 a, int32_t n) {
    return _mm512_mask_slli_epi32(IZERO,0xFF, a, n);
}

static FORCEINLINE __vec8_i32 __lshr(__vec8_i32 a, int32_t n) {
    return _mm512_mask_srli_epi32(IZERO,0xFF, a, n); 
}

static FORCEINLINE __vec8_i32 __ashr(__vec8_i32 a, int32_t n) {
    return _mm512_mask_srai_epi32(IZERO,0xFF, a, n); 
}

static FORCEINLINE __vec8_i1 __equal_i32(const __vec8_i32 &a, const __vec8_i32 &b) {
    return _mm512_mask_cmpeq_epi32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __equal_i32_and_mask(const __vec8_i32 &a, const __vec8_i32 &b,
                                                   __vec8_i1 m) {
    return _mm512_mask_cmpeq_epi32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __not_equal_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmpneq_epi32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __not_equal_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                       __vec8_i1 m) {
    return _mm512_mask_cmpneq_epi32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __unsigned_less_equal_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmple_epu32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __unsigned_less_equal_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                                 __vec8_i1 m) {
    return _mm512_mask_cmple_epu32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __signed_less_equal_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmple_epi32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __signed_less_equal_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                               __vec8_i1 m) {
    return _mm512_mask_cmple_epi32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __unsigned_greater_equal_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmpge_epu32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __unsigned_greater_equal_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                                    __vec8_i1 m) {
    return _mm512_mask_cmpge_epu32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __signed_greater_equal_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmpge_epi32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __signed_greater_equal_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                                  __vec8_i1 m) {
    return _mm512_mask_cmpge_epi32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __unsigned_less_than_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmplt_epu32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __unsigned_less_than_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                                __vec8_i1 m) {
    return _mm512_mask_cmplt_epu32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __signed_less_than_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmplt_epi32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __signed_less_than_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                              __vec8_i1 m) {
    return _mm512_mask_cmplt_epi32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __unsigned_greater_than_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmpgt_epu32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __unsigned_greater_than_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                                   __vec8_i1 m) {
    return _mm512_mask_cmpgt_epu32_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __signed_greater_than_i32(__vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_cmpgt_epi32_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_i1 __signed_greater_than_i32_and_mask(__vec8_i32 a, __vec8_i32 b,
                                                                 __vec8_i1 m) {
    return _mm512_mask_cmpgt_epi32_mask(m, a, b);
}

static FORCEINLINE __vec8_i32 __select(__vec8_i1 mask,
                                        __vec8_i32 a, __vec8_i32 b) {
    return _mm512_mask_mov_epi32(b, mask, a);
} 

static FORCEINLINE __vec8_i32 __select(bool cond, __vec8_i32 a, __vec8_i32 b) {
    return cond ? a : b;
}

static FORCEINLINE int32_t __extract_element(__vec8_i32 v, int index) { //uint32_t index) {
    return ((int32_t *)&v)[index];
}

static FORCEINLINE void __insert_element(__vec8_i32 *v, uint32_t index, int32_t val) {
    ((int32_t *)v)[index] = val;
}

template <class RetVecType> RetVecType __smear_i32(int32_t i);
template <> static FORCEINLINE __vec8_i32 __smear_i32<__vec8_i32>(int32_t i) {
    return _mm512_set_16to16_epi32(0,0,0,0,0,0,0,0, i,i,i,i,i,i,i,i);
}

static const __vec8_i32 __ispc_one = __smear_i32<__vec8_i32>(1);
static const __vec8_i32 __ispc_thirty_two = __smear_i32<__vec8_i32>(32);
static const __vec8_i32 __ispc_ffffffff = __smear_i32<__vec8_i32>(-1);
static const __vec8_i32 __ispc_stride1(0, 1, 2, 3, 4, 5, 6, 7);

template <class RetVecType> RetVecType __setzero_i32();
template <> static FORCEINLINE __vec8_i32 __setzero_i32<__vec8_i32>() {
    return _mm512_setzero_epi32();
}

template <class RetVecType> RetVecType __undef_i32();
template <> static FORCEINLINE __vec8_i32 __undef_i32<__vec8_i32>() {
    return __vec8_i32();
}

static FORCEINLINE __vec8_i32 __broadcast_i32(__vec8_i32 v, int index) {
    int32_t val = __extract_element(v, index & 0xf);
    return _mm512_set1_epi32(val);
}

#if 0 /* evghenii::doesn't work */
static FORCEINLINE __vec8_i32 __rotate_i32(__vec8_i32 v, int index) {
    __vec8_i32 idx = __smear_i32<__vec8_i32>(index);
    __vec8_i32 shuffle = _mm512_and_epi32(_mm512_add_epi32(__ispc_stride1, idx),  __smear_i32<__vec8_i32>(0x7));
    return _mm512_mask_permutevar_epi32(v, 0xffff, shuffle, v);
}
#else
ROTATE(__vec8_i32, i32, int32_t)
#endif

static FORCEINLINE __vec8_i32 __shuffle_i32(__vec8_i32 v, __vec8_i32 index) {
    return _mm512_mask_permutevar_epi32(v, 0xffff, index, v);
}
SHUFFLE2(__vec8_i32, i32, int32_t) /* evghenii::to implement */

template <int ALIGN> static FORCEINLINE __vec8_i32 __load(const __vec8_i32 *p) {
  __vec8_i32 v;
  v = _mm512_extloadunpacklo_epi32(v, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v = _mm512_extloadunpackhi_epi32(v, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  return __select(0xFF,v,IZERO);
}


template <int ALIGN> static FORCEINLINE void __store(__vec8_i32 *p, __vec8_i32 v) {
  _mm512_mask_extpackstorelo_epi32(          p,    0xFF, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_mask_extpackstorehi_epi32((uint8_t*)p+64, 0xFF, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
}

#if 0
template <> static FORCEINLINE __vec8_i32 __load<64>(const __vec8_i32 *p) {
    return _mm512_load_epi32(p);
}
template <> static FORCEINLINE void __store<64>(__vec8_i32 *p, __vec8_i32 v) {
    _mm512_store_epi32(p, v);
}
#endif
#endif /* evghenii::int32 */

///////////////////////////////////////////////////////////////////////////
// int64

BINARY_OP(__vec8_i64, __add, +)
BINARY_OP(__vec8_i64, __sub, -)
BINARY_OP(__vec8_i64, __mul, *)

BINARY_OP(__vec8_i64, __or, |)
BINARY_OP(__vec8_i64, __and, &)
BINARY_OP(__vec8_i64, __xor, ^)
BINARY_OP(__vec8_i64, __shl, <<)

BINARY_OP_CAST(__vec8_i64, uint64_t, __udiv, /)
BINARY_OP_CAST(__vec8_i64, int64_t,  __sdiv, /)

BINARY_OP_CAST(__vec8_i64, uint64_t, __urem, %)
BINARY_OP_CAST(__vec8_i64, int64_t,  __srem, %)
BINARY_OP_CAST(__vec8_i64, uint64_t, __lshr, >>)
BINARY_OP_CAST(__vec8_i64, int64_t,  __ashr, >>)

SHIFT_UNIFORM(__vec8_i64, uint64_t, __lshr, >>)
SHIFT_UNIFORM(__vec8_i64, int64_t, __ashr, >>)
SHIFT_UNIFORM(__vec8_i64, int64_t, __shl, <<)

CMP_OP(__vec8_i64, i64, int64_t,  __equal, ==)
CMP_OP(__vec8_i64, i64, int64_t,  __not_equal, !=)
CMP_OP(__vec8_i64, i64, uint64_t, __unsigned_less_equal, <=)
CMP_OP(__vec8_i64, i64, int64_t,  __signed_less_equal, <=)
CMP_OP(__vec8_i64, i64, uint64_t, __unsigned_greater_equal, >=)
CMP_OP(__vec8_i64, i64, int64_t,  __signed_greater_equal, >=)
CMP_OP(__vec8_i64, i64, uint64_t, __unsigned_less_than, <)
CMP_OP(__vec8_i64, i64, int64_t,  __signed_less_than, <)
CMP_OP(__vec8_i64, i64, uint64_t, __unsigned_greater_than, >)
CMP_OP(__vec8_i64, i64, int64_t,  __signed_greater_than, >)

SELECT(__vec8_i64)
INSERT_EXTRACT(__vec8_i64, int64_t)
SMEAR(__vec8_i64, i64, int64_t)
SETZERO(__vec8_i64, i64)
UNDEF(__vec8_i64, i64)
BROADCAST(__vec8_i64, i64, int64_t)
ROTATE(__vec8_i64, i64, int64_t)
SHUFFLES(__vec8_i64, i64, int64_t)
LOAD_STORE(__vec8_i64, int64_t)


#if 0 /* evghenii::float */
///////////////////////////////////////////////////////////////////////////
// float

BINARY_OP(__vec8_f, __add, +)
BINARY_OP(__vec8_f, __sub, -)
BINARY_OP(__vec8_f, __mul, *)
BINARY_OP(__vec8_f, __div, /)

CMP_OP(__vec8_f, float, float, __equal, ==)
CMP_OP(__vec8_f, float, float, __not_equal, !=)
CMP_OP(__vec8_f, float, float, __less_than, <)
CMP_OP(__vec8_f, float, float, __less_equal, <=)
CMP_OP(__vec8_f, float, float, __greater_than, >)
CMP_OP(__vec8_f, float, float, __greater_equal, >=)

static FORCEINLINE __vec8_i1 __ordered_float(__vec8_f a, __vec8_f b) {
    __vec8_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 8; ++i)
        ret.v |= ((a[i] == a[i]) && (b[i] == b[i])) ? (1 << i) : 0;
    return ret;
}

static FORCEINLINE __vec8_i1 __unordered_float(__vec8_f a, __vec8_f b) {
    __vec8_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 8; ++i)
        ret.v |= ((a[i] != a[i]) || (b[i] != b[i])) ? (1 << i) : 0;
    return ret;
}

#if 0
      case Instruction::FRem: intrinsic = "__frem"; break;
#endif

SELECT(__vec8_f)
INSERT_EXTRACT(__vec8_f, float)
SMEAR(__vec8_f, float, float)
SETZERO(__vec8_f, float)
UNDEF(__vec8_f, float)
BROADCAST(__vec8_f, float, float)
ROTATE(__vec8_f, float, float)
SHUFFLES(__vec8_f, float, float)
LOAD_STORE(__vec8_f, float)
#else /* evghenii::float */

///////////////////////////////////////////////////////////////////////////
// float
///////////////////////////////////////////////////////////////////////////

#define FZERO _mm512_setzero_ps()
static FORCEINLINE __vec8_f __add(__vec8_f a, __vec8_f b) { 
    return _mm512_mask_add_ps(FZERO, 0xFF, a, b);
}

static FORCEINLINE __vec8_f __sub(__vec8_f a, __vec8_f b) {
    return _mm512_mask_sub_ps(FZERO, 0xFF, a, b);
}

static FORCEINLINE __vec8_f __mul(__vec8_f a, __vec8_f b) {
    return _mm512_mask_mul_ps(FZERO, 0xFF, a, b);
}

static FORCEINLINE __vec8_f __div(__vec8_f a, __vec8_f b) {
    return _mm512_mask_div_ps(FZERO, 0xFF, a, b);
}

static FORCEINLINE __vec8_i1 __equal_float(__vec8_f a, __vec8_f b) {
    return _mm512_mask_cmpeq_ps_mask(0xFF, a, b);
}

static FORCEINLINE __vec8_i1 __equal_float_and_mask(__vec8_f a, __vec8_f b,
                                                     __vec8_i1 m) {
    return _mm512_mask_cmpeq_ps_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __not_equal_float(__vec8_f a, __vec8_f b) {
    return _mm512_mask_cmpneq_ps_mask(0xFF, a, b);
}

static FORCEINLINE __vec8_i1 __not_equal_float_and_mask(__vec8_f a, __vec8_f b,
                                                         __vec8_i1 m) {
    return _mm512_mask_cmpneq_ps_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __less_than_float(__vec8_f a, __vec8_f b) {
    return _mm512_mask_cmplt_ps_mask(0xFF, a, b);
}

static FORCEINLINE __vec8_i1 __less_than_float_and_mask(__vec8_f a, __vec8_f b,
                                                         __vec8_i1 m) {
    return _mm512_mask_cmplt_ps_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __less_equal_float(__vec8_f a, __vec8_f b) {
    return _mm512_mask_cmple_ps_mask(0xFF, a, b);
}

static FORCEINLINE __vec8_i1 __less_equal_float_and_mask(__vec8_f a, __vec8_f b,
                                                          __vec8_i1 m) {
    return _mm512_mask_cmple_ps_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __greater_than_float(__vec8_f a, __vec8_f b) {
    return _mm512_mask_cmp_ps_mask(0xFF, a, b,_CMP_GT_OS);
}

static FORCEINLINE __vec8_i1 __greater_than_float_and_mask(__vec8_f a, __vec8_f b,
                                                            __vec8_i1 m) {
    return _mm512_mask_cmp_ps_mask(m,a, b,_CMP_GT_OS);
}

static FORCEINLINE __vec8_i1 __greater_equal_float(__vec8_f a, __vec8_f b) {
    return _mm512_mask_cmp_ps_mask(0xFF, a, b,_CMP_GE_OS);
}

static FORCEINLINE __vec8_i1 __greater_equal_float_and_mask(__vec8_f a, __vec8_f b,
                                                             __vec8_i1 m) {
    return _mm512_mask_cmp_ps_mask(m,a, b,_CMP_GE_OS);
}

static FORCEINLINE __vec8_i1 __ordered_float(__vec8_f a, __vec8_f b) {
    return _mm512_mask_cmpord_ps_mask(0xFF, a, b);
}

static FORCEINLINE __vec8_i1 __unordered_float(__vec8_f a, __vec8_f b) {
    return _mm512_mask_cmpunord_ps_mask(0xFF,a, b);
}

static FORCEINLINE __vec8_f __select(__vec8_i1 mask, __vec8_f a, __vec8_f b) {
    return _mm512_mask_mov_ps(b, mask & 0xFF, a);
}

static FORCEINLINE __vec8_f __select(bool cond, __vec8_f a, __vec8_f b) {
    return cond ? a : b;
}

static FORCEINLINE float __extract_element(__vec8_f v, uint32_t index) {
  return v[index];
 //   return ((float *)&v)[index];
}

static FORCEINLINE void  __insert_element(__vec8_f *v, uint32_t index, float val) {
  (*v)[index] = val;
//    ((float *)v)[index] = val;
}

template <class RetVecType> RetVecType __smear_float(float f);
template <> static FORCEINLINE __vec8_f __smear_float<__vec8_f>(float f) {
  return _mm512_set_16to16_ps(0,0,0,0,0,0,0,0, f,f,f,f,f,f,f,f);
}

template <class RetVecType> RetVecType __setzero_float();
template <> static FORCEINLINE __vec8_f __setzero_float<__vec8_f>() {
    return _mm512_setzero_ps();
}

template <class RetVecType> RetVecType __undef_float();
template <> static FORCEINLINE __vec8_f __undef_float<__vec8_f>() {
    return __vec8_f();
}

static FORCEINLINE __vec8_f __broadcast_float(__vec8_f v, int index) {
    float val = __extract_element(v, index & 0x7);
  return _mm512_set_16to16_ps(0,0,0,0,0,0,0,0, val,val,val,val,val,val,val,val);
}
 
#if 1
static FORCEINLINE __vec8_f __shuffle_float(__vec8_f v, __vec8_i32 index) {
    return _mm512_castsi512_ps(_mm512_mask_permutevar_epi32(_mm512_castps_si512(v), 0xffff, index, _mm512_castps_si512(v)));
}
#endif
ROTATE(__vec8_f, float, float)
SHUFFLE2(__vec8_f, float, float)

#if 0
LOADS(__vec8_f, float)
#else
template <int ALIGN> static FORCEINLINE __vec8_f __load(const __vec8_f *p) {
  __vec8_f v;
  v = _mm512_extloadunpacklo_ps(v,           p,    _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  v = _mm512_extloadunpackhi_ps(v, (uint8_t*)p+64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  return __select(0xFF,v,FZERO);
}
#endif

#if 0
STORES(__vec8_f, float)
#else
template <int ALIGN> static FORCEINLINE void __store(__vec8_f *p, __vec8_f v) 
{
  _mm512_mask_extpackstorelo_ps(          p,    0xFF, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
  _mm512_mask_extpackstorehi_ps((uint8_t*)p+64, 0xFF, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
}
#endif

#endif /* evghenii::float */

static FORCEINLINE float __exp_uniform_float(float v) {    return expf(v);}
static FORCEINLINE __vec8_f __exp_varying_float(__vec8_f v) { return _mm512_mask_exp_ps(FZERO, 0xFF, v); }


static FORCEINLINE float __log_uniform_float(float v) {    return logf(v);}
static FORCEINLINE __vec8_f __log_varying_float(__vec8_f v) { return _mm512_mask_log_ps(FZERO, 0xFF, v); }

static FORCEINLINE float __pow_uniform_float(float a, float b) {    return powf(a, b);}
static FORCEINLINE __vec8_f __pow_varying_float(__vec8_f a, __vec8_f b) { return _mm512_mask_pow_ps(FZERO, 0xFF, a,b); }


static FORCEINLINE int __intbits(float v) {
    union {
        float f;
        int i;
    } u;
    u.f = v;
    return u.i;
}

static FORCEINLINE float __floatbits(int v) {
    union {
        float f;
        int i;
    } u;
    u.i = v;
    return u.f;
}

static FORCEINLINE float __half_to_float_uniform(int16_t h) {
    static const uint32_t shifted_exp = 0x7c00 << 13; // exponent mask after shift

    int32_t o = ((int32_t)(h & 0x7fff)) << 13;     // exponent/mantissa bits
    uint32_t exp = shifted_exp & o;   // just the exponent
    o += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) { // Zero/Denormal?
        o += 1 << 23;             // extra exp adjust
        o = __intbits(__floatbits(o) - __floatbits(113 << 23)); // renormalize
    }

    o |= ((int32_t)(h & 0x8000)) << 16;    // sign bit
    return __floatbits(o);
}


static FORCEINLINE __vec8_f __half_to_float_varying(__vec8_i16 v) {
    __vec8_f ret;
    for (int i = 0; i < 8; ++i)
        ret[i] = __half_to_float_uniform(v[i]);
    return ret;
}


static FORCEINLINE int16_t __float_to_half_uniform(float f) {
    uint32_t sign_mask = 0x80000000u;
    int32_t o;

    int32_t fint = __intbits(f);
    int32_t sign = fint & sign_mask;
    fint ^= sign;

    int32_t f32infty = 255 << 23;
    o = (fint > f32infty) ? 0x7e00 : 0x7c00; 

    // (De)normalized number or zero
    // update fint unconditionally to save the blending; we don't need it
    // anymore for the Inf/NaN case anyway.
    const uint32_t round_mask = ~0xfffu; 
    const int32_t magic = 15 << 23;
    const int32_t f16infty = 31 << 23;

    int32_t fint2 = __intbits(__floatbits(fint & round_mask) * __floatbits(magic)) - round_mask;
    fint2 = (fint2 > f16infty) ? f16infty : fint2; // Clamp to signed infinity if overflowed

    if (fint < f32infty)
        o = fint2 >> 13; // Take the bits!

    return (o | (sign >> 16));
}


static FORCEINLINE __vec8_i16 __float_to_half_varying(__vec8_f v) {
    __vec8_i16 ret;
    for (int i = 0; i < 8; ++i)
        ret[i] = __float_to_half_uniform(v[i]);
    return ret;
}


#if 0 /* evghenii::double */
///////////////////////////////////////////////////////////////////////////
// double

BINARY_OP(__vec8_d, __add, +)
BINARY_OP(__vec8_d, __sub, -)
BINARY_OP(__vec8_d, __mul, *)
BINARY_OP(__vec8_d, __div, /)

CMP_OP(__vec8_d, double, double, __equal, ==)
CMP_OP(__vec8_d, double, double, __not_equal, !=)
CMP_OP(__vec8_d, double, double, __less_than, <)
CMP_OP(__vec8_d, double, double, __less_equal, <=)
CMP_OP(__vec8_d, double, double, __greater_than, >)
CMP_OP(__vec8_d, double, double, __greater_equal, >=)

static FORCEINLINE __vec8_i1 __ordered_double(__vec8_d a, __vec8_d b) {
    __vec8_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 8; ++i)
        ret.v |= ((a[i] == a[i]) && (b[i] == b[i])) ? (1 << i) : 0;
    return ret;
}

static FORCEINLINE __vec8_i1 __unordered_double(__vec8_d a, __vec8_d b) {
    __vec8_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 8; ++i)
        ret.v |= ((a[i] != a[i]) || (b[i] != b[i])) ? (1 << i) : 0;
    return ret;
}

#if 0
      case Instruction::FRem: intrinsic = "__frem"; break;
#endif

SELECT(__vec8_d)
INSERT_EXTRACT(__vec8_d, double)
SMEAR(__vec8_d, double, double)
SETZERO(__vec8_d, double)
UNDEF(__vec8_d, double)
BROADCAST(__vec8_d, double, double)
ROTATE(__vec8_d, double, double)
SHUFFLES(__vec8_d, double, double)
LOAD_STORE(__vec8_d, double)
#else /* evghenii::double */
///////////////////////////////////////////////////////////////////////////
// double
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec8_d __add(__vec8_d a, __vec8_d b) { 
    return _mm512_add_pd(a, b);
}
static FORCEINLINE __vec8_d __sub(__vec8_d a, __vec8_d b) {
    return _mm512_sub_pd(a, b);
}
static FORCEINLINE __vec8_d __mul(__vec8_d a, __vec8_d b) {
    return _mm512_mul_pd(a, b);
}

static FORCEINLINE __vec8_d __div(__vec8_d a, __vec8_d b) {
    return _mm512_div_pd(a, b);
}

static FORCEINLINE __vec8_i1 __equal_double(__vec8_d a, __vec8_d b) {
    return _mm512_cmpeq_pd_mask(a, b);
}

static FORCEINLINE __vec8_i1 __equal_double_and_mask(__vec8_d a, __vec8_d b,
                                                      __vec8_i1 m) {
    return _mm512_mask_cmpeq_pd_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __not_equal_double(__vec8_d a, __vec8_d b) {
    return _mm512_cmpneq_pd_mask(a, b);
}

static FORCEINLINE __vec8_i1 __not_equal_double_and_mask(__vec8_d a, __vec8_d b,
                                                          __vec8_i1 m) {
    return _mm512_mask_cmpneq_pd_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __less_than_double(__vec8_d a, __vec8_d b) {
    return _mm512_cmplt_pd_mask(a, b);
}

static FORCEINLINE __vec8_i1 __less_than_double_and_mask(__vec8_d a, __vec8_d b,
                                                          __vec8_i1 m) {
    return _mm512_mask_cmplt_pd_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __less_equal_double(__vec8_d a, __vec8_d b) {
    return _mm512_cmple_pd_mask(a, b);
}

static FORCEINLINE __vec8_i1 __less_equal_double_and_mask(__vec8_d a, __vec8_d b,
                                                           __vec8_i1 m) {
    return _mm512_mask_cmple_pd_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __greater_than_double(__vec8_d a, __vec8_d b) {
    return _mm512_cmpnle_pd_mask(a, b);
}

static FORCEINLINE __vec8_i1 __greater_than_double_and_mask(__vec8_d a, __vec8_d b,
                                                             __vec8_i1 m) {
    return _mm512_mask_cmpnle_pd_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __greater_equal_double(__vec8_d a, __vec8_d b) {
    return _mm512_cmpnlt_pd_mask(a, b);
}

static FORCEINLINE __vec8_i1 __greater_equal_double_and_mask(__vec8_d a, __vec8_d b,
                                                              __vec8_i1 m) {
    return _mm512_mask_cmpnlt_pd_mask(m, a, b);
}

static FORCEINLINE __vec8_i1 __ordered_double(__vec8_d a, __vec8_d b) {
    return _mm512_cmpord_pd_mask(a, b);
}

static FORCEINLINE __vec8_i1 __unordered_double(__vec8_d a, __vec8_d b) {
    return _mm512_cmpunord_pd_mask(a, b);
}

static FORCEINLINE __vec8_d __select(__vec8_i1 mask, __vec8_d a, __vec8_d b) {
    return _mm512_mask_mov_pd(b, mask, a);
}


static FORCEINLINE __vec8_d __select(bool cond, __vec8_d a, __vec8_d b) {
    return cond ? a : b;
}

static FORCEINLINE double __extract_element(__vec8_d v, uint32_t index) {
    return ((double *)&v)[index];
}

static FORCEINLINE void  __insert_element(__vec8_d *v, uint32_t index, double val) {
    ((double *)v)[index] = val;
}

template <class RetVecType> RetVecType __smear_double(double d);
template <> static FORCEINLINE __vec8_d __smear_double<__vec8_d>(double d) { return _mm512_set1_pd(d); }

template <class RetVecType> RetVecType __setzero_double();
template <> static FORCEINLINE __vec8_d __setzero_double<__vec8_d>() { return _mm512_setzero_pd(); }

template <class RetVecType> RetVecType __undef_double();
template <> static FORCEINLINE __vec8_d __undef_double<__vec8_d>() {    return __vec8_d();}

static FORCEINLINE __vec8_d __broadcast_double(__vec8_d v, int index) {
    double val = __extract_element(v, index & 0xf);
    return _mm512_set1_pd(val);
}

ROTATE(__vec8_d, double, double)
SHUFFLES(__vec8_d, double, double)

template <int ALIGN> static FORCEINLINE __vec8_d __load(const __vec8_d *p) {
  __vec8_d ret;
  ret.v = _mm512_extloadunpacklo_pd(ret.v, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  ret.v = _mm512_extloadunpackhi_pd(ret.v, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  return ret;
}
 
template <int ALIGN> static FORCEINLINE void __store(__vec8_d *p, __vec8_d v) {
  _mm512_extpackstorelo_pd(p, v.v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_pd((uint8_t*)p+64, v.v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
}


#if 0
template <> static FORCEINLINE __vec8_d __load<64>(const __vec8_d *p) {
    return  _mm512_load_pd(p);
}
template <> static FORCEINLINE __vec8_d __load<128>(const __vec8_d *p) {
    return __load<64>(p);
}
template <> static FORCEINLINE void __store<64>(__vec8_d *p, __vec8_d v) {
    _mm512_store_pd(p, v.v);
}
template <> static FORCEINLINE void __store<128>(__vec8_d *p, __vec8_d v) {
    __store<64>(p, v);
}
#endif
#endif /* evghenii::double */

///////////////////////////////////////////////////////////////////////////
// casts


#define CAST(TO, STO, FROM, SFROM, FUNC)        \
static FORCEINLINE TO FUNC(TO, FROM val) {      \
    TO ret;                                     \
    for (int i = 0; i < 8; ++i)                \
        ret[i] = (STO)((SFROM)(val[i]));    \
    return ret;                                 \
}

// sign extension conversions
CAST(__vec8_i64, int64_t, __vec8_i32, int32_t, __cast_sext)
CAST(__vec8_i64, int64_t, __vec8_i16, int16_t, __cast_sext)
CAST(__vec8_i64, int64_t, __vec8_i8,  int8_t,  __cast_sext)
CAST(__vec8_i32, int32_t, __vec8_i16, int16_t, __cast_sext)
CAST(__vec8_i32, int32_t, __vec8_i8,  int8_t,  __cast_sext)
CAST(__vec8_i16, int16_t, __vec8_i8,  int8_t,  __cast_sext)

#define CAST_SEXT_I1(TYPE)                            \
static FORCEINLINE TYPE __cast_sext(TYPE, __vec8_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 8; ++i) {                    \
        ret[i] = 0;                                 \
        if (v.v & (1 << i))                           \
            ret[i] = ~ret[i];                     \
    }                                                 \
    return ret;                                       \
}

CAST_SEXT_I1(__vec8_i8)
CAST_SEXT_I1(__vec8_i16)
#if 0
CAST_SEXT_I1(__vec8_i32)
#else
static FORCEINLINE __vec8_i32 __cast_sext(const __vec8_i32 &, const __vec8_i1 &val)
{
    __vec8_i32 ret = _mm512_setzero_epi32();
    __vec8_i32 one = _mm512_set1_epi32(-1);
    return _mm512_mask_mov_epi32(ret, 0xFF & val, one);
}
#endif
CAST_SEXT_I1(__vec8_i64)

// zero extension
CAST(__vec8_i64, uint64_t, __vec8_i32, uint32_t, __cast_zext)
CAST(__vec8_i64, uint64_t, __vec8_i16, uint16_t, __cast_zext)
CAST(__vec8_i64, uint64_t, __vec8_i8,  uint8_t,  __cast_zext)
CAST(__vec8_i32, uint32_t, __vec8_i16, uint16_t, __cast_zext)
CAST(__vec8_i32, uint32_t, __vec8_i8,  uint8_t,  __cast_zext)
CAST(__vec8_i16, uint16_t, __vec8_i8,  uint8_t,  __cast_zext)

#define CAST_ZEXT_I1(TYPE)                            \
static FORCEINLINE TYPE __cast_zext(TYPE, __vec8_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 8; ++i)                      \
        ret[i] = (v.v & (1 << i)) ? 1 : 0;          \
    return ret;                                       \
}

CAST_ZEXT_I1(__vec8_i8)
CAST_ZEXT_I1(__vec8_i16)
#if 0
CAST_ZEXT_I1(__vec8_i32)
#else
static FORCEINLINE __vec8_i32 __cast_zext(const __vec8_i32 &, const __vec8_i1 &val)
{
    __vec8_i32 ret = _mm512_setzero_epi32();
    __vec8_i32 one = _mm512_set1_epi32(1);
    return _mm512_mask_mov_epi32(ret, 0xFF & val, one);
}
#endif
CAST_ZEXT_I1(__vec8_i64)

// truncations
CAST(__vec8_i32, int32_t, __vec8_i64, int64_t, __cast_trunc)
CAST(__vec8_i16, int16_t, __vec8_i64, int64_t, __cast_trunc)
CAST(__vec8_i8,  int8_t,  __vec8_i64, int64_t, __cast_trunc)
CAST(__vec8_i16, int16_t, __vec8_i32, int32_t, __cast_trunc)
CAST(__vec8_i8,  int8_t,  __vec8_i32, int32_t, __cast_trunc)
CAST(__vec8_i8,  int8_t,  __vec8_i16, int16_t, __cast_trunc)

// signed int to float/double
#if 0
CAST(__vec8_f, float, __vec8_i8,   int8_t,  __cast_sitofp)
CAST(__vec8_f, float, __vec8_i16,  int16_t, __cast_sitofp)
CAST(__vec8_f, float, __vec8_i32,  int32_t, __cast_sitofp)
#else
static FORCEINLINE __vec8_f __cast_sitofp(__vec8_f, __vec8_i8  val) {return _mm512_mask_extload_ps(FZERO, 0xFF, &val, _MM_UPCONV_PS_SINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);}
static FORCEINLINE __vec8_f __cast_sitofp(__vec8_f, __vec8_i16 val) {return _mm512_mask_extload_ps(FZERO, 0xFF, &val, _MM_UPCONV_PS_SINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);}
static FORCEINLINE __vec8_f __cast_sitofp(__vec8_f, __vec8_i32 val) {return _mm512_mask_cvtfxpnt_round_adjustepi32_ps(FZERO, 0xFF, val, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);}
#endif
CAST(__vec8_f, float, __vec8_i64,  int64_t, __cast_sitofp)
#if 0
CAST(__vec8_d, double, __vec8_i8,  int8_t,  __cast_sitofp)
CAST(__vec8_d, double, __vec8_i16, int16_t, __cast_sitofp)
CAST(__vec8_d, double, __vec8_i32, int32_t, __cast_sitofp)
#else
static FORCEINLINE __vec8_d __cast_sitofp(__vec8_d, __vec8_i8 val) {
    __vec8_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
    return  _mm512_cvtepi32lo_pd(vi);
}

static FORCEINLINE __vec8_d __cast_sitofp(__vec8_d, __vec8_i16 val) {
    __vec8_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
    return  _mm512_cvtepi32lo_pd(vi);
}

static FORCEINLINE __vec8_d __cast_sitofp(__vec8_d, __vec8_i32 val) {
    __vec8_d ret;
    return _mm512_cvtepi32lo_pd(val);
}
#endif
CAST(__vec8_d, double, __vec8_i64, int64_t, __cast_sitofp)

// unsigned int to float/double
#if 0
CAST(__vec8_f, float, __vec8_i8,   uint8_t,  __cast_uitofp)
CAST(__vec8_f, float, __vec8_i16,  uint16_t, __cast_uitofp)
CAST(__vec8_f, float, __vec8_i32,  uint32_t, __cast_uitofp)
#else
static FORCEINLINE __vec8_f __cast_uitofp(__vec8_f, __vec8_i8  val) {return _mm512_mask_extload_ps(FZERO, 0xFF, &val, _MM_UPCONV_PS_UINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);}
static FORCEINLINE __vec8_f __cast_uitofp(__vec8_f, __vec8_i16 val) {return _mm512_mask_extload_ps(FZERO, 0xFF, &val, _MM_UPCONV_PS_UINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);}
static FORCEINLINE __vec8_f __cast_uitofp(__vec8_f, __vec8_i32 val) {return _mm512_mask_cvtfxpnt_round_adjustepu32_ps(FZERO, 0xFF, val, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);}
#endif
CAST(__vec8_f, float, __vec8_i64,  uint64_t, __cast_uitofp)
#if 0
CAST(__vec8_d, double, __vec8_i8,  uint8_t,  __cast_uitofp)
CAST(__vec8_d, double, __vec8_i16, uint16_t, __cast_uitofp)
CAST(__vec8_d, double, __vec8_i32, uint32_t, __cast_uitofp)
#else
static FORCEINLINE __vec8_d __cast_uitofp(__vec8_d, __vec8_i8 val) {
    __vec8_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
    return  _mm512_cvtepu32lo_pd(vi);
}

static FORCEINLINE __vec8_d __cast_uitofp(__vec8_d, __vec8_i16 val) {
    __vec8_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_UINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
    return _mm512_cvtepu32lo_pd(vi);
}

static FORCEINLINE __vec8_d __cast_uitofp(__vec8_d, __vec8_i32 val) {
    __vec8_d ret;
    return _mm512_cvtepu32lo_pd(val);
}
#endif
CAST(__vec8_d, double, __vec8_i64, uint64_t, __cast_uitofp)

#if 0
static FORCEINLINE __vec8_f __cast_uitofp(__vec8_f, __vec8_i1 v) {
    __vec8_f ret;
    for (int i = 0; i < 8; ++i)
        ret[i] = (v.v & (1 << i)) ? 1. : 0.;
    return ret;
}
#else
static FORCEINLINE __vec8_f __cast_uitofp(__vec8_f, __vec8_i1 v) 
{
    const __m512 ret = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0);
    return _mm512_mask_mov_ps(ret, v & 0xFF, one);
}
#endif

// float/double to signed int
CAST(__vec8_i8,  int8_t,  __vec8_f, float, __cast_fptosi)
CAST(__vec8_i16, int16_t, __vec8_f, float, __cast_fptosi)
#if 0
CAST(__vec8_i32, int32_t, __vec8_f, float, __cast_fptosi)
#else
static FORCEINLINE __vec8_i32 __cast_fptosi(__vec8_i32, __vec8_f val) {
  return _mm512_mask_cvtfxpnt_round_adjustps_epi32(IZERO, 0xFF, val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}
#endif
CAST(__vec8_i64, int64_t, __vec8_f, float, __cast_fptosi)
CAST(__vec8_i8,  int8_t,  __vec8_d, double, __cast_fptosi)
CAST(__vec8_i16, int16_t, __vec8_d, double, __cast_fptosi)
#if 1
CAST(__vec8_i32, int32_t, __vec8_d, double, __cast_fptosi)
#else
#endif
CAST(__vec8_i64, int64_t, __vec8_d, double, __cast_fptosi)

// float/double to unsigned int
CAST(__vec8_i8,  uint8_t,  __vec8_f, float, __cast_fptoui)
CAST(__vec8_i16, uint16_t, __vec8_f, float, __cast_fptoui)
#if 0
CAST(__vec8_i32, uint32_t, __vec8_f, float, __cast_fptoui)
#else
static FORCEINLINE __vec8_i32 __cast_fptoui(__vec8_i32, __vec8_f val) {
  return _mm512_mask_cvtfxpnt_round_adjustps_epu32(IZERO, 0xFF, val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}
#endif
CAST(__vec8_i64, uint64_t, __vec8_f, float, __cast_fptoui)
CAST(__vec8_i8,  uint8_t,  __vec8_d, double, __cast_fptoui)
CAST(__vec8_i16, uint16_t, __vec8_d, double, __cast_fptoui)
#if 1
CAST(__vec8_i32, uint32_t, __vec8_d, double, __cast_fptoui)
#else
#endif
CAST(__vec8_i64, uint64_t, __vec8_d, double, __cast_fptoui)

// float/double conversions
#if 0
CAST(__vec8_f, float,  __vec8_d, double, __cast_fptrunc)
CAST(__vec8_d, double, __vec8_f, float,  __cast_fpext)
#else
static FORCEINLINE __vec8_f __cast_fptrunc(__vec8_f, __vec8_d val) {
    return _mm512_mask_cvtpd_pslo(FZERO, 0xFF, val);
}
static FORCEINLINE __vec8_d __cast_fpext(__vec8_d, __vec8_f val) {
    return _mm512_cvtpslo_pd(val);
}
#endif

typedef union {
    int32_t i32;
    float f;
    int64_t i64;
    double d;
} BitcastUnion;

#define CAST_BITS(TO, TO_ELT, FROM, FROM_ELT)       \
static FORCEINLINE TO __cast_bits(TO, FROM val) {   \
    TO r;                                           \
    for (int i = 0; i < 8; ++i) {                  \
        BitcastUnion u;                             \
        u.FROM_ELT = val[i];                      \
        r[i] = u.TO_ELT;                          \
    }                                               \
    return r;                                       \
}

#if 0
CAST_BITS(__vec8_f,   f,   __vec8_i32, i32)
CAST_BITS(__vec8_i32, i32, __vec8_f,   f)
#else
static FORCEINLINE __vec8_f __cast_bits(__vec8_f, __vec8_i32 val) {
    return _mm512_castsi512_ps(val);
}
static FORCEINLINE __vec8_i32 __cast_bits(__vec8_i32, __vec8_f val) {
    return _mm512_castps_si512(val);
}
#endif

#if 0
CAST_BITS(__vec8_d,   d,   __vec8_i64, i64)
CAST_BITS(__vec8_i64, i64, __vec8_d,   d)
#else
static FORCEINLINE __vec8_i64 __cast_bits(__vec8_i64, __vec8_d val) {
    return *(__vec8_i64*)&val;
}
static FORCEINLINE __vec8_d __cast_bits(__vec8_d, __vec8_i64 val) {
    return *(__vec8_d*)&val;
}
#endif

#define CAST_BITS_SCALAR(TO, FROM)                  \
static FORCEINLINE TO __cast_bits(TO, FROM v) {     \
    union {                                         \
    TO to;                                          \
    FROM from;                                      \
    } u;                                            \
    u.from = v;                                     \
    return u.to;                                    \
}

CAST_BITS_SCALAR(uint32_t, float)
CAST_BITS_SCALAR(int32_t, float)
CAST_BITS_SCALAR(float, uint32_t)
CAST_BITS_SCALAR(float, int32_t)
CAST_BITS_SCALAR(uint64_t, double)
CAST_BITS_SCALAR(int64_t, double)
CAST_BITS_SCALAR(double, uint64_t)
CAST_BITS_SCALAR(double, int64_t)

///////////////////////////////////////////////////////////////////////////
// various math functions

static FORCEINLINE void __fastmath() {
}

static FORCEINLINE float __round_uniform_float(float v) {
    return roundf(v);
}

static FORCEINLINE float __floor_uniform_float(float v)  {
    return floorf(v);
}

static FORCEINLINE float __ceil_uniform_float(float v) {
    return ceilf(v);
}

static FORCEINLINE double __round_uniform_double(double v) {
    return round(v);
}

static FORCEINLINE double __floor_uniform_double(double v) {
    return floor(v);
}

static FORCEINLINE double __ceil_uniform_double(double v) {
    return ceil(v);
}

#if 0
UNARY_OP(__vec8_f, __round_varying_float, roundf)
UNARY_OP(__vec8_f, __floor_varying_float, floorf)
UNARY_OP(__vec8_f, __ceil_varying_float, ceilf)
#else
static FORCEINLINE __vec8_f __round_varying_float(__vec8_f v) {
  return _mm512_mask_round_ps(FZERO, 0xFF, v, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);
}

static FORCEINLINE __vec8_f __floor_varying_float(__vec8_f v) {
  return _mm512_mask_floor_ps(FZERO, 0xFF, v);
}

static FORCEINLINE __vec8_f __ceil_varying_float(__vec8_f v) {
  return _mm512_mask_ceil_ps(FZERO, 0xFF, v);
}
#endif

#if 0
UNARY_OP(__vec8_d, __round_varying_double, round)
UNARY_OP(__vec8_d, __floor_varying_double, floor)
UNARY_OP(__vec8_d, __ceil_varying_double, ceil)
#else
static FORCEINLINE __vec8_d __round_varying_float(__vec8_d v) {
  return _mm512_svml_round_pd(v);
}

static FORCEINLINE __vec8_d __floor_varying_float(__vec8_d v) {
  return _mm512_floor_pd(v);
}

static FORCEINLINE __vec8_d __ceil_varying_float(__vec8_d v) {
  return _mm512_ceil_pd(v);
}
#endif


// min/max

static FORCEINLINE float __min_uniform_float(float a, float b) { return (a<b) ? a : b; }
static FORCEINLINE float __max_uniform_float(float a, float b) { return (a>b) ? a : b; }
static FORCEINLINE double __min_uniform_double(double a, double b) { return (a<b) ? a : b; }
static FORCEINLINE double __max_uniform_double(double a, double b) { return (a>b) ? a : b; }

static FORCEINLINE int32_t __min_uniform_int32(int32_t a, int32_t b) { return (a<b) ? a : b; }
static FORCEINLINE int32_t __max_uniform_int32(int32_t a, int32_t b) { return (a>b) ? a : b; }
static FORCEINLINE int32_t __min_uniform_uint32(uint32_t a, uint32_t b) { return (a<b) ? a : b; }
static FORCEINLINE int32_t __max_uniform_uint32(uint32_t a, uint32_t b) { return (a>b) ? a : b; }

static FORCEINLINE int64_t __min_uniform_int64(int64_t a, int64_t b) { return (a<b) ? a : b; }
static FORCEINLINE int64_t __max_uniform_int64(int64_t a, int64_t b) { return (a>b) ? a : b; }
static FORCEINLINE int64_t __min_uniform_uint64(uint64_t a, uint64_t b) { return (a<b) ? a : b; }
static FORCEINLINE int64_t __max_uniform_uint64(uint64_t a, uint64_t b) { return (a>b) ? a : b; }


#if 0
BINARY_OP_FUNC(__vec8_f, __max_varying_float, __max_uniform_float)
BINARY_OP_FUNC(__vec8_f, __min_varying_float, __min_uniform_float)
#else
static FORCEINLINE __vec8_f __max_varying_float (__vec8_f v1, __vec8_f v2) { return _mm512_mask_gmax_ps(FZERO, 0xFF, v1, v2);}
static FORCEINLINE __vec8_f __min_varying_float (__vec8_f v1, __vec8_f v2) { return _mm512_mask_gmin_ps(FZERO, 0xFF, v1, v2);}
#endif

#if 0
BINARY_OP_FUNC(__vec8_d, __max_varying_double, __max_uniform_double)
BINARY_OP_FUNC(__vec8_d, __min_varying_double, __min_uniform_double)
#else
static FORCEINLINE __vec8_d __max_varying_double(__vec8_d v1, __vec8_d v2) { return _mm512_gmax_pd(v1,v2); }
static FORCEINLINE __vec8_d __min_varying_double(__vec8_d v1, __vec8_d v2) { return _mm512_gmin_pd(v1,v2); }
#endif

#if 0
BINARY_OP_FUNC(__vec8_i32, __max_varying_int32, __max_uniform_int32)
BINARY_OP_FUNC(__vec8_i32, __min_varying_int32, __min_uniform_int32)
BINARY_OP_FUNC(__vec8_i32, __max_varying_uint32, __max_uniform_uint32)
BINARY_OP_FUNC(__vec8_i32, __min_varying_uint32, __min_uniform_uint32)
#else
static FORCEINLINE __vec8_i32 __max_varying_int32 (__vec8_i32 v1, __vec8_i32 v2) { return _mm512_mask_max_epi32(IZERO,0xFF, v1, v2);}
static FORCEINLINE __vec8_i32 __min_varying_int32 (__vec8_i32 v1, __vec8_i32 v2) { return _mm512_mask_min_epi32(IZERO,0xFF, v1, v2);}
static FORCEINLINE __vec8_i32 __max_varying_uint32(__vec8_i32 v1, __vec8_i32 v2) { return _mm512_mask_max_epu32(IZERO,0xFF, v1, v2);}
static FORCEINLINE __vec8_i32 __min_varying_uint32(__vec8_i32 v1, __vec8_i32 v2) { return _mm512_mask_min_epu32(IZERO,0xFF, v1, v2);}
#endif

BINARY_OP_FUNC(__vec8_i64, __max_varying_int64, __max_uniform_int64)
BINARY_OP_FUNC(__vec8_i64, __min_varying_int64, __min_uniform_int64)
BINARY_OP_FUNC(__vec8_i64, __max_varying_uint64, __max_uniform_uint64)
BINARY_OP_FUNC(__vec8_i64, __min_varying_uint64, __min_uniform_uint64)

// sqrt/rsqrt/rcp

static FORCEINLINE float __rsqrt_uniform_float(float v) {
    return 1.f / sqrtf(v);
}

static FORCEINLINE float __rcp_uniform_float(float v) {
    return 1.f / v;
}

static FORCEINLINE float __sqrt_uniform_float(float v) {
    return sqrtf(v);
}

static FORCEINLINE double __sqrt_uniform_double(double v) {
    return sqrt(v);
}

#if 0
UNARY_OP(__vec8_f, __rcp_varying_float, __rcp_uniform_float)
UNARY_OP(__vec8_f, __rsqrt_varying_float, __rsqrt_uniform_float)
UNARY_OP(__vec8_f, __sqrt_varying_float, __sqrt_uniform_float)
#else
static FORCEINLINE __vec8_f __rcp_varying_float(__vec8_f v) {
#ifdef ISPC_FAST_MATH
    return _mm512_mask_rcp23_ps(FZERO, 0xFF, v); // Approximation with 23 bits of accuracy.
#else
    return _mm512_mask_recip_ps(FZERO, 0xFF, v);
#endif
}
static FORCEINLINE __vec8_d __rcp_varying_double(__vec8_d x) {
    __vec8_i64 ex = __and(__cast_bits(__vec8_i64(), x), __smear_i64<__vec8_i64>(0x7fe0000000000000));
    __vec8_d  exp = __cast_bits(__vec8_d(), __sub(__smear_i64<__vec8_i64>(0x7fd0000000000000), ex));
    __vec8_f   xf = __cast_fptrunc(__vec8_f(), __mul(x, exp));
    __vec8_f   yf = __rcp_varying_float(xf);
    __vec8_d    y = __mul(__cast_fpext(__vec8_d(), yf), exp);
    y = __add(y, __mul(y, __sub(__smear_double<__vec8_d>(2.0), __mul(x, y))));
    y = __add(y, __mul(y, __sub(__smear_double<__vec8_d>(2.0), __mul(x, y))));
    return y;
}
static FORCEINLINE double __rcp_uniform_double(double v) 
{
  return __extract_element(__rcp_varying_double(__smear_double<__vec8_d>(v)),0);
}

static FORCEINLINE __vec8_f __rsqrt_varying_float(__vec8_f v) {
#ifdef ISPC_FAST_MATH
    return _mm512_mask_rsqrt23_ps(FZERO,0xFF,v); // Approximation with 0.775ULP accuracy
#else 
    return _mm512_mask_invsqrt_ps(FZERO,0xFF,v);
#endif
}
static FORCEINLINE __vec8_d __rsqrt_varying_double(__vec8_d x) {
    __vec8_i64 ex = __and(__cast_bits(__vec8_i64(), x), __smear_i64<__vec8_i64>(0x7fe0000000000000));
    __vec8_d  exp = __cast_bits(__vec8_d(), __sub(__smear_i64<__vec8_i64>(0x7fd0000000000000), ex));
    __vec8_d exph = __cast_bits(__vec8_d(), __sub(__smear_i64<__vec8_i64>(0x5fe0000000000000), __lshr(ex,1)));
    __vec8_f   xf = __cast_fptrunc(__vec8_f(), __mul(x, exp));
    __vec8_f   yf = __rsqrt_varying_float(xf);
    __vec8_d    y = __mul(__cast_fpext(__vec8_d(), yf), exph);
    __vec8_d   xh = __mul(x, __smear_double<__vec8_d>(0.5));
    y = __add(y, __mul(y, __sub(__smear_double<__vec8_d>(0.5), __mul(xh, __mul(y,y)))));
    y = __add(y, __mul(y, __sub(__smear_double<__vec8_d>(0.5), __mul(xh, __mul(y,y)))));
    return y;
}
static FORCEINLINE double __rsqrt_uniform_double(double v) 
{
  return __extract_element(__rsqrt_varying_double(__smear_double<__vec8_d>(v)),0);
}
static FORCEINLINE __vec8_f __sqrt_varying_float (__vec8_f v) {    return _mm512_mask_sqrt_ps(FZERO,0xFF,v);}
#endif

#if 0
UNARY_OP(__vec8_d, __sqrt_varying_double, __sqrt_uniform_double)
#else
static FORCEINLINE __vec8_d __sqrt_varying_double(__vec8_d v) {    return _mm512_sqrt_pd(v); }
#endif

///////////////////////////////////////////////////////////////////////////
// svml
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec8_f __svml_logf(__vec8_f v)              { return _mm512_mask_log_ps(FZERO,0xFF,v); }
static FORCEINLINE __vec8_f __svml_expf(__vec8_f v)              { return _mm512_mask_exp_ps(FZERO,0xFF,v); }
static FORCEINLINE __vec8_f __svml_cosf(__vec8_f v)              { return _mm512_mask_cos_ps(FZERO,0xFF,v); }
static FORCEINLINE __vec8_f __svml_powf(__vec8_f a, __vec8_f b) { return _mm512_mask_pow_ps(FZERO,0xFF,a,b); }

static FORCEINLINE __vec8_d __svml_logd(__vec8_d v)              { return _mm512_log_pd(v); }
static FORCEINLINE __vec8_d __svml_expd(__vec8_d v)              { return _mm512_exp_pd(v); }
static FORCEINLINE __vec8_d __svml_cosd(__vec8_d v)              { return _mm512_cos_pd(v); }
static FORCEINLINE __vec8_d __svml_powd(__vec8_d a, __vec8_d b) { return _mm512_pow_pd(a,b); }

///////////////////////////////////////////////////////////////////////////
// bit ops

static FORCEINLINE int32_t __popcnt_int32(uint32_t v) {
    int count = 0;
    for (; v != 0; v >>= 1)
        count += (v & 1);
    return count;
}

static FORCEINLINE int32_t __popcnt_int64(uint64_t v) {
    int count = 0;
    for (; v != 0; v >>= 1)
        count += (v & 1);
    return count;
}

static FORCEINLINE int32_t __count_trailing_zeros_i32(uint32_t v) {
    if (v == 0)
        return 32;

    int count = 0;
    while ((v & 1) == 0) {
        ++count;
        v >>= 1;
    }
    return count;
}

static FORCEINLINE int64_t __count_trailing_zeros_i64(uint64_t v) {
    if (v == 0)
        return 64;

    int count = 0;
    while ((v & 1) == 0) {
        ++count;
        v >>= 1;
    }
    return count;
}

static FORCEINLINE int32_t __count_leading_zeros_i32(uint32_t v) {
    if (v == 0)
        return 32;

    int count = 0;
    while ((v & (1<<31)) == 0) {
        ++count;
        v <<= 1;
    }
    return count;
}

static FORCEINLINE int64_t __count_leading_zeros_i64(uint64_t v) {
    if (v == 0)
        return 64;

    int count = 0;
    while ((v & (1ull<<63)) == 0) {
        ++count;
        v <<= 1;
    }
    return count;
}

///////////////////////////////////////////////////////////////////////////
// reductions

#if 0
REDUCE_ADD(float, __vec8_f, __reduce_add_float)
REDUCE_MINMAX(float, __vec8_f, __reduce_min_float, <)
REDUCE_MINMAX(float, __vec8_f, __reduce_max_float, >)
#else
static FORCEINLINE float __reduce_add_float(__vec8_f v) { return _mm512_mask_reduce_add_ps(0xFF,v); }
static FORCEINLINE float __reduce_min_float(__vec8_f v) { return _mm512_mask_reduce_min_ps(0xFF,v); }
static FORCEINLINE float __reduce_max_float(__vec8_f v) { return _mm512_mask_reduce_max_ps(0xFF,v); }
#endif

#if 0
REDUCE_ADD(double, __vec8_d, __reduce_add_double)
REDUCE_MINMAX(double, __vec8_d, __reduce_min_double, <)
REDUCE_MINMAX(double, __vec8_d, __reduce_max_double, >)
#else
static FORCEINLINE float __reduce_add_double(__vec8_d v) { return _mm512_reduce_add_pd(v); }
static FORCEINLINE float __reduce_min_double(__vec8_d v) { return _mm512_reduce_min_pd(v); }
static FORCEINLINE float __reduce_max_double(__vec8_d v) { return _mm512_reduce_max_pd(v); }
#endif



#if 0
REDUCE_ADD   (int64_t, __vec8_i32, __reduce_add_int32)
REDUCE_MINMAX(int32_t, __vec8_i32, __reduce_min_int32, <)
REDUCE_MINMAX(int32_t, __vec8_i32, __reduce_max_int32, >)
REDUCE_MINMAX(uint32_t, __vec8_i32, __reduce_min_uint32, <)
REDUCE_MINMAX(uint32_t, __vec8_i32, __reduce_max_uint32, >)
#else
static FORCEINLINE  int64_t __reduce_add_int32  (__vec8_i32 v) { return _mm512_mask_reduce_add_epi32(0xFF, v);}
static FORCEINLINE  int32_t __reduce_min_int32  (__vec8_i32 v) { return _mm512_mask_reduce_min_epi32(0xFF, v);}
static FORCEINLINE  int32_t __reduce_max_int32  (__vec8_i32 v) { return _mm512_mask_reduce_max_epi32(0xFF, v);}
static FORCEINLINE uint32_t __reduce_min_uint32 (__vec8_i32 v) { return _mm512_mask_reduce_min_epu32(0xFF, v);}
static FORCEINLINE uint32_t __reduce_max_uint32 (__vec8_i32 v) { return _mm512_mask_reduce_max_epu32(0xFF, v);}
#endif

REDUCE_ADD   ( int16_t, __vec8_i8,  __reduce_add_int8)
REDUCE_ADD   ( int32_t, __vec8_i16, __reduce_add_int16)
REDUCE_ADD   ( int64_t, __vec8_i64, __reduce_add_int64)
REDUCE_MINMAX( int64_t, __vec8_i64, __reduce_min_int64, <)
REDUCE_MINMAX( int64_t, __vec8_i64, __reduce_max_int64, >)
REDUCE_MINMAX(uint64_t, __vec8_i64, __reduce_min_uint64, <)
REDUCE_MINMAX(uint64_t, __vec8_i64, __reduce_max_uint64, >)

///////////////////////////////////////////////////////////////////////////
// masked load/store

static FORCEINLINE __vec8_i8 __masked_load_i8(void *p,
                                               __vec8_i1 mask) {
    __vec8_i8 ret;
    int8_t *ptr = (int8_t *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec8_i16 __masked_load_i16(void *p,
                                                 __vec8_i1 mask) {
    __vec8_i16 ret;
    int16_t *ptr = (int16_t *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}

#if 0
static FORCEINLINE __vec8_i32 __masked_load_i32(void *p,
                                                 __vec8_i1 mask) {
    __vec8_i32 ret;
    int32_t *ptr = (int32_t *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}
#else
static FORCEINLINE __vec8_i32 __masked_load_i32(void *p, __vec8_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    return _mm512_mask_load_epi32(__vec8_i32(), mask, p);
#else
    __vec8_i32 tmp;
    tmp = _mm512_mask_extloadunpacklo_epi32(tmp, 0xFF, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp = _mm512_mask_extloadunpackhi_epi32(tmp, 0xFF, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    __vec8_i32 ret;
    return _mm512_mask_mov_epi32(ret, 0xFF & mask, tmp);
#endif
}
#endif

#if 0
static FORCEINLINE __vec8_f __masked_load_float(void *p,
                                                 __vec8_i1 mask) {
    __vec8_f ret;
    float *ptr = (float *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}
#else
static FORCEINLINE __vec8_f __masked_load_float(void *p, __vec8_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    return _mm512_mask_load_ps(_mm512_undefined_ps(), mask,p);
#else
    __vec8_f tmp;
    tmp = _mm512_mask_extloadunpacklo_ps(tmp, 0xFF, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp = _mm512_mask_extloadunpackhi_ps(tmp, 0xFF, (uint8_t*)p+64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    __vec8_f ret;
    return _mm512_mask_mov_ps(ret, 0xFF & mask, tmp);
#endif
}
#endif

static FORCEINLINE __vec8_i64 __masked_load_i64(void *p,
                                                 __vec8_i1 mask) {
    __vec8_i64 ret;
    int64_t *ptr = (int64_t *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}

#if 0
static FORCEINLINE __vec8_d __masked_load_double(void *p,
                                                  __vec8_i1 mask) {
    __vec8_d ret;
    double *ptr = (double *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}
#else
static FORCEINLINE __vec8_d __masked_load_double(void *p, __vec8_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    __vec8_d ret = FZERO;
    ret = _mm512_mask_load_pd(ret, 0xFF & mask, p);
    return ret;
#else
    __vec8_d tmp = FZERO;
    tmp.v = _mm512_mask_extloadunpacklo_pd(tmp.v, 0xFF, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
    tmp.v = _mm512_mask_extloadunpackhi_pd(tmp.v, 0xFF, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
    __vec8_d ret = FZERO;
    ret.v = _mm512_mask_mov_pd(ret.v, mask, tmp.v);
    return ret;
#endif
}
#endif


static FORCEINLINE void __masked_store_i8(void *p, __vec8_i8 val,
                                          __vec8_i1 mask) {
    int8_t *ptr = (int8_t *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val[i];
}

static FORCEINLINE void __masked_store_i16(void *p, __vec8_i16 val,
                                           __vec8_i1 mask) {
    int16_t *ptr = (int16_t *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val[i];
}

#if 0
static FORCEINLINE void __masked_store_i32(void *p, __vec8_i32 val,
                                           __vec8_i1 mask) {
    int32_t *ptr = (int32_t *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val[i];
}
#else
static FORCEINLINE void __masked_store_i32(void *p, __vec8_i32 val, __vec8_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    _mm512_mask_store_epi32(p, mask, val.v);
#else
    __vec8_i32 tmp;
    tmp = _mm512_extloadunpacklo_epi32(tmp, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp = _mm512_extloadunpackhi_epi32(tmp, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp = _mm512_mask_mov_epi32(tmp, 0xFF & mask, val);
    _mm512_mask_extpackstorelo_epi32(          p,    0xFF, tmp, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_mask_extpackstorehi_epi32((uint8_t*)p+64, 0xFF, tmp, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
#endif
}
#endif

#if 0
static FORCEINLINE void __masked_store_float(void *p, __vec8_f val,
                                             __vec8_i1 mask) {
    float *ptr = (float *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val[i];
}
#else
static FORCEINLINE void __masked_store_float(void *p, __vec8_f val,
                                             __vec8_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    _mm512_mask_store_ps(p, 0xFF & mask, val.v);
#else
    __vec8_f tmp = FZERO;
    tmp = _mm512_extloadunpacklo_ps(tmp, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp = _mm512_extloadunpackhi_ps(tmp, (uint8_t*)p+64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp = _mm512_mask_mov_ps(tmp, 0xFF & mask, val);
    _mm512_mask_extpackstorelo_ps(          p,    0xFF, tmp, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_mask_extpackstorehi_ps((uint8_t*)p+64, 0xFF, tmp, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
}
#endif

static FORCEINLINE void __masked_store_i64(void *p, __vec8_i64 val,
                                          __vec8_i1 mask) {
    int64_t *ptr = (int64_t *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val[i];
}

#if 0
static FORCEINLINE void __masked_store_double(void *p, __vec8_d val,
                                              __vec8_i1 mask) {
    double *ptr = (double *)p;
    for (int i = 0; i < 8; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val[i];
}
#else
static FORCEINLINE void __masked_store_double(void *p, __vec8_d val,
                                              __vec8_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    _mm512_mask_store_pd(p, mask, val.v);
#else
    __vec8_d tmp;
    tmp.v = _mm512_extloadunpacklo_pd(tmp.v, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
    tmp.v = _mm512_extloadunpackhi_pd(tmp.v, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
    tmp.v = _mm512_mask_mov_pd(tmp.v, mask, val.v);
    _mm512_extpackstorelo_pd(p, tmp.v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_pd((uint8_t*)p+64, tmp.v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
#endif
}
#endif

static FORCEINLINE void __masked_store_blend_i8(void *p, __vec8_i8 val,
                                                __vec8_i1 mask) {
    __masked_store_i8(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i16(void *p, __vec8_i16 val,
                                                 __vec8_i1 mask) {
    __masked_store_i16(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i32(void *p, __vec8_i32 val,
                                                 __vec8_i1 mask) {
    __masked_store_i32(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_float(void *p, __vec8_f val,
                                                   __vec8_i1 mask) {
    __masked_store_float(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i64(void *p, __vec8_i64 val,
                                                 __vec8_i1 mask) {
    __masked_store_i64(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_double(void *p, __vec8_d val,
                                                    __vec8_i1 mask) {
    __masked_store_double(p, val, mask);
}

///////////////////////////////////////////////////////////////////////////
// gather/scatter

// offsets * offsetScale is in bytes (for all of these)

#define GATHER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)                  \
static FORCEINLINE VTYPE FUNC(unsigned char *b, uint32_t scale,         \
                              OTYPE offset, __vec8_i1 mask) {          \
    VTYPE ret;                                                          \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 8; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * offset[i]);         \
            ret[i] = *ptr;                                            \
        }                                                               \
    return ret;                                                         \
}
    

/****************/
#if 0
GATHER_BASE_OFFSETS(__vec8_i8,  int8_t,  __vec8_i32, __gather_base_offsets32_i8)
#else
static FORCEINLINE __vec8_i8 __gather_base_offsets32_i8(uint8_t *base, uint32_t scale, __vec8_i32 offsets,  __vec8_i1 mask) 
{
    // (iw): need to temporarily store as int because gathers can only return ints.
    __vec8_i32 tmp = _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), 0xFF & mask, offsets, base, 
                                                     _MM_UPCONV_EPI32_SINT8, scale,
                                                     _MM_HINT_NONE);
    // now, downconverting to chars into temporary char vector
    __vec8_i8 ret;
    _mm512_mask_extstore_epi32(ret.data,0xFF,tmp,_MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
    return ret;
}
#endif
GATHER_BASE_OFFSETS(__vec8_i8,  int8_t,  __vec8_i64, __gather_base_offsets64_i8)
/****************/
GATHER_BASE_OFFSETS(__vec8_i16, int16_t, __vec8_i32, __gather_base_offsets32_i16)
GATHER_BASE_OFFSETS(__vec8_i16, int16_t, __vec8_i64, __gather_base_offsets64_i16)
/****************/
#if 0
GATHER_BASE_OFFSETS(__vec8_i32, int32_t, __vec8_i32, __gather_base_offsets32_i32)
#else
static FORCEINLINE __vec8_i32 __gather_base_offsets32_i32(uint8_t *base, uint32_t scale, __vec8_i32 offsets,   __vec8_i1 mask) 
{
    return _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), 0xFF & mask, offsets, 
                                          base, _MM_UPCONV_EPI32_NONE, scale,
                                          _MM_HINT_NONE);
}
#endif
GATHER_BASE_OFFSETS(__vec8_i32, int32_t, __vec8_i64, __gather_base_offsets64_i32)
/****************/
#if 0
GATHER_BASE_OFFSETS(__vec8_f,   float,   __vec8_i32, __gather_base_offsets32_float)
#else
static FORCEINLINE __vec8_f __gather_base_offsets32_float(uint8_t *base, uint32_t scale, __vec8_i32 offsets, __vec8_i1 mask) 
{
    return _mm512_mask_i32extgather_ps(_mm512_undefined_ps(), 0xFF & mask, offsets,
                                       base, _MM_UPCONV_PS_NONE, scale,
                                       _MM_HINT_NONE);
}
#endif
GATHER_BASE_OFFSETS(__vec8_f,   float,   __vec8_i64, __gather_base_offsets64_float)
/****************/
GATHER_BASE_OFFSETS(__vec8_i64, int64_t, __vec8_i32, __gather_base_offsets32_i64)
GATHER_BASE_OFFSETS(__vec8_i64, int64_t, __vec8_i64, __gather_base_offsets64_i64)
/****************/
#if 0
GATHER_BASE_OFFSETS(__vec8_d,   double,  __vec8_i32, __gather_base_offsets32_double)
#else
static FORCEINLINE __vec8_d __gather_base_offsets32_double(uint8_t *base, uint32_t scale, __vec8_i32 offsets, __vec8_i1 mask) 
{
    __vec8_d ret;
    ret.v = _mm512_mask_i32loextgather_pd(_mm512_undefined_pd(), mask, offsets,
                                       base, _MM_UPCONV_PD_NONE, scale,
                                       _MM_HINT_NONE); 
#if 0
    __m512i shuffled_offsets = _mm512_permute4f128_epi32(offsets.v, _MM_PERM_DCDC);
    const __mmask8 mask8 = 0x00FF & (mask >> 8); /* evghenii::testme */
    ret.v2 = _mm512_mask_i32loextgather_pd(_mm512_undefined_pd(), mask8, shuffled_offsets,
                                       base, _MM_UPCONV_PD_NONE, scale,
                                       _MM_HINT_NONE); 
#endif
    return ret;
}
#endif
GATHER_BASE_OFFSETS(__vec8_d,   double,  __vec8_i64, __gather_base_offsets64_double)

#define GATHER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)         \
static FORCEINLINE VTYPE FUNC(PTRTYPE ptrs, __vec8_i1 mask) {   \
    VTYPE ret;                                              \
    for (int i = 0; i < 8; ++i)                            \
        if ((mask.v & (1 << i)) != 0) {                     \
            STYPE *ptr = (STYPE *)ptrs[i];                \
            ret[i] = *ptr;                                \
        }                                                   \
    return ret;                                             \
}
#define GATHER_GENERALF(VTYPE, STYPE, PTRTYPE, FUNC,FUNC1)         \
static FORCEINLINE VTYPE FUNC(PTRTYPE ptrs, __vec8_i1 mask) {   \
  return FUNC1(0, 1, ptrs, mask); \
}


#if 1
/***********/
GATHER_GENERALF(__vec8_i8,  int8_t,  __vec8_i32, __gather32_i8, __gather_base_offsets32_i8)
GATHER_GENERALF(__vec8_i16, int16_t, __vec8_i32, __gather32_i16, __gather_base_offsets32_i16)
GATHER_GENERALF(__vec8_i32, int32_t, __vec8_i32, __gather32_i32, __gather_base_offsets32_i32)
GATHER_GENERALF(__vec8_i64, int64_t, __vec8_i32, __gather32_i64, __gather_base_offsets32_i64)
GATHER_GENERALF(__vec8_f,   float,   __vec8_i32, __gather32_float, __gather_base_offsets32_float)
GATHER_GENERALF(__vec8_d,   double,  __vec8_i32, __gather32_double, __gather_base_offsets32_double)
/***********/
GATHER_GENERAL(__vec8_i8,  int8_t,  __vec8_i64, __gather64_i8);
GATHER_GENERAL(__vec8_i16, int16_t, __vec8_i64, __gather64_i16);
GATHER_GENERAL(__vec8_i32, int32_t, __vec8_i64, __gather64_i32);
GATHER_GENERAL(__vec8_i64, int64_t, __vec8_i64, __gather64_i64);
GATHER_GENERAL(__vec8_f,   float,   __vec8_i64, __gather64_float);
GATHER_GENERAL(__vec8_d,   double,  __vec8_i64, __gather64_double);
/***********/
#endif

// scatter

#define SCATTER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)                 \
static FORCEINLINE void FUNC(unsigned char *b, uint32_t scale,          \
                             OTYPE offset, VTYPE val,                   \
                             __vec8_i1 mask) {                         \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 8; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * offset[i]);         \
            *ptr = val[i];                                            \
        }                                                               \
}
    

/*****************/
SCATTER_BASE_OFFSETS(__vec8_i8,  int8_t,  __vec8_i32, __scatter_base_offsets32_i8)
SCATTER_BASE_OFFSETS(__vec8_i8,  int8_t,  __vec8_i64, __scatter_base_offsets64_i8)
/*****************/
SCATTER_BASE_OFFSETS(__vec8_i16, int16_t, __vec8_i32, __scatter_base_offsets32_i16)
SCATTER_BASE_OFFSETS(__vec8_i16, int16_t, __vec8_i64, __scatter_base_offsets64_i16)
/*****************/
#if 0
SCATTER_BASE_OFFSETS(__vec8_i32, int32_t, __vec8_i32, __scatter_base_offsets32_i32)
#else
static FORCEINLINE void __scatter_base_offsets32_i32(uint8_t *b, uint32_t scale, __vec8_i32 offsets,  __vec8_i32 val, __vec8_i1 mask)
{
    _mm512_mask_i32extscatter_epi32(b, 0xFF & mask, offsets, val, 
                                    _MM_DOWNCONV_EPI32_NONE, scale, 
                                    _MM_HINT_NONE);
}
#endif
SCATTER_BASE_OFFSETS(__vec8_i32, int32_t, __vec8_i64, __scatter_base_offsets64_i32)
/*****************/
#if 0
SCATTER_BASE_OFFSETS(__vec8_f,   float,   __vec8_i32, __scatter_base_offsets32_float)
#else
static FORCEINLINE void __scatter_base_offsets32_float(void *base, uint32_t scale, __vec8_i32 offsets,
                               __vec8_f val, __vec8_i1 mask) 
{ 
    _mm512_mask_i32extscatter_ps(base, 0xFF & mask, offsets, val, 
                                 _MM_DOWNCONV_PS_NONE, scale,
                                 _MM_HINT_NONE);
}
#endif
SCATTER_BASE_OFFSETS(__vec8_f,   float,   __vec8_i64, __scatter_base_offsets64_float)
/*****************/
SCATTER_BASE_OFFSETS(__vec8_i64, int64_t, __vec8_i32, __scatter_base_offsets32_i64)
SCATTER_BASE_OFFSETS(__vec8_i64, int64_t, __vec8_i64, __scatter_base_offsets64_i64)
/*****************/
#if 0 /* evghenii::to implement */
SCATTER_BASE_OFFSETS(__vec8_d,   double,  __vec8_i32, __scatter_base_offsets32_double)
#else /* evghenii:testme */
static FORCEINLINE void __scatter_base_offsets32_double(void *base, uint32_t scale, __vec8_i32 offsets,
                               __vec8_d val, __vec8_i1 mask) 
{ 
    _mm512_mask_i32loextscatter_pd(base, mask, offsets, val.v,
                                 _MM_DOWNCONV_PD_NONE, scale,
                                 _MM_HINT_NONE);
}
#endif
SCATTER_BASE_OFFSETS(__vec8_d,   double,  __vec8_i64, __scatter_base_offsets64_double)

#define SCATTER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)                 \
static FORCEINLINE void FUNC(PTRTYPE ptrs, VTYPE val, __vec8_i1 mask) {  \
    VTYPE ret;                                                       \
    for (int i = 0; i < 8; ++i)                                     \
        if ((mask.v & (1 << i)) != 0) {                              \
            STYPE *ptr = (STYPE *)ptrs[i];                         \
            *ptr = val[i];                                         \
        }                                                            \
}
#define SCATTER_GENERALF(VTYPE, STYPE, PTRTYPE, FUNC,FUNC1)         \
static FORCEINLINE void FUNC(PTRTYPE ptrs, VTYPE val, __vec8_i1 mask) {  \
  return FUNC1(0, 1, ptrs, val, mask); \
}

#if 1
/***********/
SCATTER_GENERALF(__vec8_i8,  int8_t,  __vec8_i32, __scatter32_i8, __scatter_base_offsets32_i8)
SCATTER_GENERALF(__vec8_i16, int16_t, __vec8_i32, __scatter32_i16, __scatter_base_offsets32_i16)
SCATTER_GENERALF(__vec8_i32, int32_t, __vec8_i32, __scatter32_i32, __scatter_base_offsets32_i32)
SCATTER_GENERALF(__vec8_i64, int64_t, __vec8_i32, __scatter32_i64, __scatter_base_offsets32_i64)
SCATTER_GENERALF(__vec8_f,   float,   __vec8_i32, __scatter32_float, __scatter_base_offsets32_float)
SCATTER_GENERALF(__vec8_d,   double,  __vec8_i32, __scatter32_double, __scatter_base_offsets32_double)
/***********/
SCATTER_GENERAL(__vec8_i8,  int8_t,  __vec8_i64, __scatter64_i8)
SCATTER_GENERAL(__vec8_i16, int16_t, __vec8_i64, __scatter64_i16)
SCATTER_GENERAL(__vec8_i32, int32_t, __vec8_i64, __scatter64_i32)
SCATTER_GENERAL(__vec8_f,   float,   __vec8_i64, __scatter64_float)
SCATTER_GENERAL(__vec8_i64, int64_t, __vec8_i64, __scatter64_i64)
SCATTER_GENERAL(__vec8_d,   double,  __vec8_i64, __scatter64_double)
/***********/
#endif

///////////////////////////////////////////////////////////////////////////
// packed load/store

#if 0
static FORCEINLINE int32_t __packed_load_active(int32_t *ptr, __vec8_i32 *val,
                                                __vec8_i1 mask) {
    int count = 0; 
    for (int i = 0; i < 8; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            val->operator[](i) = *ptr++;
            ++count;
        }
    }
    return count;
}
static FORCEINLINE int32_t __packed_store_active(int32_t *ptr, 
                                                 __vec8_i32 val,
                                                 __vec8_i1 mask) {
    int count = 0; 
    for (int i = 0; i < 8; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            *ptr++ = val[i];
            ++count;
        }
    }
    return count;
}
static FORCEINLINE int32_t __packed_load_active(uint32_t *ptr,
                                                __vec8_i32 *val,
                                                __vec8_i1 mask) {
    int count = 0; 
    for (int i = 0; i < 8; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            val->operator[](i) = *ptr++;
            ++count;
        }
    }
    return count;
}
static FORCEINLINE int32_t __packed_store_active(uint32_t *ptr, 
                                                 __vec8_i32 val,
                                                 __vec8_i1 mask) {
    int count = 0; 
    for (int i = 0; i < 8; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            *ptr++ = val[i];
            ++count;
        }
    }
    return count;
}
#else
static FORCEINLINE int32_t __packed_load_active(uint32_t *p, __vec8_i32 *val,
                                                __vec8_i1 mask) {
    __vec8_i32 v = __load<64>(val);
    v = _mm512_mask_extloadunpacklo_epi32(v, 0xFF & mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v = _mm512_mask_extloadunpackhi_epi32(v, 0xFF & mask, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    __store<64>(val, v);
    return _mm_countbits_32(uint32_t(0xFF & mask));
}
static FORCEINLINE int32_t __packed_store_active(uint32_t *p, __vec8_i32 val,
                                                 __vec8_i1 mask) {
    _mm512_mask_extpackstorelo_epi32(p, 0xFF & mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_mask_extpackstorehi_epi32((uint8_t*)p+64, 0xFF & mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    return _mm_countbits_32(uint32_t(0xFF & mask));
}
static FORCEINLINE int32_t __packed_store_active2(uint32_t *ptr, __vec8_i32 val,
                                                 __vec8_i1 mask) {
    return __packed_store_active(ptr, val, mask);
}
static FORCEINLINE int32_t __packed_load_active(int32_t *p, __vec8_i32 *val,
                                                __vec8_i1 mask) {
    return __packed_load_active((uint32_t *)p, val, mask);
}
static FORCEINLINE int32_t __packed_store_active(int32_t *p, __vec8_i32 val,
                                                 __vec8_i1 mask) {
    return __packed_store_active((uint32_t *)p, val, mask);
}
static FORCEINLINE int32_t __packed_store_active2(int32_t *ptr, __vec8_i32 val,
                                                 __vec8_i1 mask) {
    return __packed_store_active(ptr, val, mask);
}

#endif

///////////////////////////////////////////////////////////////////////////
// aos/soa

static FORCEINLINE void __soa_to_aos3_float(__vec8_f v0, __vec8_f v1, __vec8_f v2,
                                            float *ptr) {
    for (int i = 0; i < 8; ++i) {
        *ptr++ = __extract_element(v0, i);
        *ptr++ = __extract_element(v1, i);
        *ptr++ = __extract_element(v2, i);
    }
}

static FORCEINLINE void __aos_to_soa3_float(float *ptr, __vec8_f *out0, __vec8_f *out1,
                                            __vec8_f *out2) {
    for (int i = 0; i < 8; ++i) {
        __insert_element(out0, i, *ptr++);
        __insert_element(out1, i, *ptr++);
        __insert_element(out2, i, *ptr++);
    }
}

static FORCEINLINE void __soa_to_aos4_float(__vec8_f v0, __vec8_f v1, __vec8_f v2,
                                            __vec8_f v3, float *ptr) {
    for (int i = 0; i < 8; ++i) {
        *ptr++ = __extract_element(v0, i);
        *ptr++ = __extract_element(v1, i);
        *ptr++ = __extract_element(v2, i);
        *ptr++ = __extract_element(v3, i);
    }
}

static FORCEINLINE void __aos_to_soa4_float(float *ptr, __vec8_f *out0, __vec8_f *out1,
                                            __vec8_f *out2, __vec8_f *out3) {
    for (int i = 0; i < 8; ++i) {
        __insert_element(out0, i, *ptr++);
        __insert_element(out1, i, *ptr++);
        __insert_element(out2, i, *ptr++);
        __insert_element(out3, i, *ptr++);
    }
}

///////////////////////////////////////////////////////////////////////////
// prefetch

static FORCEINLINE void __prefetch_read_uniform_1(unsigned char *p) {
    _mm_prefetch((char *)p, _MM_HINT_T0); // prefetch into L1$
}

static FORCEINLINE void __prefetch_read_uniform_2(unsigned char *p) {
    _mm_prefetch((char *)p, _MM_HINT_T1); // prefetch into L2$
}

static FORCEINLINE void __prefetch_read_uniform_3(unsigned char *p) {
    // There is no L3$ on KNC, don't want to pollute L2$ unecessarily
}

static FORCEINLINE void __prefetch_read_uniform_nt(unsigned char *p) {
    _mm_prefetch((char *)p, _MM_HINT_T2); // prefetch into L2$ with non-temporal hint
    // _mm_prefetch(p, _MM_HINT_NTA); // prefetch into L1$ with non-temporal hint
}

#define PREFETCH_READ_VARYING(CACHE_NUM, HINT)                                                              \
static FORCEINLINE void __prefetch_read_varying_##CACHE_NUM##_native(uint8_t *base, uint32_t scale,         \
                                                                   __vec16_i32 offsets, __vec16_i1 mask) {  \
    _mm512_mask_prefetch_i32gather_ps (offsets, mask, base, scale, HINT);                                   \
    offsets = _mm512_permutevar_epi32(_mm512_set_16to16_pi(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8), offsets);\
    __vec16_i1 copy_mask = _mm512_kmov(mask);                                                               \
    _mm512_kswapb(mask, copy_mask);                                                                         \
    _mm512_mask_prefetch_i32gather_ps (offsets, mask, base, scale, _MM_HINT_T0);                            \
}                                                                                                           \
static FORCEINLINE void __prefetch_read_varying_##CACHE_NUM(__vec16_i64 addr, __vec16_i1 mask) {}           \

PREFETCH_READ_VARYING(1, _MM_HINT_T0)
PREFETCH_READ_VARYING(2, _MM_HINT_T1)
PREFETCH_READ_VARYING(nt, _MM_HINT_T2)

static FORCEINLINE void __prefetch_read_varying_3_native(uint8_t *base, uint32_t scale,
                                                         __vec16_i32 offsets, __vec16_i1 mask) {}

static FORCEINLINE void __prefetch_read_varying_3(__vec16_i64 addr, __vec16_i1 mask) {}

///////////////////////////////////////////////////////////////////////////
// atomics

static FORCEINLINE uint32_t __atomic_add(uint32_t *p, uint32_t v) {
#ifdef _MSC_VER
    return InterlockedAdd((LONG volatile *)p, v) - v;
#else
    return __sync_fetch_and_add(p, v);
#endif
}

static FORCEINLINE uint32_t __atomic_sub(uint32_t *p, uint32_t v) {
#ifdef _MSC_VER
    return InterlockedAdd((LONG volatile *)p, -v) + v;
#else
    return __sync_fetch_and_sub(p, v);
#endif
}

static FORCEINLINE uint32_t __atomic_and(uint32_t *p, uint32_t v) {
#ifdef _MSC_VER
    return InterlockedAnd((LONG volatile *)p, v);
#else
    return __sync_fetch_and_and(p, v);
#endif
}

static FORCEINLINE uint32_t __atomic_or(uint32_t *p, uint32_t v) {
#ifdef _MSC_VER
    return InterlockedOr((LONG volatile *)p, v);
#else
    return __sync_fetch_and_or(p, v);
#endif
}

static FORCEINLINE uint32_t __atomic_xor(uint32_t *p, uint32_t v) {
#ifdef _MSC_VER
    return InterlockedXor((LONG volatile *)p, v);
#else
    return __sync_fetch_and_xor(p, v);
#endif
}

static FORCEINLINE uint32_t __atomic_min(uint32_t *p, uint32_t v) {
    int32_t old, min;
    do {
        old = *((volatile int32_t *)p);
        min = (old < (int32_t)v) ? old : (int32_t)v;
#ifdef _MSC_VER
    } while (InterlockedCompareExchange((LONG volatile *)p, min, old) != old);
#else
    } while (__sync_bool_compare_and_swap(p, old, min) == false);
#endif
    return old;
}

static FORCEINLINE uint32_t __atomic_max(uint32_t *p, uint32_t v) {
    int32_t old, max;
    do {
        old = *((volatile int32_t *)p);
        max = (old > (int32_t)v) ? old : (int32_t)v;
#ifdef _MSC_VER
    } while (InterlockedCompareExchange((LONG volatile *)p, max, old) != old);
#else
    } while (__sync_bool_compare_and_swap(p, old, max) == false);
#endif
    return old;
}

static FORCEINLINE uint32_t __atomic_umin(uint32_t *p, uint32_t v) {
    uint32_t old, min;
    do {
        old = *((volatile uint32_t *)p);
        min = (old < v) ? old : v;
#ifdef _MSC_VER
    } while (InterlockedCompareExchange((LONG volatile *)p, min, old) != old);
#else
    } while (__sync_bool_compare_and_swap(p, old, min) == false);
#endif
    return old;
}

static FORCEINLINE uint32_t __atomic_umax(uint32_t *p, uint32_t v) {
    uint32_t old, max;
    do {
        old = *((volatile uint32_t *)p);
        max = (old > v) ? old : v;
#ifdef _MSC_VER
    } while (InterlockedCompareExchange((LONG volatile *)p, max, old) != old);
#else
    } while (__sync_bool_compare_and_swap(p, old, max) == false);
#endif
    return old;
}

static FORCEINLINE uint32_t __atomic_xchg(uint32_t *p, uint32_t v) {
#ifdef _MSC_VER
    return InterlockedExchange((LONG volatile *)p, v);
#else
    return __sync_lock_test_and_set(p, v);
#endif
}

static FORCEINLINE uint32_t __atomic_cmpxchg(uint32_t *p, uint32_t cmpval,
                                             uint32_t newval) {
#ifdef _MSC_VER
    return InterlockedCompareExchange((LONG volatile *)p, newval, cmpval);
#else
    return __sync_val_compare_and_swap(p, cmpval, newval);
#endif
}

static FORCEINLINE uint64_t __atomic_add(uint64_t *p, uint64_t v) {
#ifdef _MSC_VER
    return InterlockedAdd64((LONGLONG volatile *)p, v) - v;
#else
    return __sync_fetch_and_add(p, v);
#endif
}

static FORCEINLINE uint64_t __atomic_sub(uint64_t *p, uint64_t v) {
#ifdef _MSC_VER
    return InterlockedAdd64((LONGLONG volatile *)p, -v) + v;
#else
    return __sync_fetch_and_sub(p, v);
#endif
}

static FORCEINLINE uint64_t __atomic_and(uint64_t *p, uint64_t v) {
#ifdef _MSC_VER
    return InterlockedAnd64((LONGLONG volatile *)p, v) - v;
#else
    return __sync_fetch_and_and(p, v);
#endif
}

static FORCEINLINE uint64_t __atomic_or(uint64_t *p, uint64_t v) {
#ifdef _MSC_VER
    return InterlockedOr64((LONGLONG volatile *)p, v) - v;
#else
    return __sync_fetch_and_or(p, v);
#endif
}

static FORCEINLINE uint64_t __atomic_xor(uint64_t *p, uint64_t v) {
#ifdef _MSC_VER
    return InterlockedXor64((LONGLONG volatile *)p, v) - v;
#else
    return __sync_fetch_and_xor(p, v);
#endif
}

static FORCEINLINE uint64_t __atomic_min(uint64_t *p, uint64_t v) {
    int64_t old, min;
    do {
        old = *((volatile int64_t *)p);
        min = (old < (int64_t)v) ? old : (int64_t)v;
#ifdef _MSC_VER
    } while (InterlockedCompareExchange64((LONGLONG volatile *)p, min, old) != old);
#else
    } while (__sync_bool_compare_and_swap(p, old, min) == false);
#endif
    return old;
}

static FORCEINLINE uint64_t __atomic_max(uint64_t *p, uint64_t v) {
    int64_t old, max;
    do {
        old = *((volatile int64_t *)p);
        max = (old > (int64_t)v) ? old : (int64_t)v;
#ifdef _MSC_VER
    } while (InterlockedCompareExchange64((LONGLONG volatile *)p, max, old) != old);
#else
    } while (__sync_bool_compare_and_swap(p, old, max) == false);
#endif
    return old;
}

static FORCEINLINE uint64_t __atomic_umin(uint64_t *p, uint64_t v) {
    uint64_t old, min;
    do {
        old = *((volatile uint64_t *)p);
        min = (old < v) ? old : v;
#ifdef _MSC_VER
    } while (InterlockedCompareExchange64((LONGLONG volatile *)p, min, old) != old);
#else
    } while (__sync_bool_compare_and_swap(p, old, min) == false);
#endif
    return old;
}

static FORCEINLINE uint64_t __atomic_umax(uint64_t *p, uint64_t v) {
    uint64_t old, max;
    do {
        old = *((volatile uint64_t *)p);
        max = (old > v) ? old : v;
#ifdef _MSC_VER
    } while (InterlockedCompareExchange64((LONGLONG volatile *)p, max, old) != old);
#else
    } while (__sync_bool_compare_and_swap(p, old, max) == false);
#endif
    return old;
}

static FORCEINLINE uint64_t __atomic_xchg(uint64_t *p, uint64_t v) {
#ifdef _MSC_VER
    return InterlockedExchange64((LONGLONG volatile *)p, v);
#else
    return __sync_lock_test_and_set(p, v);
#endif
}

static FORCEINLINE uint64_t __atomic_cmpxchg(uint64_t *p, uint64_t cmpval,
                                             uint64_t newval) {
#ifdef _MSC_VER
    return InterlockedCompareExchange64((LONGLONG volatile *)p, newval, cmpval);
#else
    return __sync_val_compare_and_swap(p, cmpval, newval);
#endif
}

#ifdef WIN32
#include <windows.h>
#define __clock __rdtsc
#else // WIN32
static FORCEINLINE uint64_t __clock() {
  uint32_t low, high;
#ifdef __x86_64
  __asm__ __volatile__ ("xorl %%eax,%%eax \n    cpuid"
                        ::: "%rax", "%rbx", "%rcx", "%rdx" );
#else
  __asm__ __volatile__ ("xorl %%eax,%%eax \n    cpuid"
                        ::: "%eax", "%ebx", "%ecx", "%edx" );
#endif
  __asm__ __volatile__ ("rdtsc" : "=a" (low), "=d" (high));
  return (uint64_t)high << 32 | low;
}

#endif // !WIN32

#undef FORCEINLINE
#undef PRE_ALIGN
#undef POST_ALIGN
