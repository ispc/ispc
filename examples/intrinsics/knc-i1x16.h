/**
  Copyright (c) 2010-2014, Intel Corporation
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

#include <limits.h> // INT_MIN
#include <stdint.h> 
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <immintrin.h>
#include <zmmintrin.h>

#include <iostream> // for operator<<(m512[i])
#include <iomanip>  // for operator<<(m512[i])

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

typedef float   __vec1_f;
typedef double  __vec1_d;
typedef int8_t  __vec1_i8;
typedef int16_t __vec1_i16;
typedef int32_t __vec1_i32;
typedef int64_t __vec1_i64;

/************ mask **************/

struct __vec16_i1 
{
  __mmask16 v;

  FORCEINLINE __vec16_i1() { }
  FORCEINLINE __vec16_i1(const __mmask16 &vv) : v(vv) { }
  FORCEINLINE __vec16_i1(bool  v0, bool  v1, bool  v2, bool  v3,
                         bool  v4, bool  v5, bool  v6, bool  v7,
                         bool  v8, bool  v9, bool v10, bool v11,
                         bool v12, bool v13, bool v14, bool v15) {
    v = ((v0 & 1) |
        ((v1 & 1) << 1) |
        ((v2 & 1) << 2) |
        ((v3 & 1) << 3) |
        ((v4 & 1) << 4) |
        ((v5 & 1) << 5) |
        ((v6 & 1) << 6) |
        ((v7 & 1) << 7) |
        ((v8 & 1) << 8) |
        ((v9 & 1) << 9) |
        ((v10 & 1) << 10) |
        ((v11 & 1) << 11) |
        ((v12 & 1) << 12) |
        ((v13 & 1) << 13) |
        ((v14 & 1) << 14) |
        ((v15 & 1) << 15));
  }
  FORCEINLINE       uint8_t operator[](const int i) const {  return ((v >> i) & 1); }
  FORCEINLINE       uint8_t operator[](const int i)       {  return ((v >> i) & 1); }
  FORCEINLINE operator __mmask16() const { return v; }
};

/************ vector **************/

struct PRE_ALIGN(64) __vec16_i32 
{
  __m512i v;
  FORCEINLINE operator __m512i() const { return v; }
  FORCEINLINE __vec16_i32() : v(_mm512_undefined_epi32()) {}
  FORCEINLINE __vec16_i32(const int32_t &in) : v(_mm512_set1_epi32(in)) {}
  FORCEINLINE __vec16_i32(const __m512i &in) : v(in) {}
  FORCEINLINE __vec16_i32(const __vec16_i32 &o) : v(o.v) {}
  FORCEINLINE __vec16_i32& operator =(const __vec16_i32 &o) { v=o.v; return *this; }
  FORCEINLINE __vec16_i32(int32_t v00, int32_t v01, int32_t v02, int32_t v03, 
      int32_t v04, int32_t v05, int32_t v06, int32_t v07,
      int32_t v08, int32_t v09, int32_t v10, int32_t v11,
      int32_t v12, int32_t v13, int32_t v14, int32_t v15) :
    v ( _mm512_set_16to16_pi(v15, v14, v13, v12, v11, v10, v09, v08, v07, v06, v05, v04, v03, v02, v01, v00) ) {}
    FORCEINLINE const int32_t& operator[](const int i) const {  return ((int32_t*)this)[i]; }
    FORCEINLINE       int32_t& operator[](const int i)       {  return ((int32_t*)this)[i]; }
} POST_ALIGN(64);

PRE_ALIGN(64) struct __vec16_f 
{
  __m512 v;
  FORCEINLINE operator __m512() const { return v; }
  FORCEINLINE __vec16_f() : v(_mm512_undefined_ps()) { }
  FORCEINLINE __vec16_f(const __m512 &in) : v(in) {}
  FORCEINLINE __vec16_f(const __vec16_f &o) : v(o.v) {}
  FORCEINLINE __vec16_f& operator =(const __vec16_f &o) { v=o.v; return *this; }
  FORCEINLINE __vec16_f(float v00, float v01, float v02, float v03, 
      float v04, float v05, float v06, float v07,
      float v08, float v09, float v10, float v11,
      float v12, float v13, float v14, float v15) :
    v ( _mm512_set_16to16_ps(v15, v14, v13, v12, v11, v10, v09, v08, v07, v06, v05, v04, v03, v02, v01, v00) )  {}
  FORCEINLINE const float& operator[](const int i) const {  return ((float*)this)[i]; }
  FORCEINLINE       float& operator[](const int i)       {  return ((float*)this)[i]; }
} POST_ALIGN(64);

static void zmm2hilo(const __m512i v1, const __m512i v2, __m512i &_hi, __m512i &_lo)
{
  _hi = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xFF00, 
      _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
      v2);
  _hi = _mm512_mask_permutevar_epi32(_hi, 0x00FF, 
      _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
      v1);
  _lo = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xFF00,
      _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
      v2);
  _lo = _mm512_mask_permutevar_epi32(_lo, 0x00FF,
      _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
      v1);
}
static void hilo2zmm(const __m512i v_hi, const __m512i v_lo, __m512i &_v1, __m512i &_v2)
{
  _v2 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
      _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
      v_hi);
  _v2 = _mm512_mask_permutevar_epi32(_v2, 0x5555,
      _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
      v_lo);
  _v1 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
      _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
      v_hi);
  _v1 = _mm512_mask_permutevar_epi32(_v1, 0x5555,
      _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
      v_lo);
}

struct PRE_ALIGN(128) __vec16_d 
{
  union {
    __m512d v1;
    __m512d v_hi;
  };
  union {
    __m512d v2;
    __m512d v_lo;
  };
  FORCEINLINE __vec16_d() : v1(_mm512_undefined_pd()), v2(_mm512_undefined_pd()) {}
  FORCEINLINE __vec16_d(const __m512d _v1, const __m512d _v2) : v1(_v1), v2(_v2) {}
  FORCEINLINE __vec16_d(const __vec16_d &o) : v1(o.v1), v2(o.v2) {}
  FORCEINLINE __vec16_d& operator =(const __vec16_d &o) { v1=o.v1; v2=o.v2; return *this; }
  FORCEINLINE __vec16_d(double v00, double v01, double v02, double v03, 
      double v04, double v05, double v06, double v07,
      double v08, double v09, double v10, double v11,
      double v12, double v13, double v14, double v15) {
    v1 = _mm512_set_8to8_pd(v15, v14, v13, v12, v11, v10, v09, v08);
    v2 = _mm512_set_8to8_pd(v07, v06, v05, v04, v03, v02, v01, v00);
  }
  FORCEINLINE const double& operator[](const int i) const {  return ((double*)this)[i]; }
  FORCEINLINE       double& operator[](const int i)       {  return ((double*)this)[i]; }
  FORCEINLINE __vec16_d cvt2hilo()  const
  {
    const __m512i _v1 = _mm512_castpd_si512(v1);
    const __m512i _v2 = _mm512_castpd_si512(v2);
    __m512i _hi, _lo;
    zmm2hilo(_v1, _v2, _hi, _lo);
    return __vec16_d(_mm512_castsi512_pd(_hi), _mm512_castsi512_pd(_lo));
  }
  FORCEINLINE __vec16_d cvt2zmm() const
  {
    const __m512i _hi = _mm512_castpd_si512(v_hi);
    const __m512i _lo = _mm512_castpd_si512(v_lo);
    __m512i _v1, _v2;
    hilo2zmm(_hi,_lo, _v1,_v2);
    return __vec16_d(_mm512_castsi512_pd(_v1), _mm512_castsi512_pd(_v2));
  }
} POST_ALIGN(128);

struct PRE_ALIGN(128) __vec16_i64 
{
  union {
    __m512i v1;
    __m512i v_hi;
  };
  union
  {
    __m512i v2;
    __m512i v_lo;
  };
  FORCEINLINE __vec16_i64() : v1(_mm512_undefined_epi32()), v2(_mm512_undefined_epi32()) {}
  FORCEINLINE __vec16_i64(const __m512i _v1, const __m512i _v2) : v1(_v1), v2(_v2) {}
  FORCEINLINE __vec16_i64(const __vec16_i64 &o) : v1(o.v1), v2(o.v2) {}
  FORCEINLINE __vec16_i64& operator =(const __vec16_i64 &o) { v1=o.v1; v2=o.v2; return *this; }
  FORCEINLINE __vec16_i64(int64_t v00, int64_t v01, int64_t v02, int64_t v03, 
      int64_t v04, int64_t v05, int64_t v06, int64_t v07,
      int64_t v08, int64_t v09, int64_t v10, int64_t v11,
      int64_t v12, int64_t v13, int64_t v14, int64_t v15) {
    v2 = _mm512_set_8to8_epi64(v15, v14, v13, v12, v11, v10, v09, v08);
    v1 = _mm512_set_8to8_epi64(v07, v06, v05, v04, v03, v02, v01, v00);
  }
  FORCEINLINE const int64_t& operator[](const int i) const {  return ((int64_t*)this)[i]; }
  FORCEINLINE       int64_t& operator[](const int i)       {  return ((int64_t*)this)[i]; }
  FORCEINLINE __vec16_i64 cvt2hilo()  const
  {
    __vec16_i64 ret;
    zmm2hilo(v1,v2,ret.v_hi,ret.v_lo);
    return ret;
  }
  FORCEINLINE __vec16_i64 cvt2zmm() const
  {
    __vec16_i64 ret;
    hilo2zmm(v_hi,v_lo, ret.v1, ret.v2);
    return ret;
  }
} POST_ALIGN(128);

/************ scalar **************/

template <typename T>
struct vec16 
{
  FORCEINLINE vec16() { }
  FORCEINLINE vec16(T v0, T v1, T  v2, T  v3, T  v4, T  v5, T  v6, T  v7,
                    T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) {
    data[0] = v0;        data[1] = v1;        data[2] = v2;        data[3] = v3;
    data[4] = v4;        data[5] = v5;        data[6] = v6;        data[7] = v7;
    data[8] = v8;        data[9] = v9;        data[10] = v10;      data[11] = v11;
    data[12] = v12;      data[13] = v13;      data[14] = v14;      data[15] = v15;
  }
  T data[16]; 
  FORCEINLINE const T& operator[](const int i) const { return data[i]; }
  FORCEINLINE       T& operator[](const int i)       { return data[i]; }
};

PRE_ALIGN(16) struct __vec16_i8   : public vec16<int8_t> { 
    __vec16_i8() { }
    __vec16_i8(int8_t v0, int8_t v1, int8_t v2, int8_t v3, 
               int8_t v4, int8_t v5, int8_t v6, int8_t v7,
               int8_t v8, int8_t v9, int8_t v10, int8_t v11, 
               int8_t v12, int8_t v13, int8_t v14, int8_t v15) 
        : vec16<int8_t>(v0, v1, v2, v3, v4, v5, v6, v7,
                        v8, v9, v10, v11, v12, v13, v14, v15) { }
} POST_ALIGN(16);

PRE_ALIGN(32) struct __vec16_i16  : public vec16<int16_t> { 
    __vec16_i16() { }
    __vec16_i16(int16_t v0, int16_t v1, int16_t v2, int16_t v3, 
                int16_t v4, int16_t v5, int16_t v6, int16_t v7,
                int16_t v8, int16_t v9, int16_t v10, int16_t v11, 
                int16_t v12, int16_t v13, int16_t v14, int16_t v15) 
        : vec16<int16_t>(v0, v1, v2, v3, v4, v5, v6, v7,
                         v8, v9, v10, v11, v12, v13, v14, v15) { }
} POST_ALIGN(32);

static inline int32_t __extract_element(__vec16_i32, int);


///////////////////////////////////////////////////////////////////////////
// debugging helpers
//
inline std::ostream &operator<<(std::ostream &out, const __m512i &v)
{
  out << "[";
  for (int i=0;i<16;i++)  
    out << (i!=0?",":"") << std::dec << std::setw(8) << ((int*)&v)[i] << std::dec;
  out << "]" << std::flush;
  return out;
}

inline std::ostream &operator<<(std::ostream &out, const __m512 &v)
{
  out << "[";
  for (int i=0;i<16;i++)  
    out << (i!=0?",":"") << ((float*)&v)[i];
  out << "]" << std::flush;
  return out;
}

inline std::ostream &operator<<(std::ostream &out, const __vec16_i1 &v)
{
  out << "[";
  for (int i=0;i<16;i++)  
    out << (i!=0?",":"") << std::dec << std::setw(8) << (int)v[i] << std::dec;
  out << "]" << std::flush;
  return out;
}

inline std::ostream &operator<<(std::ostream &out, const __vec16_i8 &v)
{
  out << "[";
  for (int i=0;i<16;i++)  
    out << (i!=0?",":"") << std::dec << std::setw(8) << (int)((unsigned char*)&v)[i] << std::dec;
  out << "]" << std::flush;
  return out;
}

inline std::ostream &operator<<(std::ostream &out, const __vec16_i16 &v)
{
  out << "[";
  for (int i=0;i<16;i++)  
    out << (i!=0?",":"") << std::dec << std::setw(8) << (int)((uint16_t*)&v)[i] << std::dec;
  out << "]" << std::flush;
  return out;
}

inline std::ostream &operator<<(std::ostream &out, const __vec16_d &v)
{
  out << "[";
  for (int i=0;i<16;i++) {
    out << (i!=0?",":"") << (v[i]);
  }  
  out << "]" << std::flush;
  return out;
}

inline std::ostream &operator<<(std::ostream &out, const __vec16_i64 &v)
{
  out << "[";
  for (int i=0;i<16;i++) {
    out << (i!=0?",":"") << (v[i]);
  }  
  out << "]" << std::flush;
  return out;
}


///////////////////////////////////////////////////////////////////////////
// macros...

/* knc::macro::used */
#define BINARY_OP(TYPE, NAME, OP)                               \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                  \
    TYPE ret;                                                   \
   for (int i = 0; i < 16; ++i)                                 \
       ret[i] = a[i] OP b[i];                             \
   return ret;                                                   \
}

/* knc::macro::used */
#define BINARY_OP_CAST(TYPE, CAST, NAME, OP)                        \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret[i] = (CAST)(a[i]) OP (CAST)(b[i]);                 \
   return ret;                                                      \
}

/* knc::macro::used */
#define BINARY_OP_FUNC(TYPE, NAME, FUNC)                            \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret[i] = FUNC(a[i], b[i]);                             \
   return ret;                                                      \
}

/* knc::macro::used */
#define CMP_OP(TYPE, SUFFIX, CAST, NAME, OP)                        \
static FORCEINLINE __vec16_i1 NAME##_##SUFFIX(TYPE a, TYPE b) {     \
   __vec16_i1 ret;                                                  \
   ret.v = 0;                                                       \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v |= ((CAST)(a[i]) OP (CAST)(b[i])) << i;            \
   return ret;                                                      \
}                                                                   \
static FORCEINLINE __vec16_i1 NAME##_##SUFFIX##_and_mask(TYPE a, TYPE b,       \
                                              __vec16_i1 mask) {    \
   __vec16_i1 ret;                                                  \
   ret.v = 0;                                                       \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v |= ((CAST)(a[i]) OP (CAST)(b[i])) << i;            \
   ret.v &= mask.v;                                                 \
   return ret;                                                      \
}

/* knc::macro::used */
#define INSERT_EXTRACT(VTYPE, STYPE)                                  \
static FORCEINLINE STYPE __extract_element(VTYPE v, int index) {      \
    return ((STYPE *)&v)[index];                                      \
}                                                                     \
static FORCEINLINE void __insert_element(VTYPE *v, int index, STYPE val) { \
    ((STYPE *)v)[index] = val;                                        \
}

/* knc::macro::used */
#define LOAD_STORE(VTYPE, STYPE)                       \
template <int ALIGN>                                   \
static FORCEINLINE VTYPE __load(const VTYPE *p) {      \
    STYPE *ptr = (STYPE *)p;                           \
    VTYPE ret;                                         \
    for (int i = 0; i < 16; ++i)                       \
        ret[i] = ptr[i];                             \
    return ret;                                        \
}                                                      \
template <int ALIGN>                                   \
static FORCEINLINE void __store(VTYPE *p, VTYPE v) {   \
    STYPE *ptr = (STYPE *)p;                           \
    for (int i = 0; i < 16; ++i)                       \
        ptr[i] = v[i];                               \
}

/* knc::macro::used */
#define REDUCE_ADD(TYPE, VTYPE, NAME)           \
static FORCEINLINE TYPE NAME(VTYPE v) {         \
     TYPE ret = v[0];                         \
     for (int i = 1; i < 16; ++i)               \
         ret = ret + v[i];                    \
     return ret;                                \
}

/* knc::macro::used */
#define REDUCE_MINMAX(TYPE, VTYPE, NAME, OP)                    \
static FORCEINLINE TYPE NAME(VTYPE v) {                         \
    TYPE ret = v[0];                                          \
    for (int i = 1; i < 16; ++i)                                \
        ret = (ret OP (TYPE)v[i]) ? ret : (TYPE)v[i];       \
    return ret;                                                 \
}

/* knc::macro::used */
#define SELECT(TYPE)                                                \
static FORCEINLINE TYPE __select(__vec16_i1 mask, TYPE a, TYPE b) { \
    TYPE ret;                                                       \
    for (int i = 0; i < 16; ++i)                                    \
        ret[i] = (mask.v & (1<<i)) ? a[i] : b[i];             \
    return ret;                                                     \
}                                                                   \
static FORCEINLINE TYPE __select(bool cond, TYPE a, TYPE b) {       \
    return cond ? a : b;                                            \
}

/* knc::macro::used */
#define SHIFT_UNIFORM(TYPE, CAST, NAME, OP)                         \
static FORCEINLINE TYPE NAME(TYPE a, int32_t b) {                   \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret[i] = (CAST)(a[i]) OP b;                              \
   return ret;                                                      \
}

/* knc::macro::used */
#define SMEAR(VTYPE, NAME, STYPE)                                  \
template <class RetVecType> VTYPE __smear_##NAME(STYPE);           \
template <> FORCEINLINE VTYPE __smear_##NAME<VTYPE>(STYPE v) {     \
    VTYPE ret;                                                     \
    for (int i = 0; i < 16; ++i)                                   \
        ret[i] = v;                                              \
    return ret;                                                    \
}

/* knc::macro::used */
#define SETZERO(VTYPE, NAME)                                       \
template <class RetVecType> VTYPE __setzero_##NAME();              \
template <> FORCEINLINE VTYPE __setzero_##NAME<VTYPE>() {          \
    VTYPE ret;                                                     \
    for (int i = 0; i < 16; ++i)                                   \
        ret[i] = 0;                                              \
    return ret;                                                    \
}

/* knc::macro::used */
#define UNDEF(VTYPE, NAME)                                         \
template <class RetVecType> VTYPE __undef_##NAME();                \
template <> FORCEINLINE VTYPE __undef_##NAME<VTYPE>() {            \
    return VTYPE();                                                \
}

/* knc::macro::used */
#define BROADCAST(VTYPE, NAME, STYPE)                 \
static FORCEINLINE VTYPE __broadcast_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i)                      \
        ret[i] = v[index & 0xf];                  \
    return ret;                                       \
}                                                     \

/* knc::macro::used */
#define ROTATE(VTYPE, NAME, STYPE)                    \
static FORCEINLINE VTYPE __rotate_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i)                      \
        ret[i] = v[(i+index) & 0xf];              \
    return ret;                                       \
}                                                     \

#define SHIFT(VTYPE, NAME, STYPE)                    \
static FORCEINLINE VTYPE __shift_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i) {                    \
      int modIndex = i+index;                         \
      STYPE val = ((modIndex >= 0) && (modIndex < 16)) ? v[modIndex] : 0; \
      ret[i] = val;                                 \
    }                                                 \
    return ret;                                       \
}                                                     \

/* knc::macro::used */
#define SHUFFLES(VTYPE, NAME, STYPE)                 \
static FORCEINLINE VTYPE __shuffle_##NAME(VTYPE v, __vec16_i32 index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i)                      \
        ret[i] = v[__extract_element(index, i) & 0xf];      \
    return ret;                                       \
}                                                     \
static FORCEINLINE VTYPE __shuffle2_##NAME(VTYPE v0, VTYPE v1, __vec16_i32 index) {     \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i) {                    \
        int ii = __extract_element(index, i) & 0x1f;    \
        ret[i] = (ii < 16) ? v0[ii] : v1[ii-16];  \
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
// mask 
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_i1 __movmsk(__vec16_i1 mask) { return  _mm512_kmov    (mask);       }
static FORCEINLINE       bool __any   (__vec16_i1 mask) { return !_mm512_kortestz(mask, mask); }
static FORCEINLINE       bool __all   (__vec16_i1 mask) { return  _mm512_kortestc(mask, mask); }
static FORCEINLINE       bool __none  (__vec16_i1 mask) { return  _mm512_kortestz(mask, mask); }
static FORCEINLINE __vec16_i1 __not   (__vec16_i1 mask) { return  _mm512_knot    (mask);       }

static FORCEINLINE __vec16_i1 __equal_i1(__vec16_i1 a, __vec16_i1 b) { return _mm512_kxnor (a,b); }
static FORCEINLINE __vec16_i1 __and     (__vec16_i1 a, __vec16_i1 b) { return _mm512_kand  (a,b); }
static FORCEINLINE __vec16_i1 __xor     (__vec16_i1 a, __vec16_i1 b) { return _mm512_kxor  (a,b); }
static FORCEINLINE __vec16_i1 __or      (__vec16_i1 a, __vec16_i1 b) { return _mm512_kor   (a,b); }
static FORCEINLINE __vec16_i1 __and_not1(__vec16_i1 a, __vec16_i1 b) { return _mm512_kandn (a,b); }
static FORCEINLINE __vec16_i1 __and_not2(__vec16_i1 a, __vec16_i1 b) { return _mm512_kandnr(a,b); }

static FORCEINLINE __vec16_i1 __select(__vec16_i1 mask, __vec16_i1 a, __vec16_i1 b) { return __or(__and(a, mask), __and_not2(b, mask)); }
static FORCEINLINE __vec16_i1 __select(      bool cond, __vec16_i1 a, __vec16_i1 b) { return cond ? a : b; }

static FORCEINLINE bool __extract_element(__vec16_i1 vec, int index) { return (vec.v & (1 << index)) ? true : false; }
static FORCEINLINE void __insert_element(__vec16_i1 *vec, int index, bool val) 
{
  if (val == false)  vec->v &= ~(1 << index);
  else               vec->v |=  (1 << index);
}

template <int ALIGN> static FORCEINLINE __vec16_i1 __load(const __vec16_i1 *p) 
{
  return *p;
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_i1 *p, __vec16_i1 v) 
{
  *p = v;
}

template <class RetVecType> static RetVecType __smear_i1(int i);
template <> FORCEINLINE __vec16_i1 __smear_i1<__vec16_i1>(int i) { return i?0xFFFF:0x0; }

template <class RetVecType> static RetVecType __setzero_i1();
template <> FORCEINLINE __vec16_i1 __setzero_i1<__vec16_i1>() { return 0; }

template <class RetVecType> __vec16_i1 __undef_i1();
template <> FORCEINLINE __vec16_i1 __undef_i1<__vec16_i1>() { return __vec16_i1(); }

///////////////////////////////////////////////////////////////////////////
// int8
///////////////////////////////////////////////////////////////////////////

BINARY_OP(__vec16_i8, __add, +)
BINARY_OP(__vec16_i8, __sub, -)
BINARY_OP(__vec16_i8, __mul, *)

BINARY_OP(__vec16_i8, __or, |)
BINARY_OP(__vec16_i8, __and, &)
BINARY_OP(__vec16_i8, __xor, ^)
BINARY_OP(__vec16_i8, __shl, <<)

BINARY_OP_CAST(__vec16_i8, uint8_t, __udiv, /)
BINARY_OP_CAST(__vec16_i8, int8_t,  __sdiv, /)

BINARY_OP_CAST(__vec16_i8, uint8_t, __urem, %)
BINARY_OP_CAST(__vec16_i8, int8_t,  __srem, %)
BINARY_OP_CAST(__vec16_i8, uint8_t, __lshr, >>)
BINARY_OP_CAST(__vec16_i8, int8_t,  __ashr, >>)

SHIFT_UNIFORM(__vec16_i8, uint8_t, __lshr, >>)
SHIFT_UNIFORM(__vec16_i8, int8_t, __ashr, >>)
SHIFT_UNIFORM(__vec16_i8, int8_t, __shl, <<)

CMP_OP(__vec16_i8, i8, int8_t,  __equal, ==)
CMP_OP(__vec16_i8, i8, int8_t,  __not_equal, !=)
CMP_OP(__vec16_i8, i8, uint8_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i8, i8, int8_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i8, i8, uint8_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i8, i8, int8_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i8, i8, uint8_t, __unsigned_less_than, <)
CMP_OP(__vec16_i8, i8, int8_t,  __signed_less_than, <)
CMP_OP(__vec16_i8, i8, uint8_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i8, i8, int8_t,  __signed_greater_than, >)

SELECT(__vec16_i8)
INSERT_EXTRACT(__vec16_i8, int8_t)
SMEAR(__vec16_i8, i8, int8_t)
SETZERO(__vec16_i8, i8)
UNDEF(__vec16_i8, i8)
BROADCAST(__vec16_i8, i8, int8_t)
ROTATE(__vec16_i8, i8, int8_t)
SHIFT(__vec16_i8, i8, int8_t)
SHUFFLES(__vec16_i8, i8, int8_t)
LOAD_STORE(__vec16_i8, int8_t)

///////////////////////////////////////////////////////////////////////////
// int16
///////////////////////////////////////////////////////////////////////////

BINARY_OP(__vec16_i16, __add, +)
BINARY_OP(__vec16_i16, __sub, -)
BINARY_OP(__vec16_i16, __mul, *)

BINARY_OP(__vec16_i16, __or, |)
BINARY_OP(__vec16_i16, __and, &)
BINARY_OP(__vec16_i16, __xor, ^)
BINARY_OP(__vec16_i16, __shl, <<)

BINARY_OP_CAST(__vec16_i16, uint16_t, __udiv, /)
BINARY_OP_CAST(__vec16_i16, int16_t,  __sdiv, /)

BINARY_OP_CAST(__vec16_i16, uint16_t, __urem, %)
BINARY_OP_CAST(__vec16_i16, int16_t,  __srem, %)
BINARY_OP_CAST(__vec16_i16, uint16_t, __lshr, >>)
BINARY_OP_CAST(__vec16_i16, int16_t,  __ashr, >>)

SHIFT_UNIFORM(__vec16_i16, uint16_t, __lshr, >>)
SHIFT_UNIFORM(__vec16_i16, int16_t, __ashr, >>)
SHIFT_UNIFORM(__vec16_i16, int16_t, __shl, <<)

CMP_OP(__vec16_i16, i16, int16_t,  __equal, ==)
CMP_OP(__vec16_i16, i16, int16_t,  __not_equal, !=)
CMP_OP(__vec16_i16, i16, uint16_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i16, i16, int16_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i16, i16, uint16_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i16, i16, int16_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i16, i16, uint16_t, __unsigned_less_than, <)
CMP_OP(__vec16_i16, i16, int16_t,  __signed_less_than, <)
CMP_OP(__vec16_i16, i16, uint16_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i16, i16, int16_t,  __signed_greater_than, >)

SELECT(__vec16_i16)
INSERT_EXTRACT(__vec16_i16, int16_t)
SMEAR(__vec16_i16, i16, int16_t)
SETZERO(__vec16_i16, i16)
UNDEF(__vec16_i16, i16)
BROADCAST(__vec16_i16, i16, int16_t)
ROTATE(__vec16_i16, i16, int16_t)
SHIFT(__vec16_i16, i16, int16_t)
SHUFFLES(__vec16_i16, i16, int16_t)
LOAD_STORE(__vec16_i16, int16_t)

///////////////////////////////////////////////////////////////////////////
// int32
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_i32 __add (__vec16_i32 a, __vec16_i32 b) { return _mm512_add_epi32  (a,b); }
static FORCEINLINE __vec16_i32 __sub (__vec16_i32 a, __vec16_i32 b) { return _mm512_sub_epi32  (a,b); }
static FORCEINLINE __vec16_i32 __mul (__vec16_i32 a, __vec16_i32 b) { return _mm512_mullo_epi32(a,b); }
static FORCEINLINE __vec16_i32 __udiv(__vec16_i32 a, __vec16_i32 b) { return _mm512_div_epu32  (a,b); }
static FORCEINLINE __vec16_i32 __sdiv(__vec16_i32 a, __vec16_i32 b) { return _mm512_div_epi32  (a,b); }
static FORCEINLINE __vec16_i32 __urem(__vec16_i32 a, __vec16_i32 b) { return _mm512_rem_epu32  (a,b); }
static FORCEINLINE __vec16_i32 __srem(__vec16_i32 a, __vec16_i32 b) { return _mm512_rem_epi32  (a,b); }
static FORCEINLINE __vec16_i32 __or  (__vec16_i32 a, __vec16_i32 b) { return _mm512_or_epi32   (a,b); }
static FORCEINLINE __vec16_i32 __and (__vec16_i32 a, __vec16_i32 b) { return _mm512_and_epi32  (a,b); }
static FORCEINLINE __vec16_i32 __xor (__vec16_i32 a, __vec16_i32 b) { return _mm512_xor_epi32  (a,b); }
static FORCEINLINE __vec16_i32 __shl (__vec16_i32 a, __vec16_i32 b) { return _mm512_sllv_epi32 (a,b); }
static FORCEINLINE __vec16_i32 __lshr(__vec16_i32 a, __vec16_i32 b) { return _mm512_srlv_epi32 (a,b); }
static FORCEINLINE __vec16_i32 __ashr(__vec16_i32 a, __vec16_i32 b) { return _mm512_srav_epi32 (a,b); }
static FORCEINLINE __vec16_i32 __shl (__vec16_i32 a,     int32_t n) { return _mm512_slli_epi32 (a,n); }
static FORCEINLINE __vec16_i32 __lshr(__vec16_i32 a,     int32_t n) { return _mm512_srli_epi32 (a,n); }
static FORCEINLINE __vec16_i32 __ashr(__vec16_i32 a,     int32_t n) { return _mm512_srai_epi32 (a,n); }

static FORCEINLINE __vec16_i1 __equal_i32                 (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmpeq_epi32_mask (a,b); }
static FORCEINLINE __vec16_i1 __not_equal_i32             (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmpneq_epi32_mask(a,b); }
static FORCEINLINE __vec16_i1 __unsigned_less_equal_i32   (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmple_epu32_mask (a,b); }
static FORCEINLINE __vec16_i1 __signed_less_equal_i32     (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmple_epi32_mask (a,b); }
static FORCEINLINE __vec16_i1 __unsigned_greater_equal_i32(__vec16_i32 a, __vec16_i32 b) { return _mm512_cmpge_epu32_mask (a,b); }
static FORCEINLINE __vec16_i1 __signed_greater_equal_i32  (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmpge_epi32_mask (a,b); }
static FORCEINLINE __vec16_i1 __unsigned_less_than_i32    (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmplt_epu32_mask (a,b); }
static FORCEINLINE __vec16_i1 __signed_less_than_i32      (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmplt_epi32_mask (a,b); }
static FORCEINLINE __vec16_i1 __unsigned_greater_than_i32 (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmpgt_epu32_mask (a,b); }
static FORCEINLINE __vec16_i1 __signed_greater_than_i32   (__vec16_i32 a, __vec16_i32 b) { return _mm512_cmpgt_epi32_mask (a,b); }

static FORCEINLINE __vec16_i1 __equal_i32_and_mask                 (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmpeq_epi32_mask (m,a,b); }
static FORCEINLINE __vec16_i1 __not_equal_i32_and_mask             (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmpneq_epi32_mask(m,a,b); }
static FORCEINLINE __vec16_i1 __unsigned_less_equal_i32_and_mask   (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmple_epu32_mask (m,a,b); }
static FORCEINLINE __vec16_i1 __signed_less_equal_i32_and_mask     (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmple_epi32_mask (m,a,b); }
static FORCEINLINE __vec16_i1 __unsigned_greater_equal_i32_and_mask(__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmpge_epu32_mask (m,a,b); }
static FORCEINLINE __vec16_i1 __signed_greater_equal_i32_and_mask  (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmpge_epi32_mask (m,a,b); }
static FORCEINLINE __vec16_i1 __unsigned_less_than_i32_and_mask    (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmplt_epu32_mask (m,a,b); }
static FORCEINLINE __vec16_i1 __signed_less_than_i32_and_mask      (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmplt_epi32_mask (m,a,b); }
static FORCEINLINE __vec16_i1 __unsigned_greater_than_i32_and_mask (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmpgt_epu32_mask (m,a,b); }
static FORCEINLINE __vec16_i1 __signed_greater_than_i32_and_mask   (__vec16_i32 a, __vec16_i32 b, __vec16_i1 m) { return _mm512_mask_cmpgt_epi32_mask (m,a,b); }

static FORCEINLINE __vec16_i32 __select(__vec16_i1 mask, __vec16_i32 a, __vec16_i32 b) { return _mm512_mask_mov_epi32(b, mask, a); }
static FORCEINLINE __vec16_i32 __select(      bool cond, __vec16_i32 a, __vec16_i32 b) { return cond ? a : b; }

static FORCEINLINE int32_t __extract_element(__vec16_i32  v,  int32_t index)              { return v[index];    }
static FORCEINLINE void    __insert_element (__vec16_i32 *v, uint32_t index, int32_t val) { (*v)[index] = val;  }

template <class RetVecType> RetVecType static __smear_i32(int32_t i);
template <> FORCEINLINE __vec16_i32 __smear_i32<__vec16_i32>(int32_t i) { return _mm512_set1_epi32(i); }

static const __vec16_i32 __ispc_one = __smear_i32<__vec16_i32>(1);
static const __vec16_i32 __ispc_zero = __smear_i32<__vec16_i32>(0);
static const __vec16_i32 __ispc_thirty_two = __smear_i32<__vec16_i32>(32);
static const __vec16_i32 __ispc_ffffffff = __smear_i32<__vec16_i32>(-1);
static const __vec16_i32 __ispc_stride1(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

template <class RetVecType> static RetVecType __setzero_i32();
template <> FORCEINLINE __vec16_i32 __setzero_i32<__vec16_i32>() { return _mm512_setzero_epi32(); }

template <class RetVecType> static RetVecType __undef_i32();
template <> FORCEINLINE __vec16_i32 __undef_i32<__vec16_i32>() { return __vec16_i32(); }

static FORCEINLINE __vec16_i32 __broadcast_i32(__vec16_i32 v, int index) { return _mm512_mask_permutevar_epi32(v, 0xFFFF, _mm512_set1_epi32(index), v); }

static FORCEINLINE __vec16_i32 __rotate_i32(__vec16_i32 v, int index) 
{
  __vec16_i32 idx = __smear_i32<__vec16_i32>(index);
  __vec16_i32 shuffle = _mm512_and_epi32(_mm512_add_epi32(__ispc_stride1, idx),  __smear_i32<__vec16_i32>(0xF));
  return _mm512_mask_permutevar_epi32(v, 0xFFFF, shuffle, v);
}

SHIFT(__vec16_i32, i32, int32_t)

static FORCEINLINE __vec16_i32 __shuffle_i32 (__vec16_i32 v, __vec16_i32 index) 
{ 
  return _mm512_mask_permutevar_epi32(v, 0xFFFF, __and(index, __smear_i32<__vec16_i32>(0xF)), v); 
}
static FORCEINLINE __vec16_i32 __shuffle2_i32(__vec16_i32 v0, __vec16_i32 v1, __vec16_i32 index)
{
  const __vec16_i1 mask = __signed_less_than_i32(index, __smear_i32<__vec16_i32>(0x10));
  index  = __and(index, __smear_i32<__vec16_i32>(0xF));
  __vec16_i32 ret = __undef_i32<__vec16_i32>();
  ret = _mm512_mask_permutevar_epi32(ret,       mask,  index, v0);
  ret = _mm512_mask_permutevar_epi32(ret, __not(mask), index, v1);
  return ret;
}

template <int ALIGN> static FORCEINLINE __vec16_i32 __load(const __vec16_i32 *p) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  // return __load<64>(p);
  return _mm512_load_epi32(p);
#else
  __vec16_i32 v;
  v = _mm512_extloadunpacklo_epi32(v,           p,    _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v = _mm512_extloadunpackhi_epi32(v, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  return v;
#endif
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_i32 *p, __vec16_i32 v) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  // __store<64>(p,v);
  _mm512_store_epi32(p, v);
#else
  _mm512_extpackstorelo_epi32(          p,    v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+64, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
#endif
}

#if 0 /* knc::fails  ./tests/foreach-25.ispc ./tests/forach-26.ispc ./tests/foreach-27.ispc */
template <> FORCEINLINE __vec16_i32 __load<64>(const __vec16_i32 *p) 
{
  return _mm512_load_epi32(p);
}
template <> FORCEINLINE void __store<64>(__vec16_i32 *p, __vec16_i32 v) 
{
  _mm512_store_epi32(p, v);
}
#endif

///////////////////////////////////////////////////////////////////////////
// int64
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_i64 __add(__vec16_i64 a, __vec16_i64 b) 
{
  return __vec16_i64(_mm512_add_epi64(a.v1, b.v1), _mm512_add_epi64(a.v2,b.v2));
}

static FORCEINLINE __vec16_i64 __sub(__vec16_i64 _a, __vec16_i64 _b) 
{
#if __ICC >= 99999 /* compiler gate, icc >= 99999 will hopefully support _mm512_sub_epi64 */
  return __vec16_i64(_mm512_sub_epi64(_a.v1, _b.v1), _mm512_sub_epi64(_a.v2,_b.v2));
#else
  const __vec16_i64 a = _a.cvt2hilo();
  const __vec16_i64 b = _b.cvt2hilo();
  __vec16_i64 ret;
  __mmask16 borrow = 0;
  ret.v_lo = _mm512_subsetb_epi32(a.v_lo, b.v_lo, &borrow);
  ret.v_hi = _mm512_sbb_epi32    (a.v_hi, borrow, b.v_hi, &borrow);
  return ret.cvt2zmm();
#endif
}

static FORCEINLINE __vec16_i64 __mul(const __vec16_i32 &a, const __vec16_i64 &_b)
{
  const __vec16_i64 b = _b.cvt2hilo();
  return __vec16_i64(_mm512_mullo_epi32(a.v,b.v_lo),
      _mm512_add_epi32(_mm512_mullo_epi32(a.v, b.v_hi),
        _mm512_mulhi_epi32(a.v, b.v_lo))).cvt2zmm();
}

static FORCEINLINE __vec16_i64 __select(__vec16_i1 mask, __vec16_i64 a, __vec16_i64 b) 
{
  __vec16_i64 ret;
  ret.v1 = _mm512_mask_mov_epi64(b.v1, mask,      a.v1);
  ret.v2 = _mm512_mask_mov_epi64(b.v2, mask >> 8, a.v2);
  return ret;
}

#if __ICC >= 1400 /* compiler gate, icc >= 14.0.0 support _mm512_mullox_epi64 */
static FORCEINLINE __vec16_i64 __mul(__vec16_i64 a, __vec16_i64 b) 
{
  return __vec16_i64(_mm512_mullox_epi64(a.v1,b.v1), _mm512_mullox_epi64(a.v2,b.v2));
}
#else  /* __ICC >= 1400 */
static FORCEINLINE void __abs_i32i64(__m512i &_hi, __m512i &_lo)
{
  /*   abs(x) : 
   * mask  = x >> 32;
   * abs(x) = (x^mask) - mask
   */ 
  const __vec16_i32 mask = __ashr(_hi, __ispc_thirty_two);
  __vec16_i32 hi = __xor(_hi, mask);
  __vec16_i32 lo = __xor(_lo, mask);
  __mmask16 borrow = 0;
  _lo = _mm512_subsetb_epi32(lo, mask, &borrow);
  _hi = _mm512_sbb_epi32    (hi, borrow, mask, &borrow);
}
static FORCEINLINE __vec16_i64 __mul(__vec16_i64 _a, __vec16_i64 _b) 
{ 
  __vec16_i64 a = _a.cvt2hilo();
  __vec16_i64 b = _b.cvt2hilo();

  /* sign = (a^b) >> 32, if sign == 0 then a*b >= 0, otherwise a*b < 0 */
  const __vec16_i1 sign = __not_equal_i32(__ashr(__xor(a.v_hi, b.v_hi), __ispc_thirty_two), __ispc_zero);
  __abs_i32i64(a.v_hi, a.v_lo);  /* abs(a) */
  __abs_i32i64(b.v_hi, b.v_lo);  /* abs(b) */
  const __vec16_i32 lo_m1 = _mm512_mullo_epi32(a.v_lo, b.v_lo);
  const __vec16_i32 hi_m1 = _mm512_mulhi_epu32(a.v_lo, b.v_lo);
  const __vec16_i32 hi_m2 = _mm512_mullo_epi32(a.v_hi, b.v_lo);
  const __vec16_i32 hi_m3 = _mm512_mullo_epi32(a.v_lo, b.v_hi);
  __mmask16 carry;
  const __vec16_i32 hi_p23 = _mm512_addsetc_epi32(hi_m2, hi_m3, &carry);
  const __vec16_i32 hi = _mm512_adc_epi32(hi_p23, carry, hi_m1, &carry);
  const __vec16_i32 lo = lo_m1;
  const __vec16_i64 ret_abs = __vec16_i64(hi,lo).cvt2zmm();
  /* if sign != 0, means either a or b is negative, then negate the result */

  return __select(sign, __sub(__vec16_i64(__ispc_zero, __ispc_zero), ret_abs), ret_abs);
}
#endif  /* __ICC >= 1400 */


static FORCEINLINE __vec16_i64 __or (__vec16_i64 a, __vec16_i64 b) { return __vec16_i64(_mm512_or_epi64 (a.v1, b.v1), _mm512_or_epi64 (a.v2, b.v2)); }
static FORCEINLINE __vec16_i64 __and(__vec16_i64 a, __vec16_i64 b) { return __vec16_i64(_mm512_and_epi64(a.v1, b.v1), _mm512_and_epi64(a.v2, b.v2)); }
static FORCEINLINE __vec16_i64 __xor(__vec16_i64 a, __vec16_i64 b) { return __vec16_i64(_mm512_xor_epi64(a.v1, b.v1), _mm512_xor_epi64(a.v2, b.v2)); }

static FORCEINLINE __vec16_i64 __udiv(__vec16_i64 a, __vec16_i64 b) { return __vec16_i64(_mm512_div_epu64(a.v1,b.v1), _mm512_div_epu64(a.v2,b.v2)); }
static FORCEINLINE __vec16_i64 __sdiv(__vec16_i64 a, __vec16_i64 b) { return __vec16_i64(_mm512_div_epi64(a.v1,b.v1), _mm512_div_epi64(a.v2,b.v2)); }

static FORCEINLINE __vec16_i64 __urem(__vec16_i64 a, __vec16_i64 b) { return __vec16_i64(_mm512_rem_epu64(a.v1,b.v1), _mm512_rem_epu64(a.v2,b.v2)); }
static FORCEINLINE __vec16_i64 __srem(__vec16_i64 a, __vec16_i64 b) { return __vec16_i64(_mm512_rem_epi64(a.v1,b.v1), _mm512_rem_epi64(a.v2,b.v2)); }


static FORCEINLINE __vec16_i64 __shl(__vec16_i64 _a, __vec16_i64 _b) 
{
  const __vec16_i64 a = _a.cvt2hilo();
  const __vec16_i64 b = _b.cvt2hilo();
  /* this is a safety gate in case b-shift >= 32 */
  const __vec16_i32 xfer = __select(
      __signed_less_than_i32(b.v_lo, __ispc_thirty_two), 
      __lshr(a.v_lo,   __sub(__ispc_thirty_two, b.v_lo)),
      __shl (a.v_lo,   __sub(b.v_lo, __ispc_thirty_two))
      );
  const __vec16_i32   hi = __or(__shl(a.v_hi, b.v_lo), xfer);
  const __vec16_i32   lo =      __shl(a.v_lo, b.v_lo);
  return __vec16_i64(hi,lo).cvt2zmm();
}
static FORCEINLINE __vec16_i64 __lshr(__vec16_i64 _a, __vec16_i64 _b) 
{
  const __vec16_i64 a = _a.cvt2hilo();
  const __vec16_i64 b = _b.cvt2hilo();
  /* this is a safety gate in case b-shift >= 32 */
  const __vec16_i32 xfer = __select(
      __signed_less_than_i32(b.v_lo, __ispc_thirty_two), 
      __shl (a.v_hi,   __sub(__ispc_thirty_two, b.v_lo)),
      __lshr(a.v_hi,   __sub(b.v_lo, __ispc_thirty_two))
      );
  const __vec16_i32   lo = __or(__lshr(a.v_lo, b.v_lo), xfer);
  const __vec16_i32   hi =      __lshr(a.v_hi, b.v_lo);
  return __vec16_i64(hi,lo).cvt2zmm();
}
static FORCEINLINE __vec16_i64 __ashr(__vec16_i64 _a, __vec16_i64 _b) 
{
  const __vec16_i64 a = _a.cvt2hilo();
  const __vec16_i64 b = _b.cvt2hilo();
  /* this is a safety gate in case b-shift >= 32 */
  const __vec16_i32 xfer = __select(
      __signed_less_than_i32(b.v_lo, __ispc_thirty_two), 
      __shl (a.v_hi,   __sub(__ispc_thirty_two, b.v_lo)),
      __ashr(a.v_hi,   __sub(b.v_lo, __ispc_thirty_two))
      );
  const __vec16_i32   lo = __or(__lshr(a.v_lo, b.v_lo), xfer);
  const __vec16_i32   hi =      __ashr(a.v_hi, b.v_lo);
  return __vec16_i64(hi,lo).cvt2zmm();
}

template <class RetVecType> RetVecType __smear_i64(const int64_t &l);
template <> FORCEINLINE  __vec16_i64 __smear_i64<__vec16_i64>(const int64_t &l) { return __vec16_i64(_mm512_set1_epi64(l), _mm512_set1_epi64(l)); }

template <class RetVecType> RetVecType __setzero_i64();
template <> FORCEINLINE  __vec16_i64 __setzero_i64<__vec16_i64>() { return __vec16_i64(_mm512_setzero_epi32(), _mm512_setzero_epi32()); }

template <class RetVecType> RetVecType __undef_i64();
template <> FORCEINLINE  __vec16_i64 __undef_i64<__vec16_i64>() { return __vec16_i64(_mm512_undefined_epi32(), _mm512_undefined_epi32()); }

static FORCEINLINE __vec16_i64 __lshr(__vec16_i64 a, uint64_t shift) { return __lshr(a, __smear_i64<__vec16_i64>(shift)); }
static FORCEINLINE __vec16_i64 __ashr(__vec16_i64 a,  int64_t shift) { return __ashr(a, __smear_i64<__vec16_i64>(shift)); }
static FORCEINLINE __vec16_i64 __shl (__vec16_i64 a,  int64_t shift) { return __shl (a, __smear_i64<__vec16_i64>(shift)); }

static FORCEINLINE __vec16_i1 __equal_i64(__vec16_i64 _a, __vec16_i64 _b) 
{
  const __vec16_i64 a = _a.cvt2hilo();
  const __vec16_i64 b = _b.cvt2hilo();
  const __mmask16 lo_match = _mm512_cmpeq_epi32_mask(a.v_lo,b.v_lo);
  return _mm512_mask_cmpeq_epi32_mask(lo_match,a.v_hi,b.v_hi);
}
static FORCEINLINE __vec16_i1 __equal_i64_and_mask(__vec16_i64 _a, __vec16_i64 _b, __vec16_i1 mask) 
{
  const __vec16_i64 a = _a.cvt2hilo();
  const __vec16_i64 b = _b.cvt2hilo();
  __mmask16 lo_match = _mm512_cmpeq_epi32_mask(a.v_lo,b.v_lo);
  __mmask16 full_match = _mm512_mask_cmpeq_epi32_mask(lo_match,a.v_hi,b.v_hi);
  return _mm512_kand(full_match, (__mmask16)mask);
}
static FORCEINLINE __vec16_i1 __not_equal_i64(__vec16_i64 a, __vec16_i64 b) 
{
  return __not(__equal_i64(a,b));
}
static FORCEINLINE __vec16_i1 __not_equal_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 mask) 
{
  return __and(__not(__equal_i64(a,b)), mask);
}
CMP_OP(__vec16_i64, i64, uint64_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i64, i64, int64_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i64, i64, uint64_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i64, i64, int64_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i64, i64, uint64_t, __unsigned_less_than, <)
CMP_OP(__vec16_i64, i64, int64_t,  __signed_less_than, <)
CMP_OP(__vec16_i64, i64, uint64_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i64, i64, int64_t,  __signed_greater_than, >)


INSERT_EXTRACT(__vec16_i64, int64_t)


#define CASTL2I(_v_, _v_hi_, _v_lo_) \
  __vec16_i32 _v_hi_, _v_lo_;  \
  { \
  const __vec16_i64 v      = _v_.cvt2hilo(); \
  _v_hi_   = v.v_hi; \
  _v_lo_   = v.v_lo; }
#define CASTI2L(_ret_hi_, _ret_lo_) \
  __vec16_i64(_ret_hi_, _ret_lo_).cvt2zmm()
static FORCEINLINE __vec16_i64 __broadcast_i64(__vec16_i64 _v, int index) 
{
  CASTL2I(_v, v_hi, v_lo);
  const __vec16_i32 ret_hi = __broadcast_i32(v_hi, index);
  const __vec16_i32 ret_lo = __broadcast_i32(v_lo, index);
  return CASTI2L(ret_hi, ret_lo);
}
static FORCEINLINE __vec16_i64 __rotate_i64(const __vec16_i64 _v, const int index) 
{
  CASTL2I(_v, v_hi, v_lo);
  const __vec16_i32 ret_hi = __rotate_i32(v_hi, index);
  const __vec16_i32 ret_lo = __rotate_i32(v_lo, index);
  return CASTI2L(ret_hi, ret_lo);
}
SHIFT(__vec16_i64, i64, int64_t)

static FORCEINLINE __vec16_i64 __shuffle_double(__vec16_i64 _v, const __vec16_i32 index) 
{
  CASTL2I(_v, v_hi, v_lo);
  const __vec16_i32 ret_hi = __shuffle_i32(v_hi, index);
  const __vec16_i32 ret_lo = __shuffle_i32(v_lo, index);
  return CASTI2L(ret_hi, ret_lo);
}
static FORCEINLINE __vec16_i64 __shuffle2_double(__vec16_i64 _v0, __vec16_i64 _v1, const __vec16_i32 index)
{
  CASTL2I(_v0, v0_hi, v0_lo);
  CASTL2I(_v1, v1_hi, v1_lo);
  const __vec16_i32 ret_hi = __shuffle2_i32(v0_hi, v1_hi, index);
  const __vec16_i32 ret_lo = __shuffle2_i32(v0_lo, v1_lo, index);
  return CASTI2L(ret_hi, ret_lo);
}
#undef CASTI2L
#undef CASTL2I

template <int ALIGN> static FORCEINLINE __vec16_i64 __load(const __vec16_i64 *p) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  // return __load<128>(p);
  __m512i v2 = _mm512_load_epi32(p);
  __m512i v1 = _mm512_load_epi32(((uint8_t*)p)+64);
  return __vec16_i64(v2,v1);
#else
  __vec16_i32 v1;
  __vec16_i32 v2;
  v2 = _mm512_extloadunpacklo_epi32(v2, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v2 = _mm512_extloadunpackhi_epi32(v2, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v1 = _mm512_extloadunpacklo_epi32(v1, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v1 = _mm512_extloadunpackhi_epi32(v1, (uint8_t*)p+128, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  return __vec16_i64(v2,v1);
#endif
}


template <int ALIGN> static FORCEINLINE void __store(__vec16_i64 *p, __vec16_i64 v) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  // __store<128>(p,v);
  __m512i v1 = v.v2;
  __m512i v2 = v.v1;
  _mm512_store_epi64(p, v2);
  _mm512_store_epi64(((uint8_t*)p)+64, v1);
#else
  __m512i v1 = v.v2;
  __m512i v2 = v.v1;
  _mm512_extpackstorelo_epi32(p, v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+64, v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorelo_epi32((uint8_t*)p+64, v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+128, v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
#endif
}

#if 0 /* knc::fails  as with _i32 this may generate fails ... so commetining it out */
template <> FORCEINLINE __vec16_i64 __load<64>(const __vec16_i64 *p) 
{
  __m512i v2 = _mm512_load_epi32(p);
  __m512i v1 = _mm512_load_epi32(((uint8_t*)p)+64);
  return __vec16_i64(v2,v1);
}
template <> FORCEINLINE __vec16_i64 __load<128>(const __vec16_i64 *p) {    return __load<64>(p); }
template <> FORCEINLINE void __store<64>(__vec16_i64 *p, __vec16_i64 v) 
{
  __m512i v1 = v.v2;
  __m512i v2 = v.v1;
  _mm512_store_epi64(p, v2);
  _mm512_store_epi64(((uint8_t*)p)+64, v1);
}
template <> FORCEINLINE void __store<128>(__vec16_i64 *p, __vec16_i64 v) {    __store<64>(p, v); }
#endif


///////////////////////////////////////////////////////////////////////////
// float
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_f __add(__vec16_f a, __vec16_f b) { return _mm512_add_ps(a,b); }
static FORCEINLINE __vec16_f __sub(__vec16_f a, __vec16_f b) { return _mm512_sub_ps(a,b); }
static FORCEINLINE __vec16_f __mul(__vec16_f a, __vec16_f b) { return _mm512_mul_ps(a,b); }
static FORCEINLINE __vec16_f __div(__vec16_f a, __vec16_f b) { return _mm512_div_ps(a,b); }

static FORCEINLINE __vec16_i1 __equal_float        (__vec16_f a, __vec16_f b) { return _mm512_cmpeq_ps_mask (a,b);            }
static FORCEINLINE __vec16_i1 __not_equal_float    (__vec16_f a, __vec16_f b) { return _mm512_cmpneq_ps_mask(a,b);            }
static FORCEINLINE __vec16_i1 __less_than_float    (__vec16_f a, __vec16_f b) { return _mm512_cmplt_ps_mask (a,b);            }
static FORCEINLINE __vec16_i1 __less_equal_float   (__vec16_f a, __vec16_f b) { return _mm512_cmple_ps_mask (a,b);            }
static FORCEINLINE __vec16_i1 __greater_than_float (__vec16_f a, __vec16_f b) { return _mm512_cmp_ps_mask   (a,b,_CMP_GT_OS); }
static FORCEINLINE __vec16_i1 __greater_equal_float(__vec16_f a, __vec16_f b) { return _mm512_cmp_ps_mask   (a,b,_CMP_GE_OS); }

static FORCEINLINE __vec16_i1 __equal_float_and_mask        (__vec16_f a, __vec16_f b, __vec16_i1 m) { return _mm512_mask_cmpeq_ps_mask (m,a,b);            }
static FORCEINLINE __vec16_i1 __not_equal_float_and_mask    (__vec16_f a, __vec16_f b, __vec16_i1 m) { return _mm512_mask_cmpneq_ps_mask(m,a,b);            }
static FORCEINLINE __vec16_i1 __less_than_float_and_mask    (__vec16_f a, __vec16_f b, __vec16_i1 m) { return _mm512_mask_cmplt_ps_mask (m,a,b);            }
static FORCEINLINE __vec16_i1 __less_equal_float_and_mask   (__vec16_f a, __vec16_f b, __vec16_i1 m) { return _mm512_mask_cmple_ps_mask (m,a,b);            }
static FORCEINLINE __vec16_i1 __greater_than_float_and_mask (__vec16_f a, __vec16_f b, __vec16_i1 m) { return _mm512_mask_cmp_ps_mask   (m,a,b,_CMP_GT_OS); }
static FORCEINLINE __vec16_i1 __greater_equal_float_and_mask(__vec16_f a, __vec16_f b, __vec16_i1 m) { return _mm512_mask_cmp_ps_mask   (m,a,b,_CMP_GE_OS); }

static FORCEINLINE __vec16_i1   __ordered_float(__vec16_f a, __vec16_f b) { return _mm512_cmpord_ps_mask  (a,b); }
static FORCEINLINE __vec16_i1 __unordered_float(__vec16_f a, __vec16_f b) { return _mm512_cmpunord_ps_mask(a,b); }

static FORCEINLINE __vec16_f __select(__vec16_i1 mask, __vec16_f a, __vec16_f b) { return _mm512_mask_mov_ps(b, mask, a); }
static FORCEINLINE __vec16_f __select(      bool cond, __vec16_f a, __vec16_f b) { return cond ? a : b; }

static FORCEINLINE float __extract_element(__vec16_f  v, uint32_t index)            { return v[index];   }
static FORCEINLINE void   __insert_element(__vec16_f *v, uint32_t index, float val) { (*v)[index] = val; }

template <class RetVecType> static RetVecType __smear_float(float f);
template <> FORCEINLINE __vec16_f __smear_float<__vec16_f>(float f) { return _mm512_set_1to16_ps(f); }

template <class RetVecType> static RetVecType __setzero_float();
template <> FORCEINLINE __vec16_f __setzero_float<__vec16_f>() { return _mm512_setzero_ps(); }

template <class RetVecType> static RetVecType __undef_float();
template <> FORCEINLINE __vec16_f __undef_float<__vec16_f>() { return __vec16_f(); }

static FORCEINLINE __vec16_f __broadcast_float(__vec16_f _v, int index) 
{
  const __vec16_i32 v = _mm512_castps_si512(_v);
  return _mm512_castsi512_ps(_mm512_mask_permutevar_epi32(v, 0xFFFF, _mm512_set1_epi32(index), v));
}
 
static FORCEINLINE __vec16_f __rotate_float(__vec16_f _v, int index) 
{
  const __vec16_i32 v =  _mm512_castps_si512(_v);
  const __vec16_i32 idx = __smear_i32<__vec16_i32>(index);
  const __vec16_i32 shuffle = _mm512_and_epi32(_mm512_add_epi32(__ispc_stride1, idx),  __smear_i32<__vec16_i32>(0xF));
  return _mm512_castsi512_ps(_mm512_mask_permutevar_epi32(v, 0xFFFF, shuffle, v));
}
SHIFT(__vec16_f, float, float)
static FORCEINLINE __vec16_f __shuffle_float(__vec16_f v, __vec16_i32 index) 
{
  return _mm512_castsi512_ps(_mm512_mask_permutevar_epi32(_mm512_castps_si512(v), 0xffff, index, _mm512_castps_si512(v)));
}
static FORCEINLINE __vec16_f __shuffle2_float(__vec16_f _v0, __vec16_f _v1, __vec16_i32 index)
{
  const __vec16_i32 v0 =  _mm512_castps_si512(_v0);
  const __vec16_i32 v1 =  _mm512_castps_si512(_v1);
  const __vec16_i1 mask = __signed_less_than_i32(index, __smear_i32<__vec16_i32>(0x10));
  index  = __and(index, __smear_i32<__vec16_i32>(0xF));
  __vec16_i32 ret = __undef_i32<__vec16_i32>();
  ret = _mm512_mask_permutevar_epi32(ret,       mask,  index, v0);
  ret = _mm512_mask_permutevar_epi32(ret, __not(mask), index, v1);
  return _mm512_castsi512_ps(ret);
}

template <int ALIGN> static FORCEINLINE __vec16_f __load(const __vec16_f *p) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  // return __load<64>(p);
  return _mm512_load_ps(p);
#else
  __vec16_f v;
  v = _mm512_extloadunpacklo_ps(v,           p,    _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  v = _mm512_extloadunpackhi_ps(v, (uint8_t*)p+64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  return v;
#endif
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_f *p, __vec16_f v) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  // __store<64>(p,v);
  _mm512_store_ps(p, v);
#else
  _mm512_extpackstorelo_ps(          p,    v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_ps((uint8_t*)p+64, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
}

#if 0 /* knc::fails  ./tests/gs-improve-progindex.ispc with segfault */
template <> FORCEINLINE __vec16_f __load<64>(const __vec16_f *p) 
{
    return _mm512_load_ps(p);
}
/* this one doesn't fail but it is  commented out for completeness, no aligned load/stores */
template <> FORCEINLINE void __store<64>(__vec16_f *p, __vec16_f v) 
{
  _mm512_store_ps(p, v);
}
#endif

/******** bitcast ******/

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

///////////////////////////////////////////////////////////////////////////
// half<->float : this one passes the tests 
// source : 
// http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion 
///////////////////////////////////////////////////////////////////////////
class Float16Compressor
{
  union Bits
  {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const shift = 13;
  static int const shiftSign = 16;

  static int32_t const infN = 0x7F800000; // flt32 infinity
  static int32_t const maxN = 0x477FE000; // max flt16 normal as a flt32
  static int32_t const minN = 0x38800000; // min flt16 normal as a flt32
  static int32_t const signN = 0x80000000; // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign; // flt16 sign bit

  static int32_t const mulN = 0x52000000; // (1 << 23) / minN
  static int32_t const mulC = 0x33800000; // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF; // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400; // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  public:

  static uint16_t compress(float value)
  {
    Bits v, s;
    v.f = value;
    uint32_t sign = v.si & signN;
    v.si ^= sign;
    sign >>= shiftSign; // logical shift
    s.si = mulN;
    s.si = s.f * v.f; // correct subnormals
    v.si ^= (s.si ^ v.si) & -(minN > v.si);
    v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
    v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
    v.ui >>= shift; // logical shift
    v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
    v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
    return v.ui | sign;
  }

  static float decompress(uint16_t value)
  {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }
};

static FORCEINLINE float __half_to_float_uniform(int16_t h) 
{
  return Float16Compressor::decompress(h);
}
static FORCEINLINE __vec16_f __half_to_float_varying(__vec16_i16 v) 
{
  __vec16_f ret;
  for (int i = 0; i < 16; ++i)
    ret[i] = __half_to_float_uniform(v[i]);
  return ret;
}


static FORCEINLINE int16_t __float_to_half_uniform(float f) 
{
  return Float16Compressor::compress(f);
}
static FORCEINLINE __vec16_i16 __float_to_half_varying(__vec16_f v) 
{
  __vec16_i16 ret;
  for (int i = 0; i < 16; ++i)
    ret[i] = __float_to_half_uniform(v[i]);
  return ret;
}


///////////////////////////////////////////////////////////////////////////
// double
///////////////////////////////////////////////////////////////////////////

#define VECOP(OP) __vec16_d(_mm512_##OP(a.v1,b.v1),_mm512_##OP(a.v2,b.v2))
static FORCEINLINE __vec16_d __add(__vec16_d a, __vec16_d b) { return VECOP(add_pd); }
static FORCEINLINE __vec16_d __sub(__vec16_d a, __vec16_d b) { return VECOP(sub_pd); }
static FORCEINLINE __vec16_d __mul(__vec16_d a, __vec16_d b) { return VECOP(mul_pd); }
static FORCEINLINE __vec16_d __div(__vec16_d a, __vec16_d b) { return VECOP(div_pd); }
#undef VECOP

#define CMPOP(OP) _mm512_kmovlhb(_mm512_##OP(a.v1,b.v1),_mm512_##OP(a.v2,b.v2))
static FORCEINLINE __vec16_i1 __equal_double        (__vec16_d a, __vec16_d b) { return CMPOP(cmpeq_pd_mask);    }
static FORCEINLINE __vec16_i1 __not_equal_double    (__vec16_d a, __vec16_d b) { return CMPOP(cmpneq_pd_mask);   }
static FORCEINLINE __vec16_i1 __less_than_double    (__vec16_d a, __vec16_d b) { return CMPOP(cmplt_pd_mask);    }
static FORCEINLINE __vec16_i1 __less_equal_double   (__vec16_d a, __vec16_d b) { return CMPOP(cmple_pd_mask);    }
static FORCEINLINE __vec16_i1 __greater_than_double (__vec16_d a, __vec16_d b) { return CMPOP(cmpnle_pd_mask);   }
static FORCEINLINE __vec16_i1 __greater_equal_double(__vec16_d a, __vec16_d b) { return CMPOP(cmpnlt_pd_mask);   }
static FORCEINLINE __vec16_i1 __ordered_double      (__vec16_d a, __vec16_d b) { return CMPOP(cmpord_pd_mask);   }
static FORCEINLINE __vec16_i1 __unordered_double    (__vec16_d a, __vec16_d b) { return CMPOP(cmpunord_pd_mask); }
#undef CMPOP

#define CMPOPMASK(OP) _mm512_kmovlhb(_mm512_mask_##OP(m,a.v1,b.v1),_mm512_mask_##OP(_mm512_kswapb(m,m),a.v2,b.v2))
static FORCEINLINE __vec16_i1 __equal_double_and_mask        (__vec16_d a, __vec16_d b, __vec16_i1 m) { return CMPOPMASK(cmpeq_pd_mask);  }
static FORCEINLINE __vec16_i1 __not_equal_double_and_mask    (__vec16_d a, __vec16_d b, __vec16_i1 m) { return CMPOPMASK(cmpneq_pd_mask); }
static FORCEINLINE __vec16_i1 __less_than_double_and_mask    (__vec16_d a, __vec16_d b, __vec16_i1 m) { return CMPOPMASK(cmplt_pd_mask);  }
static FORCEINLINE __vec16_i1 __less_equal_double_and_mask   (__vec16_d a, __vec16_d b, __vec16_i1 m) { return CMPOPMASK(cmple_pd_mask);  }
static FORCEINLINE __vec16_i1 __greater_than_double_and_mask (__vec16_d a, __vec16_d b, __vec16_i1 m) { return CMPOPMASK(cmpnle_pd_mask); }
static FORCEINLINE __vec16_i1 __greater_equal_double_and_mask(__vec16_d a, __vec16_d b, __vec16_i1 m) { return CMPOPMASK(cmpnlt_pd_mask); }
#undef CMOPMASK


static FORCEINLINE __vec16_d __select(__vec16_i1 m, __vec16_d a, __vec16_d b) 
{
  return __vec16_d(_mm512_mask_mov_pd(b.v1, m, a.v1), _mm512_mask_mov_pd(b.v2, _mm512_kswapb(m, m), a.v2));
}
static FORCEINLINE __vec16_d __select(bool cond, __vec16_d a, __vec16_d b) 
{
    return cond ? a : b;
}

static FORCEINLINE double __extract_element(__vec16_d  v, uint32_t index)             { return v[index];   }
static FORCEINLINE void    __insert_element(__vec16_d *v, uint32_t index, double val) { (*v)[index] = val; }

template <class RetVecType> static RetVecType __smear_double(double d);
template <> FORCEINLINE __vec16_d __smear_double<__vec16_d>(double d) { return __vec16_d(_mm512_set1_pd(d), _mm512_set1_pd(d)); }

template <class RetVecType> static RetVecType __setzero_double();
template <> FORCEINLINE __vec16_d __setzero_double<__vec16_d>() { return __vec16_d(_mm512_setzero_pd(), _mm512_setzero_pd()); }

template <class RetVecType> static RetVecType __undef_double();
template <> FORCEINLINE __vec16_d __undef_double<__vec16_d>() { return __vec16_d(); }

#define CASTD2F(_v_, _v_hi_, _v_lo_) \
  __vec16_f _v_hi_, _v_lo_;  \
  { \
  const __vec16_d v      = _v_.cvt2hilo(); \
  _v_hi_   = _mm512_castpd_ps(v.v_hi); \
  _v_lo_   = _mm512_castpd_ps(v.v_lo); }
#define CASTF2D(_ret_hi_, _ret_lo_) \
  __vec16_d(_mm512_castps_pd(_ret_hi_), _mm512_castps_pd(_ret_lo_)).cvt2zmm()
static FORCEINLINE __vec16_d __broadcast_double(__vec16_d _v, int index) 
{
  CASTD2F(_v, v_hi, v_lo);
  const __vec16_f ret_hi = __broadcast_float(v_hi, index);
  const __vec16_f ret_lo = __broadcast_float(v_lo, index);
  return CASTF2D(ret_hi, ret_lo);
}
static FORCEINLINE __vec16_d __rotate_double(const __vec16_d _v, const int index) 
{
  CASTD2F(_v, v_hi, v_lo);
  const __vec16_f ret_hi = __rotate_float(v_hi, index);
  const __vec16_f ret_lo = __rotate_float(v_lo, index);
  return CASTF2D(ret_hi, ret_lo);
}
SHIFT(__vec16_d, double, double)
static FORCEINLINE __vec16_d __shuffle_double(__vec16_d _v, const __vec16_i32 index) 
{
  CASTD2F(_v, v_hi, v_lo);
  const __vec16_f ret_hi = __shuffle_float(v_hi, index);
  const __vec16_f ret_lo = __shuffle_float(v_lo, index);
  return CASTF2D(ret_hi, ret_lo);
}
static FORCEINLINE __vec16_d __shuffle2_double(__vec16_d _v0, __vec16_d _v1, const __vec16_i32 index)
{
  CASTD2F(_v0, v0_hi, v0_lo);
  CASTD2F(_v1, v1_hi, v1_lo);
  const __vec16_f ret_hi = __shuffle2_float(v0_hi, v1_hi, index);
  const __vec16_f ret_lo = __shuffle2_float(v0_lo, v1_lo, index);
  return CASTF2D(ret_hi, ret_lo);
}
#undef CASTF2D
#undef CASTD2F

template <int ALIGN> static FORCEINLINE __vec16_d __load(const __vec16_d *p) \
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  // return __load<128>(p);
  return __vec16_d(_mm512_load_pd(p), _mm512_load_pd(((uint8_t*)p)+64));
#else
  __vec16_d ret;
  ret.v1 = _mm512_extloadunpacklo_pd(ret.v1, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  ret.v1 = _mm512_extloadunpackhi_pd(ret.v1, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  ret.v2 = _mm512_extloadunpacklo_pd(ret.v2, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  ret.v2 = _mm512_extloadunpackhi_pd(ret.v2, (uint8_t*)p+128, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  return ret;
#endif
}
 
template <int ALIGN> static FORCEINLINE void __store(__vec16_d *p, __vec16_d v) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  // return __store<128>(p,v);
  _mm512_store_pd(p, v.v1);
  _mm512_store_pd(((uint8_t*)p)+64, v.v2);
#else
  _mm512_extpackstorelo_pd(p, v.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_pd((uint8_t*)p+64, v.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorelo_pd((uint8_t*)p+64, v.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_pd((uint8_t*)p+128, v.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
#endif
}


#if 0 /* knc::fails  as with _f this may generate fails ... so commetining it out */
template <> FORCEINLINE __vec16_d __load<64>(const __vec16_d *p) 
{
  return __vec16_d(_mm512_load_pd(p), _mm512_load_pd(((uint8_t*)p)+64));
}
template <> FORCEINLINE void __store<64>(__vec16_d *p, __vec16_d v) 
{
  _mm512_store_pd(p, v.v1);
  _mm512_store_pd(((uint8_t*)p)+64, v.v2);
}
template <> FORCEINLINE __vec16_d __load <128>(const __vec16_d *p)        { return __load<64>(p); }
template <> FORCEINLINE      void __store<128>(__vec16_d *p, __vec16_d v) { __store<64>(p, v);    }
#endif

///////////////////////////////////////////////////////////////////////////
// casts
///////////////////////////////////////////////////////////////////////////


/* knc::macro::used */
#define CAST(TO, STO, FROM, SFROM, FUNC)        \
static FORCEINLINE TO FUNC(TO, FROM val) {      \
    TO ret;                                     \
    for (int i = 0; i < 16; ++i)                \
        ret[i] = (STO)((SFROM)(val[i]));    \
    return ret;                                 \
}

// sign extension conversions

// CAST(__vec16_i64, int64_t, __vec16_i32, int32_t, __cast_sext)
static FORCEINLINE __vec16_i64 __cast_sext(const __vec16_i64 &, const __vec16_i32 &val)
{
  return __vec16_i64(_mm512_srai_epi32(val.v,31), val.v).cvt2zmm();
}
CAST(__vec16_i64, int64_t, __vec16_i16, int16_t, __cast_sext)
CAST(__vec16_i64, int64_t, __vec16_i8,  int8_t,  __cast_sext)
CAST(__vec16_i32, int32_t, __vec16_i16, int16_t, __cast_sext)
CAST(__vec16_i32, int32_t, __vec16_i8,  int8_t,  __cast_sext)
CAST(__vec16_i16, int16_t, __vec16_i8,  int8_t,  __cast_sext)

/* knc::macro::used */
#define CAST_SEXT_I1(TYPE)                            \
static FORCEINLINE TYPE __cast_sext(TYPE, __vec16_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 16; ++i) {                    \
        ret[i] = 0;                                 \
        if (v.v & (1 << i))                           \
            ret[i] = ~ret[i];                     \
    }                                                 \
    return ret;                                       \
}

CAST_SEXT_I1(__vec16_i8)
CAST_SEXT_I1(__vec16_i16)

//CAST_SEXT_I1(__vec16_i32)
static FORCEINLINE __vec16_i32 __cast_sext(const __vec16_i32 &, const __vec16_i1 &val)
{
  __vec16_i32 ret = _mm512_setzero_epi32();
  __vec16_i32 one = _mm512_set1_epi32(-1);
  return _mm512_mask_mov_epi32(ret, val, one);
}

CAST_SEXT_I1(__vec16_i64)

// zero extension
// CAST(__vec16_i64, uint64_t, __vec16_i32, uint32_t, __cast_zext)
static FORCEINLINE __vec16_i64 __cast_zext(const __vec16_i64 &, const __vec16_i32 &val)
{
  return __vec16_i64(_mm512_setzero_epi32(), val.v).cvt2zmm();
}

CAST(__vec16_i64, uint64_t, __vec16_i16, uint16_t, __cast_zext)
CAST(__vec16_i64, uint64_t, __vec16_i8,  uint8_t,  __cast_zext)
CAST(__vec16_i32, uint32_t, __vec16_i16, uint16_t, __cast_zext)
CAST(__vec16_i32, uint32_t, __vec16_i8,  uint8_t,  __cast_zext)
CAST(__vec16_i16, uint16_t, __vec16_i8,  uint8_t,  __cast_zext)

/* knc::macro::used */
#define CAST_ZEXT_I1(TYPE)                            \
static FORCEINLINE TYPE __cast_zext(TYPE, __vec16_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 16; ++i)                      \
        ret[i] = (v.v & (1 << i)) ? 1 : 0;          \
    return ret;                                       \
}

CAST_ZEXT_I1(__vec16_i8)
CAST_ZEXT_I1(__vec16_i16)

//CAST_ZEXT_I1(__vec16_i32)
static FORCEINLINE __vec16_i32 __cast_zext(const __vec16_i32 &, const __vec16_i1 &val)
{
  __vec16_i32 ret = _mm512_setzero_epi32();
  __vec16_i32 one = _mm512_set1_epi32(1);
  return _mm512_mask_mov_epi32(ret, val, one);
}

CAST_ZEXT_I1(__vec16_i64)

// truncations
CAST(__vec16_i32, int32_t, __vec16_i64, int64_t, __cast_trunc)
CAST(__vec16_i16, int16_t, __vec16_i64, int64_t, __cast_trunc)
CAST(__vec16_i8,  int8_t,  __vec16_i64, int64_t, __cast_trunc)
CAST(__vec16_i16, int16_t, __vec16_i32, int32_t, __cast_trunc)
CAST(__vec16_i8,  int8_t,  __vec16_i32, int32_t, __cast_trunc)
CAST(__vec16_i8,  int8_t,  __vec16_i16, int16_t, __cast_trunc)

// signed int to float/double

//CAST(__vec16_f, float, __vec16_i8,   int8_t,  __cast_sitofp)
static FORCEINLINE __vec16_f __cast_sitofp(__vec16_f, __vec16_i8  val) {return _mm512_extload_ps(&val, _MM_UPCONV_PS_SINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);}
//CAST(__vec16_f, float, __vec16_i16,  int16_t, __cast_sitofp)
static FORCEINLINE __vec16_f __cast_sitofp(__vec16_f, __vec16_i16 val) {return _mm512_extload_ps(&val, _MM_UPCONV_PS_SINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);}
//CAST(__vec16_f, float, __vec16_i32,  int32_t, __cast_sitofp)
static FORCEINLINE __vec16_f __cast_sitofp(__vec16_f, __vec16_i32 val) {return _mm512_cvtfxpnt_round_adjustepi32_ps(val, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);}

CAST(__vec16_f, float, __vec16_i64,  int64_t, __cast_sitofp)

//CAST(__vec16_d, double, __vec16_i8,  int8_t,  __cast_sitofp)
static FORCEINLINE __vec16_d __cast_sitofp(__vec16_d, __vec16_i8 val) 
{
  __vec16_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
  __vec16_d ret;
  ret.v1 = _mm512_cvtepi32lo_pd(vi);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(vi, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepi32lo_pd(other8);
  return ret;
}

// CAST(__vec16_d, double, __vec16_i16, int16_t, __cast_sitofp)
static FORCEINLINE __vec16_d __cast_sitofp(__vec16_d, __vec16_i16 val) 
{
  __vec16_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
  __vec16_d ret;
  ret.v1 = _mm512_cvtepi32lo_pd(vi);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(vi, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepi32lo_pd(other8);
  return ret;
}

// CAST(__vec16_d, double, __vec16_i32, int32_t, __cast_sitofp)
static FORCEINLINE __vec16_d __cast_sitofp(__vec16_d, __vec16_i32 val) 
{
  __vec16_d ret;
  ret.v1 = _mm512_cvtepi32lo_pd(val);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(val, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepi32lo_pd(other8);
  return ret;
}

CAST(__vec16_d, double, __vec16_i64, int64_t, __cast_sitofp)

// unsigned int to float/double

// CAST(__vec16_f, float, __vec16_i8,   uint8_t,  __cast_uitofp)
static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i8  val) {return _mm512_extload_ps(&val, _MM_UPCONV_PS_UINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);}
//CAST(__vec16_f, float, __vec16_i16,  uint16_t, __cast_uitofp)
static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i16 val) {return _mm512_extload_ps(&val, _MM_UPCONV_PS_UINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);}
//CAST(__vec16_f, float, __vec16_i32,  uint32_t, __cast_uitofp)
static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i32 val) {return _mm512_cvtfxpnt_round_adjustepu32_ps(val, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);}

CAST(__vec16_f, float, __vec16_i64,  uint64_t, __cast_uitofp)

// CAST(__vec16_d, double, __vec16_i8,  uint8_t,  __cast_uitofp)
static FORCEINLINE __vec16_d __cast_uitofp(__vec16_d, __vec16_i8 val) 
{
  __vec16_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
  __vec16_d ret;
  ret.v1 = _mm512_cvtepu32lo_pd(vi);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(vi, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepu32lo_pd(other8);
  return ret;
}

// CAST(__vec16_d, double, __vec16_i16, uint16_t, __cast_uitofp)
static FORCEINLINE __vec16_d __cast_uitofp(__vec16_d, __vec16_i16 val) 
{
  __vec16_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_UINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
  __vec16_d ret;
  ret.v1 = _mm512_cvtepu32lo_pd(vi);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(vi, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepu32lo_pd(other8);
  return ret;
}

// CAST(__vec16_d, double, __vec16_i32, uint32_t, __cast_uitofp)
static FORCEINLINE __vec16_d __cast_uitofp(__vec16_d, __vec16_i32 val) 
{
  __vec16_d ret;
  ret.v1 = _mm512_cvtepu32lo_pd(val);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(val, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepu32lo_pd(other8);
  return ret;
}

CAST(__vec16_d, double, __vec16_i64, uint64_t, __cast_uitofp)

static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i1 v) 
{
  const __m512 ret = _mm512_setzero_ps();
  const __m512 one = _mm512_set1_ps(1.0);
  return _mm512_mask_mov_ps(ret, v, one);
}

// float/double to signed int
CAST(__vec16_i8,  int8_t,  __vec16_f, float, __cast_fptosi)
CAST(__vec16_i16, int16_t, __vec16_f, float, __cast_fptosi)

// CAST(__vec16_i32, int32_t, __vec16_f, float, __cast_fptosi)
static FORCEINLINE __vec16_i32 __cast_fptosi(__vec16_i32, __vec16_f val) 
{
  return _mm512_cvtfxpnt_round_adjustps_epi32(val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

CAST(__vec16_i64, int64_t, __vec16_f, float, __cast_fptosi)
CAST(__vec16_i8,  int8_t,  __vec16_d, double, __cast_fptosi)
CAST(__vec16_i16, int16_t, __vec16_d, double, __cast_fptosi)
#if 0 /* knc::2implement */
#else
CAST(__vec16_i32, int32_t, __vec16_d, double, __cast_fptosi)
#endif
CAST(__vec16_i64, int64_t, __vec16_d, double, __cast_fptosi)

// float/double to unsigned int
CAST(__vec16_i8,  uint8_t,  __vec16_f, float, __cast_fptoui)
CAST(__vec16_i16, uint16_t, __vec16_f, float, __cast_fptoui)

// CAST(__vec16_i32, uint32_t, __vec16_f, float, __cast_fptoui)
static FORCEINLINE __vec16_i32 __cast_fptoui(__vec16_i32, __vec16_f val) 
{
  return _mm512_cvtfxpnt_round_adjustps_epu32(val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

CAST(__vec16_i64, uint64_t, __vec16_f, float, __cast_fptoui)
CAST(__vec16_i8,  uint8_t,  __vec16_d, double, __cast_fptoui)
CAST(__vec16_i16, uint16_t, __vec16_d, double, __cast_fptoui)
#if 0 /* knc::2implement */
#else
CAST(__vec16_i32, uint32_t, __vec16_d, double, __cast_fptoui)
#endif
CAST(__vec16_i64, uint64_t, __vec16_d, double, __cast_fptoui)

// float/double conversions

// CAST(__vec16_f, float,  __vec16_d, double, __cast_fptrunc)
static FORCEINLINE __vec16_f __cast_fptrunc(__vec16_f, __vec16_d val) 
{
  __m512i r0i = _mm512_castps_si512(_mm512_cvtpd_pslo(val.v1));
  __m512i r1i = _mm512_castps_si512(_mm512_cvtpd_pslo(val.v2));

  return _mm512_castsi512_ps(_mm512_mask_permute4f128_epi32(r0i, 0xFF00, r1i, _MM_PERM_BABA));
}

// CAST(__vec16_d, double, __vec16_f, float,  __cast_fpext)
static FORCEINLINE __vec16_d __cast_fpext(__vec16_d, __vec16_f val) 
{
  __vec16_d ret;
  ret.v1 = _mm512_cvtpslo_pd(val.v);
  __vec16_f other8 = _mm512_castsi512_ps(_mm512_permute4f128_epi32(_mm512_castps_si512(val.v), _MM_PERM_DCDC));
  ret.v2 = _mm512_cvtpslo_pd(other8);
  return ret;
}

typedef union {
    int32_t i32;
    float f;
    int64_t i64;
    double d;
} BitcastUnion;

/* knc::macro::not used */
#define CAST_BITS(TO, TO_ELT, FROM, FROM_ELT)       \
static FORCEINLINE TO __cast_bits(TO, FROM val) {   \
    TO r;                                           \
    for (int i = 0; i < 16; ++i) {                  \
        BitcastUnion u;                             \
        u.FROM_ELT = val[i];                      \
        r[i] = u.TO_ELT;                          \
    }                                               \
    return r;                                       \
}

// CAST_BITS(__vec16_f,   f,   __vec16_i32, i32)
static FORCEINLINE __vec16_f __cast_bits(__vec16_f, __vec16_i32 val) { return _mm512_castsi512_ps(val); }
// CAST_BITS(__vec16_i32, i32, __vec16_f,   f)
static FORCEINLINE __vec16_i32 __cast_bits(__vec16_i32, __vec16_f val) { return _mm512_castps_si512(val); }

// CAST_BITS(__vec16_d,   d,   __vec16_i64, i64)
static FORCEINLINE __vec16_i64 __cast_bits(__vec16_i64, __vec16_d val) { return *(__vec16_i64*)&val; }
// CAST_BITS(__vec16_i64, i64, __vec16_d,   d)
static FORCEINLINE __vec16_d __cast_bits(__vec16_d, __vec16_i64 val) { return *(__vec16_d*)&val; }

/* knc::macro::used */
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
///////////////////////////////////////////////////////////////////////////

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

static FORCEINLINE __vec16_f __round_varying_float(__vec16_f v) { return _mm512_round_ps(v, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE); }
static FORCEINLINE __vec16_f __floor_varying_float(__vec16_f v) { return _mm512_floor_ps(v); }
static FORCEINLINE __vec16_f __ceil_varying_float(__vec16_f v) { return _mm512_ceil_ps(v); }

static FORCEINLINE __vec16_d __round_varying_float(__vec16_d v)  { return __vec16_d(_mm512_svml_round_pd(v.v1), _mm512_svml_round_pd(v.v2)); }
static FORCEINLINE __vec16_d __floor_varying_float(__vec16_d v)  { return __vec16_d(_mm512_floor_pd(v.v1), _mm512_floor_pd(v.v2)); }
static FORCEINLINE __vec16_d __ceil_varying_float(__vec16_d v)  { return __vec16_d(_mm512_ceil_pd(v.v1), _mm512_ceil_pd(v.v2)); }

// min/max

static FORCEINLINE float  __min_uniform_float (float  a, float  b) { return (a<b) ? a : b; }
static FORCEINLINE float  __max_uniform_float (float  a, float  b) { return (a>b) ? a : b; }
static FORCEINLINE double __min_uniform_double(double a, double b) { return (a<b) ? a : b; }
static FORCEINLINE double __max_uniform_double(double a, double b) { return (a>b) ? a : b; }

static FORCEINLINE int32_t __min_uniform_int32 ( int32_t a,  int32_t b) { return (a<b) ? a : b; }
static FORCEINLINE int32_t __max_uniform_int32 ( int32_t a,  int32_t b) { return (a>b) ? a : b; }
static FORCEINLINE int32_t __min_uniform_uint32(uint32_t a, uint32_t b) { return (a<b) ? a : b; }
static FORCEINLINE int32_t __max_uniform_uint32(uint32_t a, uint32_t b) { return (a>b) ? a : b; }

static FORCEINLINE int64_t __min_uniform_int64 ( int64_t a,  int64_t b) { return (a<b) ? a : b; }
static FORCEINLINE int64_t __max_uniform_int64 ( int64_t a,  int64_t b) { return (a>b) ? a : b; }
static FORCEINLINE int64_t __min_uniform_uint64(uint64_t a, uint64_t b) { return (a<b) ? a : b; }
static FORCEINLINE int64_t __max_uniform_uint64(uint64_t a, uint64_t b) { return (a>b) ? a : b; }

static FORCEINLINE __vec16_f __max_varying_float (__vec16_f v1, __vec16_f v2) { return _mm512_gmax_ps(v1, v2);}
static FORCEINLINE __vec16_f __min_varying_float (__vec16_f v1, __vec16_f v2) { return _mm512_gmin_ps(v1, v2);}
static FORCEINLINE __vec16_d __max_varying_double(__vec16_d v1, __vec16_d v2) { return __vec16_d(_mm512_gmax_pd(v1.v1, v2.v1),_mm512_gmax_pd(v1.v2,v2.v2));}
static FORCEINLINE __vec16_d __min_varying_double(__vec16_d v1, __vec16_d v2) { return __vec16_d(_mm512_gmin_pd(v1.v1, v2.v1),_mm512_gmin_pd(v1.v2,v2.v2));}

static FORCEINLINE __vec16_i32 __max_varying_int32 (__vec16_i32 v1, __vec16_i32 v2) { return _mm512_max_epi32(v1, v2);}
static FORCEINLINE __vec16_i32 __min_varying_int32 (__vec16_i32 v1, __vec16_i32 v2) { return _mm512_min_epi32(v1, v2);}
static FORCEINLINE __vec16_i32 __max_varying_uint32(__vec16_i32 v1, __vec16_i32 v2) { return _mm512_max_epu32(v1, v2);}
static FORCEINLINE __vec16_i32 __min_varying_uint32(__vec16_i32 v1, __vec16_i32 v2) { return _mm512_min_epu32(v1, v2);}

BINARY_OP_FUNC(__vec16_i64, __max_varying_int64,  __max_uniform_int64)
BINARY_OP_FUNC(__vec16_i64, __min_varying_int64,  __min_uniform_int64)
BINARY_OP_FUNC(__vec16_i64, __max_varying_uint64, __max_uniform_uint64)
BINARY_OP_FUNC(__vec16_i64, __min_varying_uint64, __min_uniform_uint64)

// sqrt/rsqrt/rcp

static FORCEINLINE float  __rsqrt_uniform_float(float  v) { return 1.f / sqrtf(v); }
static FORCEINLINE float  __rcp_uniform_float  (float  v) { return 1.f / v;        }
static FORCEINLINE float  __sqrt_uniform_float (float  v) { return sqrtf(v);       }
static FORCEINLINE double __sqrt_uniform_double(double v) { return sqrt (v);       }

static FORCEINLINE __vec16_f __rcp_varying_float(__vec16_f v) 
{
#ifdef ISPC_FAST_MATH
  return _mm512_rcp23_ps(v); // Approximation with 23 bits of accuracy.
#else
  return _mm512_recip_ps(v);
#endif
}
static FORCEINLINE __vec16_d __rcp_varying_double(__vec16_d x) {
    __vec16_i64 ex = __and(__cast_bits(__vec16_i64(), x), __smear_i64<__vec16_i64>(0x7fe0000000000000));
    __vec16_d  exp = __cast_bits(__vec16_d(), __sub(__smear_i64<__vec16_i64>(0x7fd0000000000000), ex));
    __vec16_f   xf = __cast_fptrunc(__vec16_f(), __mul(x, exp));
    __vec16_f   yf = __rcp_varying_float(xf);
    __vec16_d    y = __mul(__cast_fpext(__vec16_d(), yf), exp);
    y = __add(y, __mul(y, __sub(__smear_double<__vec16_d>(2.0), __mul(x, y))));
    y = __add(y, __mul(y, __sub(__smear_double<__vec16_d>(2.0), __mul(x, y))));
    return y;
}
static FORCEINLINE double __rcp_uniform_double(double v) 
{
  return __extract_element(__rcp_varying_double(__smear_double<__vec16_d>(v)),0);
}

static FORCEINLINE __vec16_f __rsqrt_varying_float(__vec16_f v) 
{
#ifdef ISPC_FAST_MATH
  return _mm512_rsqrt23_ps(v); // Approximation with 0.775ULP accuracy
#else 
  return _mm512_invsqrt_ps(v);
#endif
}
static FORCEINLINE __vec16_d __rsqrt_varying_double(__vec16_d x) {
    __vec16_i64 ex = __and(__cast_bits(__vec16_i64(), x), __smear_i64<__vec16_i64>(0x7fe0000000000000));
    __vec16_d  exp = __cast_bits(__vec16_d(), __sub(__smear_i64<__vec16_i64>(0x7fd0000000000000), ex));
    __vec16_d exph = __cast_bits(__vec16_d(), __sub(__smear_i64<__vec16_i64>(0x5fe0000000000000), __lshr(ex,1)));
    __vec16_f   xf = __cast_fptrunc(__vec16_f(), __mul(x, exp));
    __vec16_f   yf = __rsqrt_varying_float(xf);
    __vec16_d    y = __mul(__cast_fpext(__vec16_d(), yf), exph);
    __vec16_d   xh = __mul(x, __smear_double<__vec16_d>(0.5));
    y = __add(y, __mul(y, __sub(__smear_double<__vec16_d>(0.5), __mul(xh, __mul(y,y)))));
    y = __add(y, __mul(y, __sub(__smear_double<__vec16_d>(0.5), __mul(xh, __mul(y,y)))));
    return y;
}
static FORCEINLINE double __rsqrt_uniform_double(double v) 
{
  return __extract_element(__rsqrt_varying_double(__smear_double<__vec16_d>(v)),0);
}

static FORCEINLINE __vec16_f __sqrt_varying_float (__vec16_f v) { return _mm512_sqrt_ps(v);}
static FORCEINLINE __vec16_d __sqrt_varying_double(__vec16_d v) { return __vec16_d(_mm512_sqrt_pd(v.v1),_mm512_sqrt_pd(v.v2));}

///////////////////////////////////////////////////////////////////////////
// svml
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_f __svml_sinf  (__vec16_f v)              { return _mm512_sin_ps(v);     }
static FORCEINLINE __vec16_f __svml_asinf (__vec16_f v)              { return _mm512_asin_ps(v);    }
static FORCEINLINE __vec16_f __svml_cosf  (__vec16_f v)              { return _mm512_cos_ps(v);     }
static FORCEINLINE __vec16_f __svml_tanf  (__vec16_f v)              { return _mm512_tan_ps(v);     }
static FORCEINLINE __vec16_f __svml_atanf (__vec16_f v)              { return _mm512_atan_ps(v);    }
static FORCEINLINE __vec16_f __svml_atan2f(__vec16_f a, __vec16_f b) { return _mm512_atan2_ps(a,b); }
static FORCEINLINE __vec16_f __svml_expf  (__vec16_f v)              { return _mm512_exp_ps(v);     }
static FORCEINLINE __vec16_f __svml_logf  (__vec16_f v)              { return _mm512_log_ps(v);     }
static FORCEINLINE __vec16_f __svml_powf  (__vec16_f a, __vec16_f b) { return _mm512_pow_ps(a,b);   }

static FORCEINLINE __vec16_d __svml_sind  (__vec16_d v)              { return __vec16_d(_mm512_sin_pd(v.v1), _mm512_sin_pd(v.v2)); }
static FORCEINLINE __vec16_d __svml_asind (__vec16_d v)              { return __vec16_d(_mm512_asin_pd(v.v1), _mm512_asin_pd(v.v2)); }
static FORCEINLINE __vec16_d __svml_cosd  (__vec16_d v)              { return __vec16_d(_mm512_cos_pd(v.v1), _mm512_cos_pd(v.v2)); }
static FORCEINLINE __vec16_d __svml_tand  (__vec16_d v)              { return __vec16_d(_mm512_tan_pd(v.v1), _mm512_tan_pd(v.v2)); }
static FORCEINLINE __vec16_d __svml_atand (__vec16_d v)              { return __vec16_d(_mm512_atan_pd(v.v1), _mm512_atan_pd(v.v2)); }
static FORCEINLINE __vec16_d __svml_atan2d(__vec16_d a, __vec16_d b) { return __vec16_d(_mm512_atan2_pd(a.v1,b.v1), _mm512_atan2_pd(a.v2,b.v2)); }
static FORCEINLINE __vec16_d __svml_expd  (__vec16_d v)              { return __vec16_d(_mm512_exp_pd(v.v1), _mm512_exp_pd(v.v2)); }
static FORCEINLINE __vec16_d __svml_logd  (__vec16_d v)              { return __vec16_d(_mm512_log_pd(v.v1), _mm512_log_pd(v.v2)); }
static FORCEINLINE __vec16_d __svml_powd  (__vec16_d a, __vec16_d b) { return __vec16_d(_mm512_pow_pd(a.v1,b.v1), _mm512_pow_pd(a.v2,b.v2)); }

///////////////////////////////////////////////////////////////////////////
// bit ops
///////////////////////////////////////////////////////////////////////////

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
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE float __reduce_add_float(__vec16_f v) { return _mm512_reduce_add_ps(v); }
static FORCEINLINE float __reduce_min_float(__vec16_f v) { return _mm512_reduce_min_ps(v); }
static FORCEINLINE float __reduce_max_float(__vec16_f v) { return _mm512_reduce_max_ps(v); }

static FORCEINLINE float __reduce_add_double(__vec16_d v) { return _mm512_reduce_add_pd(v.v1) + _mm512_reduce_add_pd(v.v2); }
static FORCEINLINE float __reduce_min_double(__vec16_d v) { return std::min(_mm512_reduce_min_pd(v.v1), _mm512_reduce_min_pd(v.v2)); }
static FORCEINLINE float __reduce_max_double(__vec16_d v) { return std::max(_mm512_reduce_max_pd(v.v1), _mm512_reduce_max_pd(v.v2)); }



static FORCEINLINE  int64_t __reduce_add_int32  (__vec16_i32 v) { return _mm512_reduce_add_epi32(v);}
static FORCEINLINE  int32_t __reduce_min_int32  (__vec16_i32 v) { return _mm512_reduce_min_epi32(v);}
static FORCEINLINE  int32_t __reduce_max_int32  (__vec16_i32 v) { return _mm512_reduce_max_epi32(v);}
static FORCEINLINE uint32_t __reduce_min_uint32 (__vec16_i32 v) { return _mm512_reduce_min_epu32(v);}
static FORCEINLINE uint32_t __reduce_max_uint32 (__vec16_i32 v) { return _mm512_reduce_max_epu32(v);}

REDUCE_ADD   ( int16_t, __vec16_i8,  __reduce_add_int8)
REDUCE_ADD   ( int32_t, __vec16_i16, __reduce_add_int16)
REDUCE_ADD   ( int64_t, __vec16_i64, __reduce_add_int64)
REDUCE_MINMAX( int64_t, __vec16_i64, __reduce_min_int64, <)
REDUCE_MINMAX( int64_t, __vec16_i64, __reduce_max_int64, >)
REDUCE_MINMAX(uint64_t, __vec16_i64, __reduce_min_uint64, <)
REDUCE_MINMAX(uint64_t, __vec16_i64, __reduce_max_uint64, >)

///////////////////////////////////////////////////////////////////////////
// masked load/store
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_i8 __masked_load_i8(void *p,
                                               __vec16_i1 mask) {
    __vec16_i8 ret;
    int8_t *ptr = (int8_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec16_i16 __masked_load_i16(void *p,
                                                 __vec16_i1 mask) {
    __vec16_i16 ret;
    int16_t *ptr = (int16_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec16_i32 __masked_load_i32(void *p, __vec16_i1 mask) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  return _mm512_mask_load_epi32(__vec16_i32(), mask, p);
#else
  __vec16_i32 tmp;
  tmp.v = _mm512_mask_extloadunpacklo_epi32(tmp.v, 0xFFFF, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  tmp.v = _mm512_mask_extloadunpackhi_epi32(tmp.v, 0xFFFF, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  __vec16_i32 ret;
  return _mm512_mask_mov_epi32(ret.v, mask, tmp.v);
#endif
}

static FORCEINLINE __vec16_f __masked_load_float(void *p, __vec16_i1 mask) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  return _mm512_mask_load_ps(_mm512_undefined_ps(), mask,p);
#else
  __vec16_f tmp;
  tmp.v = _mm512_mask_extloadunpacklo_ps(tmp.v, 0xFFFF, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  tmp.v = _mm512_mask_extloadunpackhi_ps(tmp.v, 0xFFFF, (uint8_t*)p+64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  __vec16_f ret;
  return _mm512_mask_mov_ps(ret.v, mask, tmp.v);
#endif
}

static FORCEINLINE __vec16_i64 __masked_load_i64(void *p,
                                                 __vec16_i1 mask) {
    __vec16_i64 ret;
    int64_t *ptr = (int64_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec16_d __masked_load_double(void *p, __vec16_i1 mask) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  __vec16_d ret;
  __vec16_i1 tmp_m = mask;
  tmp_m = _mm512_kswapb(tmp_m, tmp_m);
  ret.v1 = _mm512_mask_load_pd(ret.v1, mask, p);
  ret.v2 = _mm512_mask_load_pd(ret.v2, tmp_m, (uint8_t*)p+64);
  return ret;
#else
  __vec16_d tmp;
  tmp.v1 = _mm512_mask_extloadunpacklo_pd(tmp.v1, 0xFF, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  tmp.v1 = _mm512_mask_extloadunpackhi_pd(tmp.v1, 0xFF, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  tmp.v2 = _mm512_mask_extloadunpacklo_pd(tmp.v2, 0xFF, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  tmp.v2 = _mm512_mask_extloadunpackhi_pd(tmp.v2, 0xFF, (uint8_t*)p+128, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  __vec16_d ret;
  __vec16_i1 tmp_m = mask;
  tmp_m = _mm512_kswapb(tmp_m, tmp_m);
  ret.v1 = _mm512_mask_mov_pd(ret.v1, mask, tmp.v1);
  ret.v2 = _mm512_mask_mov_pd(ret.v2, tmp_m, tmp.v2);
  return ret;
#endif
}


static FORCEINLINE void __masked_store_i8(void *p, __vec16_i8 val,
                                          __vec16_i1 mask) {
    int8_t *ptr = (int8_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val[i];
}

static FORCEINLINE void __masked_store_i16(void *p, __vec16_i16 val,
                                           __vec16_i1 mask) {
    int16_t *ptr = (int16_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val[i];
}

static FORCEINLINE void __masked_store_i32(void *p, const __vec16_i32 val, const __vec16_i1 mask) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  _mm512_mask_store_epi32(p, mask, val.v);
#else
  _mm512_mask_i32extscatter_epi32(p, mask, __ispc_stride1, val, _MM_DOWNCONV_EPI32_NONE, _MM_SCALE_4, _MM_HINT_NONE);
#endif
}

static FORCEINLINE void __masked_store_float(void *p, const __vec16_f val, const __vec16_i1 mask) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  _mm512_mask_store_ps(p, mask, val.v);
#else
  _mm512_mask_i32extscatter_ps(p, mask, __ispc_stride1, val, _MM_DOWNCONV_PS_NONE, _MM_SCALE_4, _MM_HINT_NONE);
#endif
}

static FORCEINLINE void __masked_store_i64(void *p, const __vec16_i64 val, const __vec16_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  __vec16_i1 tmp_m = mask;
  tmp_m = _mm512_kswapb(tmp_m, tmp_m);
  _mm512_mask_store_epi64(p, mask, val.v1);
  _mm512_mask_store_epi64((uint8_t*)p+64, tmp_m, val.v2);
#else
  _mm512_mask_i32loextscatter_epi64(          p,                      mask,  __ispc_stride1, val.v1, _MM_DOWNCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE);
  _mm512_mask_i32loextscatter_epi64((int64_t*)p+8, _mm512_kswapb(mask,mask), __ispc_stride1, val.v2, _MM_DOWNCONV_EPI64_NONE, _MM_SCALE_8, _MM_HINT_NONE);
#endif
}

static FORCEINLINE void __masked_store_double(void *p, const __vec16_d val, const __vec16_i1 mask) 
{
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  __vec16_i1 tmp_m = mask;
  tmp_m = _mm512_kswapb(tmp_m, tmp_m);
  _mm512_mask_store_pd(p, mask, val.v1);
  _mm512_mask_store_pd((uint8_t*)p+64, tmp_m, val.v2);
#else
  _mm512_mask_i32loextscatter_pd(           p,                    mask,  __ispc_stride1, val.v1, _MM_DOWNCONV_PD_NONE, _MM_SCALE_8, _MM_HINT_NONE);
  _mm512_mask_i32loextscatter_pd((double*)p+8, _mm512_kswapb(mask,mask), __ispc_stride1, val.v2, _MM_DOWNCONV_PD_NONE, _MM_SCALE_8, _MM_HINT_NONE);
#endif
}

static FORCEINLINE void __masked_store_blend_i8(void *p, __vec16_i8 val,
                                                __vec16_i1 mask) {
    __masked_store_i8(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i16(void *p, __vec16_i16 val,
                                                 __vec16_i1 mask) {
    __masked_store_i16(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i32(void *p, __vec16_i32 val,
                                                 __vec16_i1 mask) {
    __masked_store_i32(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_float(void *p, __vec16_f val,
                                                   __vec16_i1 mask) {
    __masked_store_float(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i64(void *p, __vec16_i64 val,
                                                 __vec16_i1 mask) {
    __masked_store_i64(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_double(void *p, __vec16_d val,
                                                    __vec16_i1 mask) {
    __masked_store_double(p, val, mask);
}

///////////////////////////////////////////////////////////////////////////
// gather/scatter
///////////////////////////////////////////////////////////////////////////

// offsets * offsetScale is in bytes (for all of these)

/* knc::macro::used */
#define GATHER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)                  \
static FORCEINLINE VTYPE FUNC(unsigned char *b, uint32_t scale,         \
                              OTYPE offset, __vec16_i1 mask) {          \
    VTYPE ret;                                                          \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 16; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * offset[i]);         \
            ret[i] = *ptr;                                            \
        }                                                               \
    return ret;                                                         \
}
    

/****************/
// GATHER_BASE_OFFSETS(__vec16_i8,  int8_t,  __vec16_i32, __gather_base_offsets32_i8)
static FORCEINLINE __vec16_i8 __gather_base_offsets32_i8(uint8_t *base, uint32_t scale, __vec16_i32 offsets,  __vec16_i1 mask) 
{
  // (iw): need to temporarily store as int because gathers can only return ints.
  __vec16_i32 tmp = _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask, offsets, base, 
                                                   _MM_UPCONV_EPI32_SINT8, scale,
                                                   _MM_HINT_NONE);
  // now, downconverting to chars into temporary char vector
  __vec16_i8 ret;
  _mm512_extstore_epi32(ret.data,tmp,_MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
  return ret;
}
// GATHER_BASE_OFFSETS(__vec16_i8,  int8_t,  __vec16_i64, __gather_base_offsets64_i8)
static FORCEINLINE __vec16_i8 __gather_base_offsets64_i8(uint8_t *_base, uint32_t scale, __vec16_i64 _offsets, __vec16_i1 mask) 
{ 
  const __vec16_i64 offsets = _offsets.cvt2hilo();
  const __vec16_i32 signed_offsets = _mm512_add_epi32(offsets.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  __vec16_i1 still_to_do = mask;
  __vec16_i32 tmp;
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((unsigned long)_base  +
        ((scale*(unsigned long)hi32) << 32) + scale*(unsigned long)(-(long)INT_MIN));
    tmp = _mm512_mask_i32extgather_epi32(tmp, match, signed_offsets, base,
        _MM_UPCONV_EPI32_SINT8, scale,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match,still_to_do);
  }
  __vec16_i8 ret;
  _mm512_extstore_epi32(ret.data,tmp,_MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
  return ret;
}
/****************/
GATHER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i32, __gather_base_offsets32_i16)
GATHER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i64, __gather_base_offsets64_i16)
/****************/
// GATHER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i32, __gather_base_offsets32_i32)
static FORCEINLINE __vec16_i32 __gather_base_offsets32_i32(uint8_t *base, uint32_t scale, __vec16_i32 offsets,   __vec16_i1 mask) 
{
  return _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask, offsets, 
                                        base, _MM_UPCONV_EPI32_NONE, scale,
                                        _MM_HINT_NONE);
}
// GATHER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i64, __gather_base_offsets64_i32)
static FORCEINLINE __vec16_i32 __gather_base_offsets64_i32(uint8_t *_base, uint32_t scale, __vec16_i64 _offsets,  __vec16_i1 mask) 
{
  const __vec16_i64 offsets = _offsets.cvt2hilo();
  const __vec16_i32 signed_offsets = _mm512_add_epi32(offsets.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  // There is no gather instruction with 64-bit offsets in KNC.
  // We have to manually iterate over the upper 32 bits ;-)
  __vec16_i1  still_to_do = mask;
  __vec16_i32 ret;
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);
         
    void * base = (void*)((unsigned long)_base  +
        ((scale*(unsigned long)hi32) << 32) + scale*(unsigned long)(-(long)INT_MIN));
    ret = _mm512_mask_i32extgather_epi32(ret, match, signed_offsets, base,
        _MM_UPCONV_EPI32_NONE, scale,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match, still_to_do);
  }

  return ret;
}
/****************/
// GATHER_BASE_OFFSETS(__vec16_f,   float,   __vec16_i32, __gather_base_offsets32_float)
static FORCEINLINE __vec16_f __gather_base_offsets32_float(uint8_t *base, uint32_t scale, __vec16_i32 offsets, __vec16_i1 mask) 
{
  return _mm512_mask_i32extgather_ps(_mm512_undefined_ps(), mask, offsets,
                                     base, _MM_UPCONV_PS_NONE, scale,
                                     _MM_HINT_NONE);
}
// GATHER_BASE_OFFSETS(__vec16_f,   float,   __vec16_i64, __gather_base_offsets64_float)
static FORCEINLINE __vec16_f __gather_base_offsets64_float(uint8_t *_base, uint32_t scale, __vec16_i64 _offsets,  __vec16_i1 mask) 
{
  const __vec16_i64 offsets = _offsets.cvt2hilo();
  const __vec16_i32 signed_offsets = _mm512_add_epi32(offsets.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  // There is no gather instruction with 64-bit offsets in KNC.
  // We have to manually iterate over the upper 32 bits ;-)
  __vec16_i1 still_to_do = mask;
  __vec16_f ret;
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((unsigned long)_base  +
        ((scale*(unsigned long)hi32) << 32) + scale*(unsigned long)(-(long)INT_MIN));
    ret = _mm512_mask_i32extgather_ps(ret, match, signed_offsets, base,
        _MM_UPCONV_PS_NONE, scale,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match, still_to_do);
  }

  return ret;
}
/****************/
GATHER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i32, __gather_base_offsets32_i64)
GATHER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i64, __gather_base_offsets64_i64)
/****************/
// GATHER_BASE_OFFSETS(__vec16_d,   double,  __vec16_i32, __gather_base_offsets32_double)
static FORCEINLINE __vec16_d __gather_base_offsets32_double(uint8_t *base, uint32_t scale, __vec16_i32 offsets, __vec16_i1 mask) 
{
  __vec16_d ret;
  ret.v1 = _mm512_mask_i32loextgather_pd(_mm512_undefined_pd(), mask, offsets,
                                         base, _MM_UPCONV_PD_NONE, scale,
                                         _MM_HINT_NONE); 
  __m512i shuffled_offsets = _mm512_permute4f128_epi32(offsets.v, _MM_PERM_DCDC);
  const __mmask8 mask8 = 0x00FF & (mask >> 8); /* knc::testme */
  ret.v2 = _mm512_mask_i32loextgather_pd(_mm512_undefined_pd(), mask8, shuffled_offsets,
                                         base, _MM_UPCONV_PD_NONE, scale,
                                         _MM_HINT_NONE); 
  return ret;
}
GATHER_BASE_OFFSETS(__vec16_d,   double,  __vec16_i64, __gather_base_offsets64_double)

/* knc::macro::used */
#define GATHER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)         \
static FORCEINLINE VTYPE FUNC(PTRTYPE ptrs, __vec16_i1 mask) {   \
    VTYPE ret;                                              \
    for (int i = 0; i < 16; ++i)                            \
        if ((mask.v & (1 << i)) != 0) {                     \
            STYPE *ptr = (STYPE *)ptrs[i];                \
            ret[i] = *ptr;                                \
        }                                                   \
    return ret;                                             \
}
/* knc::macro::used */
#define GATHER_GENERALF(VTYPE, STYPE, PTRTYPE, FUNC,FUNC1)         \
static FORCEINLINE VTYPE FUNC(PTRTYPE ptrs, __vec16_i1 mask) {   \
  return FUNC1(0, 1, ptrs, mask); \
}


/***********/
GATHER_GENERALF(__vec16_i8,  int8_t,  __vec16_i32, __gather32_i8, __gather_base_offsets32_i8)
GATHER_GENERALF(__vec16_i16, int16_t, __vec16_i32, __gather32_i16, __gather_base_offsets32_i16)
GATHER_GENERALF(__vec16_i32, int32_t, __vec16_i32, __gather32_i32, __gather_base_offsets32_i32)
GATHER_GENERALF(__vec16_i64, int64_t, __vec16_i32, __gather32_i64, __gather_base_offsets32_i64)
GATHER_GENERALF(__vec16_f,   float,   __vec16_i32, __gather32_float, __gather_base_offsets32_float)
GATHER_GENERALF(__vec16_d,   double,  __vec16_i32, __gather32_double, __gather_base_offsets32_double)
/***********/
GATHER_GENERAL(__vec16_i8,  int8_t,  __vec16_i64, __gather64_i8);
GATHER_GENERAL(__vec16_i16, int16_t, __vec16_i64, __gather64_i16);
GATHER_GENERAL(__vec16_i32, int32_t, __vec16_i64, __gather64_i32);
GATHER_GENERAL(__vec16_i64, int64_t, __vec16_i64, __gather64_i64);
GATHER_GENERAL(__vec16_f,   float,   __vec16_i64, __gather64_float);
GATHER_GENERAL(__vec16_d,   double,  __vec16_i64, __gather64_double);
/***********/

// scatter

/* knc::macro::used */
#define SCATTER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)                 \
static FORCEINLINE void FUNC(unsigned char *b, uint32_t scale,          \
                             OTYPE offset, VTYPE val,                   \
                             __vec16_i1 mask) {                         \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 16; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * offset[i]);         \
            *ptr = val[i];                                            \
        }                                                               \
}
    

/*****************/
SCATTER_BASE_OFFSETS(__vec16_i8,  int8_t,  __vec16_i32, __scatter_base_offsets32_i8)
SCATTER_BASE_OFFSETS(__vec16_i8,  int8_t,  __vec16_i64, __scatter_base_offsets64_i8)
/*****************/
SCATTER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i32, __scatter_base_offsets32_i16)
SCATTER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i64, __scatter_base_offsets64_i16)
/*****************/
// SCATTER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i32, __scatter_base_offsets32_i32)
static FORCEINLINE void __scatter_base_offsets32_i32(uint8_t *b, uint32_t scale, __vec16_i32 offsets,  __vec16_i32 val, __vec16_i1 mask)
{
  _mm512_mask_i32extscatter_epi32(b, mask, offsets, val, 
                                  _MM_DOWNCONV_EPI32_NONE, scale, 
                                  _MM_HINT_NONE);
}
// SCATTER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i64, __scatter_base_offsets64_i32)
static FORCEINLINE void __scatter_base_offsets64_i32(uint8_t *_base, uint32_t scale, __vec16_i64 _offsets, __vec16_i32 value, __vec16_i1 mask) 
{
  const __vec16_i64 offsets = _offsets.cvt2hilo();
  const __vec16_i32 signed_offsets = _mm512_add_epi32(offsets.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  
  __vec16_i1 still_to_do = mask;
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((unsigned long)_base  +
        ((scale*(unsigned long)hi32) << 32) + scale*(unsigned long)(-(long)INT_MIN));  
    _mm512_mask_i32extscatter_epi32(base, match, signed_offsets, 
        value,
        _MM_DOWNCONV_EPI32_NONE, scale,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match,still_to_do);
  }
}
/*****************/
// SCATTER_BASE_OFFSETS(__vec16_f,   float,   __vec16_i32, __scatter_base_offsets32_float)
static FORCEINLINE void __scatter_base_offsets32_float(void *base, uint32_t scale, __vec16_i32 offsets,
                               __vec16_f val, __vec16_i1 mask) 
{ 
  _mm512_mask_i32extscatter_ps(base, mask, offsets, val, 
                               _MM_DOWNCONV_PS_NONE, scale,
                               _MM_HINT_NONE);
}
//SCATTER_BASE_OFFSETS(__vec16_f,   float,   __vec16_i64, __scatter_base_offsets64_float)
static FORCEINLINE void __scatter_base_offsets64_float(uint8_t *_base, uint32_t scale, __vec16_i64 _offsets, __vec16_f value, __vec16_i1 mask) 
{ 
  const __vec16_i64 offsets = _offsets.cvt2hilo();
  const __vec16_i32 signed_offsets = _mm512_add_epi32(offsets.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  
  __vec16_i1 still_to_do = mask;
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((unsigned long)_base  +
        ((scale*(unsigned long)hi32) << 32) + scale*(unsigned long)(-(long)INT_MIN));   

    _mm512_mask_i32extscatter_ps(base, match, signed_offsets, 
        value,
        _MM_DOWNCONV_PS_NONE, scale,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match,still_to_do);
  }
}
/*****************/
SCATTER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i32, __scatter_base_offsets32_i64)
SCATTER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i64, __scatter_base_offsets64_i64)
/*****************/
// SCATTER_BASE_OFFSETS(__vec16_d,   double,  __vec16_i32, __scatter_base_offsets32_double)
static FORCEINLINE void __scatter_base_offsets32_double(void *base, uint32_t scale, __vec16_i32 offsets,
                               __vec16_d val, __vec16_i1 mask) 
{ 
  _mm512_mask_i32loextscatter_pd(base, mask, offsets, val.v1, 
                                 _MM_DOWNCONV_PD_NONE, scale,
                                 _MM_HINT_NONE);
  __m512i shuffled_offsets = _mm512_permute4f128_epi32(offsets.v, _MM_PERM_DCDC);
  const __mmask8 mask8 = 0x00FF & (mask >> 8); /* knc::testme */
  _mm512_mask_i32loextscatter_pd(base, mask8, shuffled_offsets, val.v2, 
                                 _MM_DOWNCONV_PD_NONE, scale,
                                 _MM_HINT_NONE);
}
SCATTER_BASE_OFFSETS(__vec16_d,   double,  __vec16_i64, __scatter_base_offsets64_double)

/* knc::macro::used */
#define SCATTER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)                 \
static FORCEINLINE void FUNC(PTRTYPE ptrs, VTYPE val, __vec16_i1 mask) {  \
    VTYPE ret;                                                       \
    for (int i = 0; i < 16; ++i)                                     \
        if ((mask.v & (1 << i)) != 0) {                              \
            STYPE *ptr = (STYPE *)ptrs[i];                         \
            *ptr = val[i];                                         \
        }                                                            \
}
/* knc::macro::used */
#define SCATTER_GENERALF(VTYPE, STYPE, PTRTYPE, FUNC,FUNC1)         \
static FORCEINLINE void FUNC(PTRTYPE ptrs, VTYPE val, __vec16_i1 mask) {  \
  return FUNC1(0, 1, ptrs, val, mask); \
}

/***********/
SCATTER_GENERALF(__vec16_i8,  int8_t,  __vec16_i32, __scatter32_i8, __scatter_base_offsets32_i8)
SCATTER_GENERALF(__vec16_i16, int16_t, __vec16_i32, __scatter32_i16, __scatter_base_offsets32_i16)
SCATTER_GENERALF(__vec16_i32, int32_t, __vec16_i32, __scatter32_i32, __scatter_base_offsets32_i32)
SCATTER_GENERALF(__vec16_i64, int64_t, __vec16_i32, __scatter32_i64, __scatter_base_offsets32_i64)
SCATTER_GENERALF(__vec16_f,   float,   __vec16_i32, __scatter32_float, __scatter_base_offsets32_float)
SCATTER_GENERALF(__vec16_d,   double,  __vec16_i32, __scatter32_double, __scatter_base_offsets32_double)
/***********/
SCATTER_GENERAL(__vec16_i8,  int8_t,  __vec16_i64, __scatter64_i8)
SCATTER_GENERAL(__vec16_i16, int16_t, __vec16_i64, __scatter64_i16)
SCATTER_GENERAL(__vec16_i32, int32_t, __vec16_i64, __scatter64_i32)
SCATTER_GENERAL(__vec16_f,   float,   __vec16_i64, __scatter64_float)
SCATTER_GENERAL(__vec16_i64, int64_t, __vec16_i64, __scatter64_i64)
SCATTER_GENERAL(__vec16_d,   double,  __vec16_i64, __scatter64_double)
/***********/

///////////////////////////////////////////////////////////////////////////
// packed load/store
///////////////////////////////////////////////////////////////////////////


static FORCEINLINE int32_t __packed_load_active(uint32_t *p, __vec16_i32 *val, __vec16_i1 mask)
{
  __vec16_i32 v = __load<64>(val);
  v = _mm512_mask_extloadunpacklo_epi32(v, mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v = _mm512_mask_extloadunpackhi_epi32(v, mask, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  __store<64>(val, v);
  return _mm_countbits_32(uint32_t(mask));
}

static FORCEINLINE int32_t __packed_store_active(uint32_t *p, __vec16_i32 val, __vec16_i1 mask) 
{
  _mm512_mask_extpackstorelo_epi32(p, mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_mask_extpackstorehi_epi32((uint8_t*)p+64, mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  return _mm_countbits_32(uint32_t(mask));
}

static FORCEINLINE int32_t __packed_store_active2(uint32_t *p, __vec16_i32 val, __vec16_i1 mask) 
{
  return __packed_store_active(p, val, mask);
}

static FORCEINLINE int32_t __packed_load_active(int32_t *p, __vec16_i32 *val, __vec16_i1 mask)
{
  return __packed_load_active((uint32_t *)p, val, mask);
}

static FORCEINLINE int32_t __packed_store_active(int32_t *p, __vec16_i32 val, __vec16_i1 mask) 
{
  return __packed_store_active((uint32_t *)p, val, mask);
}

static FORCEINLINE int32_t __packed_store_active2(int32_t *p, __vec16_i32 val, __vec16_i1 mask) 
{
  return __packed_store_active(p, val, mask);
}

///////////////////////////////////////////////////////////////////////////
// aos/soa
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE void __soa_to_aos3_float(__vec16_f v0, __vec16_f v1, __vec16_f v2,
                                            float *ptr) {
    for (int i = 0; i < 16; ++i) {
        *ptr++ = __extract_element(v0, i);
        *ptr++ = __extract_element(v1, i);
        *ptr++ = __extract_element(v2, i);
    }
}

static FORCEINLINE void __aos_to_soa3_float(float *ptr, __vec16_f *out0, __vec16_f *out1,
                                            __vec16_f *out2) {
    for (int i = 0; i < 16; ++i) {
        __insert_element(out0, i, *ptr++);
        __insert_element(out1, i, *ptr++);
        __insert_element(out2, i, *ptr++);
    }
}

static FORCEINLINE void __soa_to_aos4_float(__vec16_f v0, __vec16_f v1, __vec16_f v2,
                                            __vec16_f v3, float *ptr) {
    for (int i = 0; i < 16; ++i) {
        *ptr++ = __extract_element(v0, i);
        *ptr++ = __extract_element(v1, i);
        *ptr++ = __extract_element(v2, i);
        *ptr++ = __extract_element(v3, i);
    }
}

static FORCEINLINE void __aos_to_soa4_float(float *ptr, __vec16_f *out0, __vec16_f *out1,
                                            __vec16_f *out2, __vec16_f *out3) {
    for (int i = 0; i < 16; ++i) {
        __insert_element(out0, i, *ptr++);
        __insert_element(out1, i, *ptr++);
        __insert_element(out2, i, *ptr++);
        __insert_element(out3, i, *ptr++);
    }
}

///////////////////////////////////////////////////////////////////////////
// prefetch
///////////////////////////////////////////////////////////////////////////

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
///////////////////////////////////////////////////////////////////////////

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


///////////////////////////////////////////////////////////////////////////
// Transcendentals


#define TRANSCENDENTALS(op) \
static FORCEINLINE __vec16_f __##op##_varying_float(__vec16_f v) { return _mm512_##op##_ps(v); } \
static FORCEINLINE float __##op##_uniform_float(float v) { return op##f(v); } \
static FORCEINLINE __vec16_d __##op##_varying_double(__vec16_d v) { return __vec16_d(_mm512_##op##_pd(v.v1),_mm512_##op##_pd(v.v2)); } \
static FORCEINLINE double __##op##_uniform_double(double a) { return op(a); }

TRANSCENDENTALS(log)
TRANSCENDENTALS(exp)

static FORCEINLINE float __pow_uniform_float(float a, float b) {    return powf(a, b);}
static FORCEINLINE __vec16_f __pow_varying_float(__vec16_f a, __vec16_f b) { return _mm512_pow_ps(a,b); }
static FORCEINLINE double __pow_uniform_double(double a, double b) {    return pow(a,b);}
static FORCEINLINE __vec16_d __pow_varying_double(__vec16_d a, __vec16_d b) { return __vec16_d(_mm512_pow_pd(a.v1,b.v1),_mm512_pow_pd(a.v2,b.v2)); }

///////////////////////////////////////////////////////////////////////////
// Trigonometry

TRANSCENDENTALS(sin)
TRANSCENDENTALS(asin)
TRANSCENDENTALS(cos)
TRANSCENDENTALS(acos)
TRANSCENDENTALS(tan)
TRANSCENDENTALS(atan)

static FORCEINLINE float __atan2_uniform_float(float a, float b) {    return atan2f(a, b);}
static FORCEINLINE __vec16_f __atan2_varying_float(__vec16_f a, __vec16_f b) { return _mm512_atan2_ps(a,b); }
static FORCEINLINE double __atan2_uniform_double(double a, double b) {    return atan2(a,b);}
static FORCEINLINE __vec16_d __atan2_varying_double(__vec16_d a, __vec16_d b) { return __vec16_d(_mm512_atan2_pd(a.v1,b.v1),_mm512_atan2_pd(a.v2,b.v2)); }

#undef FORCEINLINE
#undef PRE_ALIGN
#undef POST_ALIGN
