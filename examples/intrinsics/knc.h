/**
  Copyright (c) 2010-2015, Intel Corporation
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
#include <unistd.h>
#include <algorithm>

#ifndef __INTEL_COMPILER
#error "Only Intel(R) C++ Compiler is supported"
#endif

#include <immintrin.h>
#include <zmmintrin.h>

#include <iostream> // for operator<<(m512[i])
#include <iomanip>  // for operator<<(m512[i])

#if __INTEL_COMPILER < 1500
#warning "Your compiler version is outdated which can reduce performance in some cases. Please, update your compiler!"
#endif


#if 0
  #define STRING(x) #x
  #define TOSTRING(x) STRING(x)
  #define PING std::cout << __FILE__ << " (" << __LINE__ << "): " << __FUNCTION__ << std::endl
  #define PRINT(x) std::cout << STRING(x) << " = " << (x) << std::endl
  #define PRINT2(x,y) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << std::endl
  #define PRINT3(x,y,z) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << ", " << STRING(z) << " = " << (z) << std::endl
  #define PRINT4(x,y,z,w) std::cout << STRING(x) << " = " << (x) << ", " << STRING(y) << " = " << (y) << ", " << STRING(z) << " = " << (z) << ", " << STRING(w) << " = " << (w) << std::endl
#endif

#define FORCEINLINE __forceinline
#ifdef _MSC_VER
#define PRE_ALIGN(x)  /*__declspec(align(x))*/
#define POST_ALIGN(x)  
#define roundf(x) (floorf(x + .5f))
#define round(x) (floor(x + .5))
#else
#define PRE_ALIGN(x)
#define POST_ALIGN(x)  __attribute__ ((aligned(x)))
#endif

#define KNC 1
extern "C" {
  int printf(const unsigned char *, ...);
  int puts(unsigned char *);
  unsigned int putchar(unsigned int);
  int fflush(void *);
  uint8_t *memcpy(uint8_t *, uint8_t *, uint64_t);
  uint8_t *memset(uint8_t *, uint8_t, uint64_t);
  void memset_pattern16(void *, const void *, uint64_t);
}

typedef float   __vec1_f;
typedef double  __vec1_d;
typedef int8_t  __vec1_i8;
typedef int16_t __vec1_i16;
typedef int32_t __vec1_i32;
typedef int64_t __vec1_i64;

struct __vec16_i32;

#if 1
/* (iw) actually, this *SHOULD* be the right implementation for a
vec16_i1: this one is a class that can have a constructor (which
ISPC sometimes emits for these vectors...) This version might 
not be working with embree's ISPC bindings, probably because 
embree still uses the 'wrong' implementation */
typedef struct PRE_ALIGN(2) __vec16_i1
{
  FORCEINLINE operator __mmask16() const { return v; }
  FORCEINLINE __vec16_i1() { }
  FORCEINLINE __vec16_i1(const __mmask16 &vv) : v(vv) { }
  FORCEINLINE __vec16_i1(bool v0, bool v1, bool v2, bool v3,
                         bool v4, bool v5, bool v6, bool v7,
                         bool v8, bool v9, bool v10, bool v11,
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
  __mmask16 v;
} POST_ALIGN(2) __vec16_i1;

#else
typedef __mmask16 POST_ALIGN(2) __vec16_i1;
#endif

typedef struct PRE_ALIGN(64) __vec16_f {
  FORCEINLINE operator __m512() const { return v; }
  FORCEINLINE __vec16_f() : v(_mm512_undefined_ps()) { }
  FORCEINLINE __vec16_f(const __m512 &in) : v(in) {}
  FORCEINLINE __vec16_f(const __vec16_f &o) : v(o.v) {}
  FORCEINLINE __vec16_f& operator =(const __vec16_f &o) { v=o.v; return *this; }
  FORCEINLINE __vec16_f(float v00, float v01, float v02, float v03, 
                        float v04, float v05, float v06, float v07,
                        float v08, float v09, float v10, float v11,
                        float v12, float v13, float v14, float v15) {
    v = _mm512_set_16to16_ps(v15, v14, v13, v12, v11, v10, v09, v08, v07, v06, v05, v04, v03, v02, v01, v00);
  }
  FORCEINLINE const float& operator[](const int i) const {  return ((float*)this)[i]; }
  FORCEINLINE       float& operator[](const int i)       {  return ((float*)this)[i]; }
  __m512 v;
} POST_ALIGN(64) __vec16_f;

typedef struct PRE_ALIGN(64) __vec16_d {
  FORCEINLINE __vec16_d() : v1(_mm512_undefined_pd()), v2(_mm512_undefined_pd()) {}
  FORCEINLINE __vec16_d(const __vec16_d &o) : v1(o.v1), v2(o.v2) {}
  FORCEINLINE __vec16_d(const __m512d _v1, const __m512d _v2) : v1(_v1), v2(_v2) {}
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
  __m512d v1;
  __m512d v2;
} POST_ALIGN(64) __vec16_d;

typedef struct PRE_ALIGN(64) __vec16_i32 {
  FORCEINLINE operator __m512i() const { return v; }
  FORCEINLINE __vec16_i32() : v(_mm512_undefined_epi32()) {}
  FORCEINLINE __vec16_i32(const int32_t &in) : v(_mm512_set1_epi32(in)) {}
  FORCEINLINE __vec16_i32(const __m512i &in) : v(in) {}
  FORCEINLINE __vec16_i32(const __vec16_i32 &o) : v(o.v) {}
  FORCEINLINE __vec16_i32& operator =(const __vec16_i32 &o) { v=o.v; return *this; }
  FORCEINLINE __vec16_i32(int32_t v00, int32_t v01, int32_t v02, int32_t v03, 
                          int32_t v04, int32_t v05, int32_t v06, int32_t v07,
                          int32_t v08, int32_t v09, int32_t v10, int32_t v11,
                          int32_t v12, int32_t v13, int32_t v14, int32_t v15) {
    v = _mm512_set_16to16_pi(v15, v14, v13, v12, v11, v10, v09, v08, v07, v06, v05, v04, v03, v02, v01, v00);
  }
  FORCEINLINE const int32_t& operator[](const int i) const {  return ((int32_t*)this)[i]; }
  FORCEINLINE       int32_t& operator[](const int i)       {  return ((int32_t*)this)[i]; }
  __m512i v;
} POST_ALIGN(64) __vec16_i32;

typedef struct PRE_ALIGN(64) __vec16_i64 {
  FORCEINLINE __vec16_i64() : v_lo(_mm512_undefined_epi32()),  v_hi(_mm512_undefined_epi32()) {}
  FORCEINLINE __vec16_i64(const __vec16_i64 &o) : v_lo(o.v_lo), v_hi(o.v_hi) {}
  FORCEINLINE __vec16_i64(__m512i l, __m512i h) : v_lo(l), v_hi(h) {}
  FORCEINLINE __vec16_i64& operator =(const __vec16_i64 &o) { v_lo=o.v_lo; v_hi=o.v_hi; return *this; }
  FORCEINLINE __vec16_i64(int64_t v00, int64_t v01, int64_t v02, int64_t v03, 
                          int64_t v04, int64_t v05, int64_t v06, int64_t v07,
                          int64_t v08, int64_t v09, int64_t v10, int64_t v11,
                          int64_t v12, int64_t v13, int64_t v14, int64_t v15) {
    __m512i v1 = _mm512_set_8to8_epi64(v15, v14, v13, v12, v11, v10, v09, v08);
    __m512i v2 = _mm512_set_8to8_epi64(v07, v06, v05, v04, v03, v02, v01, v00);
    v_hi = _mm512_mask_permutevar_epi32(v_hi, 0xFF00, 
                  _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
                  v1);
    v_hi = _mm512_mask_permutevar_epi32(v_hi, 0x00FF, 
                  _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
                  v2);
    v_lo = _mm512_mask_permutevar_epi32(v_lo, 0xFF00,
                  _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
                  v1);
    v_lo = _mm512_mask_permutevar_epi32(v_lo, 0x00FF,
                  _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
                  v2);
  } 
  FORCEINLINE       int64_t operator[](const int i) const {
      return ((uint64_t(((int32_t*)this)[i])<<32)+((int32_t*)this)[i+16]); }
  FORCEINLINE       int64_t operator[](const int i)       {
      return ((uint64_t(((int32_t*)this)[i])<<32)+((int32_t*)this)[i+16]); }
  __m512i v_hi;
  __m512i v_lo;
} POST_ALIGN(64) __vec16_i64;

FORCEINLINE __vec16_i64 zmm2hilo(const __m512i v1, const __m512i v2){
  __vec16_i64 v;
  v.v_hi = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xFF00,
                  _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
                  v2);
  v.v_hi = _mm512_mask_permutevar_epi32(v.v_hi, 0x00FF,
                  _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
                  v1);
  v.v_lo = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xFF00,
                  _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
                  v2);
  v.v_lo = _mm512_mask_permutevar_epi32(v.v_lo, 0x00FF,
                  _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
                  v1);
  return v;
}

FORCEINLINE void hilo2zmm(const __vec16_i64 &v, __m512i &_v1, __m512i &_v2) {
  _v2 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
               _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
               v.v_hi);
  _v2 = _mm512_mask_permutevar_epi32(_v2, 0x5555,
               _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
               v.v_lo);
  _v1 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
               _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
               v.v_hi);
  _v1 = _mm512_mask_permutevar_epi32(_v1, 0x5555,
               _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
               v.v_lo);
}

FORCEINLINE __vec16_i64 hilo2zmm(const __vec16_i64 &v) {
  __vec16_i64 ret;
  hilo2zmm(v, ret.v_hi, ret.v_lo);
  return ret;
}


template <typename T>
struct vec16 {
  FORCEINLINE vec16() { }
  FORCEINLINE vec16(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
      T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) {
    v[0] = v0;        v[1] = v1;        v[2] = v2;        v[3] = v3;
    v[4] = v4;        v[5] = v5;        v[6] = v6;        v[7] = v7;
    v[8] = v8;        v[9] = v9;        v[10] = v10;      v[11] = v11;
    v[12] = v12;      v[13] = v13;      v[14] = v14;      v[15] = v15;
  }
  FORCEINLINE const T& operator[](const int i) const { return v[i]; }
  FORCEINLINE       T& operator[](const int i)       { return v[i]; }
  T v[16]; 
};

PRE_ALIGN(16) struct __vec16_i8   : public vec16<int8_t> { 
  FORCEINLINE __vec16_i8() { }
  FORCEINLINE __vec16_i8(const int8_t  v0,  const int8_t  v1,  const int8_t  v2,  const int8_t  v3, 
                         const int8_t  v4,  const int8_t  v5,  const int8_t  v6,  const int8_t  v7,
                         const int8_t  v8,  const int8_t  v9,  const int8_t  v10, const int8_t  v11, 
                         const int8_t  v12, const int8_t  v13, const int8_t  v14, const int8_t  v15)
    : vec16<int8_t>(v0, v1, v2, v3, v4, v5, v6, v7,
                    v8, v9, v10, v11, v12, v13, v14, v15) { }
  FORCEINLINE __vec16_i8(const __vec16_i8 &o);
  FORCEINLINE __vec16_i8& operator =(const __vec16_i8 &o);
} POST_ALIGN(16);

PRE_ALIGN(32) struct __vec16_i16  : public vec16<int16_t> { 
  FORCEINLINE __vec16_i16() { }
  FORCEINLINE __vec16_i16(const __vec16_i16 &o);
  FORCEINLINE __vec16_i16& operator =(const __vec16_i16 &o);
  FORCEINLINE __vec16_i16(int16_t v0,  int16_t v1,  int16_t v2,  int16_t v3, 
                          int16_t v4,  int16_t v5,  int16_t v6,  int16_t v7,
                          int16_t v8,  int16_t v9,  int16_t v10, int16_t v11, 
                          int16_t v12, int16_t v13, int16_t v14, int16_t v15) 
    : vec16<int16_t>(v0, v1, v2, v3, v4, v5, v6, v7,
                     v8, v9, v10, v11, v12, v13, v14, v15) { }
} POST_ALIGN(32);



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
  uint32_t *ptr = (uint32_t*)&v;
  for (int i=0;i<16;i++) {
    uint64_t val = (uint64_t(ptr[i])<<32)+ptr[i+16];
    out << (i!=0?",":"") << std::dec << std::setw(8) << ((uint64_t)val) << std::dec;
  }  
  out << "]" << std::flush;
  return out;
}

///////////////////////////////////////////////////////////////////////////
// macros...

FORCEINLINE __vec16_i8::__vec16_i8(const __vec16_i8 &o)
{
  for (int i=0;i<16;i++)
    v[i] = o.v[i];
}

FORCEINLINE __vec16_i8& __vec16_i8::operator=(const __vec16_i8 &o)
{
  for (int i=0;i<16;i++)
    v[i] = o.v[i];
  return *this;
}

FORCEINLINE __vec16_i16::__vec16_i16(const __vec16_i16 &o)
{
  for (int i=0;i<16;i++)
    v[i] = o.v[i];
}

FORCEINLINE __vec16_i16& __vec16_i16::operator=(const __vec16_i16 &o)
{
  for (int i=0;i<16;i++)
    v[i] = o.v[i];
  return *this;
}
///////////////////////////////////////////////////////////////////////////
// mask ops
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_i1 __movmsk(__vec16_i1 mask) {
  return _mm512_kmov(mask);
}

static FORCEINLINE bool __any(__vec16_i1 mask) {
  return !_mm512_kortestz(mask, mask);
}

static FORCEINLINE bool __all(__vec16_i1 mask) {
  return _mm512_kortestc(mask, mask);
}

static FORCEINLINE bool __none(__vec16_i1 mask) {
  return _mm512_kortestz(mask, mask);
}

static FORCEINLINE __vec16_i1 __equal_i1(__vec16_i1 a, __vec16_i1 b) {
  return _mm512_knot( _mm512_kxor(a, b));
}

static FORCEINLINE __vec16_i1 __and(__vec16_i1 a, __vec16_i1 b) {
  return _mm512_kand(a, b);
}

static FORCEINLINE __vec16_i1 __not(__vec16_i1 a) {
  return _mm512_knot(a);
}

static FORCEINLINE __vec16_i1 __and_not1(__vec16_i1 a, __vec16_i1 b) {
  return _mm512_kandn(a, b);
}

static FORCEINLINE __vec16_i1 __and_not2(__vec16_i1 a, __vec16_i1 b) {
  return _mm512_kandnr(a, b);
}

static FORCEINLINE __vec16_i1 __xor(__vec16_i1 a, __vec16_i1 b) {
  return _mm512_kxor(a, b);
}

static FORCEINLINE __vec16_i1 __xnor(__vec16_i1 a, __vec16_i1 b) {
  return _mm512_kxnor(a, b);
}

static FORCEINLINE __vec16_i1 __or(__vec16_i1 a, __vec16_i1 b) {
  return _mm512_kor(a, b);
}

static FORCEINLINE __vec16_i1 __select(__vec16_i1 mask, __vec16_i1 a, 
    __vec16_i1 b) {
  return ((a & mask) | (b & ~mask));
  //return __or(__and(a, mask), __andnr(b, mask));
}

static FORCEINLINE __vec16_i1 __select(bool cond, __vec16_i1 a, __vec16_i1 b) {
  return cond ? a : b;
}


static FORCEINLINE bool __extract_element(__vec16_i1 mask, uint32_t index) {
  return (mask & (1 << index)) ? true : false;
}


static FORCEINLINE int64_t __extract_element(const __vec16_i64 &v, uint32_t index)
{
  //uint *src = (uint *)&v;
  const uint *src = (const uint *)&v;
  return src[index+16] | (uint64_t(src[index]) << 32);
}







/*
   static FORCEINLINE void __insert_element(__vec16_i1 *vec, int index, 
   bool val) {
   if (val == false)
   vec->v &= ~(1 << index);
   else
   vec->v |= (1 << index);
   }
   */

template <int ALIGN> static FORCEINLINE __vec16_i1 __load(const __vec16_i1 *p) {
  const uint16_t *ptr = (const uint16_t *)p;
  __vec16_i1 r;
  r = *ptr;
  return r;
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_i1 *p, __vec16_i1 v) {
  uint16_t *ptr = (uint16_t *)p;
  *ptr = v;
}

template <class RetVecType> static RetVecType __smear_i1(int i);
template <> FORCEINLINE __vec16_i1 __smear_i1<__vec16_i1>(int i) {
  return i?0xFFFF:0x0;
}

template <class RetVecType> static RetVecType __setzero_i1();
template <> FORCEINLINE __vec16_i1 __setzero_i1<__vec16_i1>() {
  return 0;
}

template <class RetVecType> static RetVecType __undef_i1();
template <> FORCEINLINE __vec16_i1 __undef_i1<__vec16_i1>() {
  return __vec16_i1();
}


///////////////////////////////////////////////////////////////////////////
// int32
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_i32 __add(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_add_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __sub(__vec16_i32 a, __vec16_i32 b) { 
  return _mm512_sub_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __mul(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_mullo_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __udiv(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_div_epu32(a, b);
}

static FORCEINLINE __vec16_i32 __sdiv(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_div_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __urem(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_rem_epu32(a, b);
}

static FORCEINLINE __vec16_i32 __srem(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_rem_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __or(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_or_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __and(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_and_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __xor(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_xor_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __shl(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_sllv_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __lshr(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_srlv_epi32(a, b);
}

static FORCEINLINE __vec16_i32 __ashr(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_srav_epi32(a, b); 
}

static FORCEINLINE __vec16_i32 __shl(__vec16_i32 a, int32_t n) {
  return _mm512_slli_epi32(a, n);
}

static FORCEINLINE __vec16_i32 __lshr(__vec16_i32 a, int32_t n) {
  return _mm512_srli_epi32(a, n); 
}

static FORCEINLINE __vec16_i32 __ashr(__vec16_i32 a, int32_t n) {
  return _mm512_srai_epi32(a, n); 
}

static FORCEINLINE __vec16_i1 __equal_i32(const __vec16_i32 &a, const __vec16_i32 &b) {
  return _mm512_cmpeq_epi32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __equal_i32_and_mask(const __vec16_i32 &a, const __vec16_i32 &b,
    __vec16_i1 m) {
  return _mm512_mask_cmpeq_epi32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __not_equal_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmpneq_epi32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __not_equal_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmpneq_epi32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __unsigned_less_equal_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmple_epu32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __unsigned_less_equal_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmple_epu32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __signed_less_equal_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmple_epi32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __signed_less_equal_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmple_epi32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __unsigned_greater_equal_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmpge_epu32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __unsigned_greater_equal_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmpge_epu32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __signed_greater_equal_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmpge_epi32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __signed_greater_equal_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmpge_epi32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __unsigned_less_than_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmplt_epu32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __unsigned_less_than_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmplt_epu32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __signed_less_than_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmplt_epi32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __signed_less_than_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmplt_epi32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __unsigned_greater_than_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmpgt_epu32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __unsigned_greater_than_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmpgt_epu32_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __signed_greater_than_i32(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_cmpgt_epi32_mask(a, b);
}

static FORCEINLINE __vec16_i1 __signed_greater_than_i32_and_mask(__vec16_i32 a, __vec16_i32 b,
    __vec16_i1 m) {
  return _mm512_mask_cmpgt_epi32_mask(m, a, b);
}

static FORCEINLINE __vec16_i32 __select(__vec16_i1 mask,
    __vec16_i32 a, __vec16_i32 b) {
  return _mm512_mask_mov_epi32(b.v, mask, a.v);
} 

static FORCEINLINE __vec16_i32 __select(bool cond, __vec16_i32 a, __vec16_i32 b) {
  return cond ? a : b;
}

static FORCEINLINE int32_t __extract_element(__vec16_i32 v, uint32_t index) {
  return ((int32_t *)&v)[index];
}

static FORCEINLINE void __insert_element(__vec16_i32 *v, uint32_t index, int32_t val) {
  ((int32_t *)v)[index] = val;
}

template <class RetVecType> static RetVecType __smear_i32(int32_t i);
template <> FORCEINLINE __vec16_i32 __smear_i32<__vec16_i32>(int32_t i) {
  return _mm512_set1_epi32(i);
}

static const __vec16_i32 __ispc_one = __smear_i32<__vec16_i32>(1);
static const __vec16_i32 __ispc_zero = __smear_i32<__vec16_i32>(0);
static const __vec16_i32 __ispc_thirty_two = __smear_i32<__vec16_i32>(32);
static const __vec16_i32 __ispc_ffffffff = __smear_i32<__vec16_i32>(-1);
static const __vec16_i32 __ispc_stride1(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

template <class RetVecType> static RetVecType __setzero_i32();
template <> FORCEINLINE __vec16_i32 __setzero_i32<__vec16_i32>() {
  return _mm512_setzero_epi32();
}

template <class RetVecType> static RetVecType __undef_i32();
template <> FORCEINLINE __vec16_i32 __undef_i32<__vec16_i32>() {
  return __vec16_i32();
}

static FORCEINLINE __vec16_i32 __broadcast_i32(__vec16_i32 v, int index) {
  int32_t val = __extract_element(v, index & 0xf);
  return _mm512_set1_epi32(val);
}

static FORCEINLINE __vec16_i8 __cast_trunc(__vec16_i8, const __vec16_i16 i16) {
  return __vec16_i8((uint8_t)i16[0],  (uint8_t)i16[1],  (uint8_t)i16[2],  (uint8_t)i16[3],
                    (uint8_t)i16[4],  (uint8_t)i16[5],  (uint8_t)i16[6],  (uint8_t)i16[7],
                    (uint8_t)i16[8],  (uint8_t)i16[9],  (uint8_t)i16[10], (uint8_t)i16[11],
                    (uint8_t)i16[12], (uint8_t)i16[13], (uint8_t)i16[14], (uint8_t)i16[15]);
}

static FORCEINLINE __vec16_i16 __cast_trunc(__vec16_i16, const __vec16_i32 i32) {
  __vec16_i16 ret;
  __vec16_i32 i32_trunk = _mm512_and_epi32(i32, __smear_i32<__vec16_i32>(65535));
  _mm512_extstore_epi32(ret.v, i32_trunk, _MM_DOWNCONV_EPI32_UINT16, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i8 __cast_trunc(__vec16_i8, const __vec16_i32 i32) {
  __vec16_i8 ret;
  __vec16_i32 i32_trunk = _mm512_and_epi32(i32, __smear_i32<__vec16_i32>(255));
  _mm512_extstore_epi32(ret.v, i32, _MM_DOWNCONV_EPI32_UINT8, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i32 __cast_trunc(__vec16_i32, const __vec16_i64 i64) {
  return __vec16_i32(i64.v_lo);
}

static FORCEINLINE __vec16_i16 __cast_trunc(__vec16_i16, const __vec16_i64 i64) {
  return __cast_trunc(__vec16_i16(), i64.v_lo);
}

static FORCEINLINE __vec16_i8 __cast_trunc(__vec16_i8, const __vec16_i64 i64) {
  return __cast_trunc(__vec16_i8(), i64.v_lo);
}

static FORCEINLINE __vec16_i32 __rotate_i32(__vec16_i32 v, int index) {
  __vec16_i32 idx = __smear_i32<__vec16_i32>(index);
  __vec16_i32 shuffle = _mm512_and_epi32(_mm512_add_epi32(__ispc_stride1, idx),  __smear_i32<__vec16_i32>(0xf));
  return _mm512_mask_permutevar_epi32(v, 0xffff, shuffle, v);
}

static FORCEINLINE __vec16_i32 __shuffle_i32(__vec16_i32 v, __vec16_i32 index) {
  return _mm512_mask_permutevar_epi32(v, 0xffff, index, v);
}

static FORCEINLINE __vec16_i32 __shuffle2_i32(__vec16_i32 v0, __vec16_i32 v1, __vec16_i32 index) {
    const __vec16_i1 mask = __signed_less_than_i32(index, __smear_i32<__vec16_i32>(0x10));
    index = __and(index, __smear_i32<__vec16_i32>(0xF));
    __vec16_i32 ret = __undef_i32<__vec16_i32>();
    ret = _mm512_mask_permutevar_epi32(ret, mask, index, v0);
    ret = _mm512_mask_permutevar_epi32(ret, __not(mask), index, v1);
    return ret;
}

static FORCEINLINE __vec16_i32 __shift_i32(__vec16_i32 v, int index) {
  __vec16_i32 mod_index = _mm512_add_epi32(__ispc_stride1, __smear_i32<__vec16_i32>(index));
  __vec16_i1 mask_ge = _mm512_cmpge_epi32_mask (mod_index, __smear_i32<__vec16_i32>(0));
  __vec16_i1 mask_le = _mm512_cmple_epi32_mask (mod_index, __smear_i32<__vec16_i32>(0xF));
  __vec16_i1 mask = mask_ge & mask_le;
  __vec16_i32 ret = __smear_i32<__vec16_i32>(0);
  ret = _mm512_mask_permutevar_epi32(ret, mask, mod_index, v);
  return ret;
}

template <int ALIGN> static FORCEINLINE __vec16_i32 __load(const __vec16_i32 *p) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  return _mm512_load_epi32(p);
#else
  __vec16_i32 v;
  v = _mm512_extloadunpacklo_epi32(v, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v = _mm512_extloadunpackhi_epi32(v, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  return v;
#endif
}

#if 0
template <> FORCEINLINE __vec16_i32 __load<64>(const __vec16_i32 *p) {
  return _mm512_load_epi32(p);
}
#endif

template <int ALIGN> static FORCEINLINE void __store(__vec16_i32 *p, __vec16_i32 v) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  _mm512_store_epi32(p, v);
#else
  _mm512_extpackstorelo_epi32(p, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+64, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
#endif
}

#if 0
template <> FORCEINLINE void __store<64>(__vec16_i32 *p, __vec16_i32 v) {
  _mm512_store_epi32(p, v);
}
#endif

///////////////////////////////////////////////////////////////////////////
// int64
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_i64 __select(__vec16_i1 mask,
  __vec16_i64 a, __vec16_i64 b) {
  __vec16_i64 ret;
  ret.v_hi = _mm512_mask_mov_epi32(b.v_hi, mask, a.v_hi);
  ret.v_lo = _mm512_mask_mov_epi32(b.v_lo, mask, a.v_lo);
  return ret;
}

static FORCEINLINE 
void __masked_store_i64(void *p, const __vec16_i64 &v, __vec16_i1 mask) 
{
  __m512i v1;
  __m512i v2;
  v1 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
      _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
      v.v_hi);
  v1 = _mm512_mask_permutevar_epi32(v1, 0x5555,
      _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
      v.v_lo);
  v2 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
      _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
      v.v_hi);
  v2 = _mm512_mask_permutevar_epi32(v2, 0x5555,
      _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
      v.v_lo);
  _mm512_mask_store_epi64(p, mask, v2);
  _mm512_mask_store_epi64(((uint8_t*)p)+64, mask>>8, v1);
}

static FORCEINLINE void __insert_element(__vec16_i64 *v, uint32_t index, int64_t val) {
  ((int32_t *)&v->v_hi)[index] = val>>32;
  ((int32_t *)&v->v_lo)[index] = val;
}


template <class RetVecType> static RetVecType __setzero_i64();
template <> FORCEINLINE __vec16_i64 __setzero_i64<__vec16_i64>() {
  __vec16_i64 ret;
  ret.v_lo = _mm512_setzero_epi32();
  ret.v_hi = _mm512_setzero_epi32();
  return ret;
}

template <class RetVecType> static RetVecType __undef_i64();
template <> FORCEINLINE __vec16_i64 __undef_i64<__vec16_i64>() {
  return __vec16_i64();
}

static FORCEINLINE __vec16_i64 __add(const __vec16_i64 &a, const __vec16_i64 &b)
{    
  __mmask16 carry = 0;
  __m512i lo = _mm512_addsetc_epi32(a.v_lo, b.v_lo, &carry);
  __m512i hi = _mm512_adc_epi32(a.v_hi, carry, b.v_hi, &carry);
  return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i64 __sub(const __vec16_i64 &a, const __vec16_i64 &b)
{    
  __mmask16 borrow = 0;
  __m512i lo = _mm512_subsetb_epi32(a.v_lo, b.v_lo, &borrow);
  __m512i hi = _mm512_sbb_epi32(a.v_hi, borrow, b.v_hi, &borrow);
  return __vec16_i64(lo, hi);
}

/*! 64x32 bit mul -- address computations often use a scale that we
  know is 32 bits; and 32x64 is faster than 64x64 */
static FORCEINLINE __vec16_i64 __mul(const __vec16_i32 &a, const __vec16_i64 &b)
{
  return __vec16_i64(_mm512_mullo_epi32(a.v,b.v_lo),
      _mm512_add_epi32(_mm512_mullo_epi32(a.v, b.v_hi),
        _mm512_mulhi_epi32(a.v, b.v_lo)));
}

static FORCEINLINE void __abs_i32i64(__m512i &_hi, __m512i &_lo)
{
  /*   abs(x) : 
   * mask  = x >> 64; // sign bits
   * abs(x) = (x^mask) - mask
   */ 
  const __vec16_i32 mask = __ashr(_hi, __ispc_thirty_two);
  __vec16_i32 hi = __xor(_hi, mask);
  __vec16_i32 lo = __xor(_lo, mask);
  __mmask16 borrow = 0;
  _lo = _mm512_subsetb_epi32(lo, mask, &borrow);
  _hi = _mm512_sbb_epi32    (hi, borrow, mask, &borrow);
}

static FORCEINLINE __vec16_i64 __mul(__vec16_i64 a, __vec16_i64 b)
{
  const __vec16_i1 sign = __not_equal_i32(__ashr(__xor(a.v_hi, b.v_hi), __ispc_thirty_two), __ispc_zero);
  __abs_i32i64(a.v_hi, a.v_lo);  /* abs(a) */
  __abs_i32i64(b.v_hi, b.v_lo);  /* abs(b) */
  __vec16_i32 lo =    _mm512_mullo_epi32(a.v_lo, b.v_lo);
  __vec16_i32 hi_m1 = _mm512_mulhi_epu32(a.v_lo, b.v_lo);
  __vec16_i32 hi_m2 = _mm512_mullo_epi32(a.v_hi, b.v_lo);
  __vec16_i32 hi_m3 = _mm512_mullo_epi32(a.v_lo, b.v_hi);
  
  __mmask16 carry = 0;
  __vec16_i32 hi_p23 = _mm512_addsetc_epi32(hi_m2, hi_m3, &carry);
  __vec16_i32 hi = _mm512_adc_epi32(hi_p23, carry, hi_m1, &carry);

  __vec16_i64 ret_abs(lo, hi);
  return __select(sign, __sub(__vec16_i64(__ispc_zero, __ispc_zero), ret_abs), ret_abs);
}

static FORCEINLINE __vec16_i64 __sdiv(const __vec16_i64 &a, const __vec16_i64 &b)
{
  __vec16_i64 ret;
  for(int i=0; i<16; i++) {
    int64_t dividend = __extract_element(a, i);
    int64_t divisor = __extract_element(b, i);
    int64_t quotient = dividend / divisor; // SVML
    __insert_element(&ret, i, quotient);
  }
  return ret;
}

static FORCEINLINE __vec16_i64 __udiv(const __vec16_i64 &a, const __vec16_i64 &b)
{
  __vec16_i64 ret;
  for(int i=0; i<16; i++) {
    uint64_t dividend = __extract_element(a, i);
    uint64_t divisor = __extract_element(b, i);
    uint64_t quotient = dividend / divisor; // SVML
    __insert_element(&ret, i, quotient);
  }
  return ret;
}

static FORCEINLINE __vec16_i64 __or(__vec16_i64 a, __vec16_i64 b) {
  return __vec16_i64(_mm512_or_epi32(a.v_lo, b.v_lo), _mm512_or_epi32(a.v_hi, b.v_hi));
}

static FORCEINLINE __vec16_i64 __and(__vec16_i64 a, __vec16_i64 b) {
  return __vec16_i64(_mm512_and_epi32(a.v_lo, b.v_lo), _mm512_and_epi32(a.v_hi, b.v_hi));
}

static FORCEINLINE __vec16_i64 __xor(__vec16_i64 a, __vec16_i64 b) {
  return __vec16_i64(_mm512_xor_epi32(a.v_lo, b.v_lo), _mm512_xor_epi32(a.v_hi, b.v_hi));
}

static FORCEINLINE __vec16_i64 __shl(__vec16_i64 a, __vec16_i64 b) {
  /* this is a safety gate in case b-shift >= 32 */
  const __vec16_i32 xfer = __select(
    __signed_less_than_i32(b.v_lo, __ispc_thirty_two), 
    __lshr(a.v_lo, __sub(__ispc_thirty_two, b.v_lo)),
    __shl (a.v_lo, __sub(b.v_lo, __ispc_thirty_two))
    );
  const __vec16_i32 hi = __or(__shl(a.v_hi, b.v_lo), xfer);
  const __vec16_i32 lo =      __shl(a.v_lo, b.v_lo); 
  return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i64 __shl(__vec16_i64 a, unsigned long long b) {
  __vec16_i32 hi;
  if (b <= 32) hi = _mm512_or_epi32(_mm512_slli_epi32(a.v_hi, b), _mm512_srli_epi32(a.v_lo, 32-b));
  else hi = _mm512_slli_epi32(a.v_lo, b - 32);
  __vec16_i32 lo = _mm512_slli_epi32(a.v_lo, b); 
  return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i64 __lshr(__vec16_i64 a, __vec16_i64 b) {
  /* this is a safety gate in case b-shift >= 32 */
  const __vec16_i32 xfer = __select(
    __signed_less_than_i32(b.v_lo, __ispc_thirty_two), 
    __shl (a.v_hi, __sub(__ispc_thirty_two, b.v_lo)),
    __lshr(a.v_hi, __sub(b.v_lo, __ispc_thirty_two))
    );
  const __vec16_i32 lo = __or(__lshr(a.v_lo, b.v_lo), xfer);
  const __vec16_i32 hi =      __lshr(a.v_hi, b.v_lo);

  return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i64 __lshr(__vec16_i64 a, unsigned long long b) {
  /* this is a safety gate in case b-shift >= 32 */
  __vec16_i32 xfer;
  if (32 <= b) xfer = __lshr(a.v_hi, b-32);
  else xfer = _mm512_and_epi32(_mm512_slli_epi32(__ispc_ffffffff, 32-b), _mm512_slli_epi32(a.v_hi, 32-b));
  __vec16_i32 hi = _mm512_srli_epi32(a.v_hi, b);
  __vec16_i32 lo = _mm512_or_epi32(xfer, _mm512_srli_epi32(a.v_lo, b));
  return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i64 __ashr(__vec16_i64 a, __vec16_i64 b) {
  /* this is a safety gate in case b-shift >= 32 */
  const __vec16_i32 xfer = __select(
    __signed_less_than_i32(b.v_lo, __ispc_thirty_two), 
    __shl (a.v_hi, __sub(__ispc_thirty_two, b.v_lo)),
    __ashr(a.v_hi, __sub(b.v_lo, __ispc_thirty_two))
    );
  const __vec16_i32 lo = __or(__lshr(a.v_lo, b.v_lo), xfer);
  const __vec16_i32 hi =      __ashr(a.v_hi, b.v_lo);
  return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i64 __ashr(__vec16_i64 a, unsigned long long b) {
  __vec16_i32 xfer;
  if (b < 32) xfer = _mm512_slli_epi32(_mm512_and_epi32(a.v_hi, _mm512_set1_epi32((1<<b)-1)), 32-b);
  else  xfer = _mm512_srai_epi32(a.v_hi, b-32);
  __vec16_i32 hi = _mm512_srai_epi32(a.v_hi, b);
  __vec16_i32 lo = _mm512_or_epi32(xfer, _mm512_srli_epi32(a.v_lo, b));
  return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i1 __equal_i64(const __vec16_i64 &a, const __vec16_i64 &b) {
  const __mmask16 lo_match = _mm512_cmpeq_epi32_mask(a.v_lo,b.v_lo);
  return _mm512_mask_cmpeq_epi32_mask(lo_match,a.v_hi,b.v_hi);
}

static FORCEINLINE __vec16_i1 __equal_i64_and_mask(const __vec16_i64 &a, const __vec16_i64 &b,
    __vec16_i1 mask) {
  __mmask16 lo_match = _mm512_mask_cmpeq_epi32_mask((__mmask16)mask, a.v_lo,b.v_lo);
  __mmask16 full_match = _mm512_mask_cmpeq_epi32_mask(lo_match,a.v_hi,b.v_hi);
  return full_match;
}

static FORCEINLINE __vec16_i1 __not_equal_i64(const __vec16_i64 &a, const __vec16_i64 &b) {
  return __not(__equal_i64(a,b));
}

static FORCEINLINE __vec16_i1 __not_equal_i64_and_mask(const __vec16_i64 &a, const __vec16_i64 &b,
    __vec16_i1 mask) {
  return __and(__not(__equal_i64(a,b)), mask);
}


static FORCEINLINE __vec16_i1 __unsigned_less_than_i64(__vec16_i64 a, __vec16_i64 b) {
  __vec16_i1 ret_hi = _mm512_cmplt_epu32_mask(a.v_hi, b.v_hi);
  __vec16_i1 hi_eq = _mm512_cmpeq_epu32_mask(a.v_hi, b.v_hi);
  __vec16_i1 ret_lo = _mm512_mask_cmplt_epu32_mask(hi_eq, a.v_lo, b.v_lo);
  return ret_hi | ret_lo;
}

static FORCEINLINE __vec16_i1 __unsigned_less_than_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 m)
{
  return __unsigned_less_than_i64(a, b) & m;
}

static FORCEINLINE __vec16_i1 __unsigned_greater_than_i64(__vec16_i64 a, __vec16_i64 b) {
  __vec16_i1 ret_hi = _mm512_cmpgt_epu32_mask(a.v_hi, b.v_hi);
  __vec16_i1 hi_eq = _mm512_cmpeq_epu32_mask(a.v_hi, b.v_hi);
  __vec16_i1 ret_lo = _mm512_mask_cmpgt_epu32_mask(hi_eq, a.v_lo, b.v_lo);
  return ret_hi | ret_lo;
}

static FORCEINLINE __vec16_i1 __unsigned_greater_than_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 m)
{
  return __unsigned_greater_than_i64(a, b) & m;
}

static FORCEINLINE __vec16_i1 __unsigned_less_equal_i64(__vec16_i64 a, __vec16_i64 b) {
  __vec16_i1 ret_s_cmp = __unsigned_less_than_i64(a, b);
  __vec16_i1 ret_equal = __equal_i64(a, b);
  return ret_equal | ret_s_cmp;
}

static FORCEINLINE __vec16_i1 __unsigned_less_equal_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 m) 
{
  return __unsigned_less_equal_i64(a, b) & m;
}

static FORCEINLINE __vec16_i1 __unsigned_greater_equal_i64(__vec16_i64 a, __vec16_i64 b) {
  __vec16_i1 ret_s_cmp = __unsigned_greater_than_i64(a, b);
  __vec16_i1 ret_equal = __equal_i64(a, b);
  return ret_equal | ret_s_cmp;
}

static FORCEINLINE __vec16_i1 __unsigned_greater_equal_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 m)
{
  return __unsigned_greater_equal_i64(a, b) & m;
}

static FORCEINLINE __vec16_i1 __signed_less_than_i64(__vec16_i64 a, __vec16_i64 b) {
  __vec16_i1 ret_hi = _mm512_cmplt_epi32_mask(a.v_hi, b.v_hi);
  __vec16_i1 hi_eq = _mm512_cmpeq_epi32_mask(a.v_hi, b.v_hi);
  __vec16_i1 ret_lo = _mm512_mask_cmplt_epu32_mask(hi_eq, a.v_lo, b.v_lo);
  return ret_hi | ret_lo; 
}

static FORCEINLINE __vec16_i1 __signed_less_than_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 m)
{
  return __signed_less_than_i64(a, b) & m;
}

static FORCEINLINE __vec16_i1 __signed_greater_than_i64(__vec16_i64 a, __vec16_i64 b) {
  __vec16_i1 ret_hi = _mm512_cmpgt_epi32_mask(a.v_hi, b.v_hi);
  __vec16_i1 hi_eq = _mm512_cmpeq_epi32_mask(a.v_hi, b.v_hi);
  __vec16_i1 ret_lo = _mm512_mask_cmpgt_epu32_mask(hi_eq, a.v_lo, b.v_lo);
  return ret_hi | ret_lo;
}

static FORCEINLINE __vec16_i1 __signed_greater_than_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 m)
{
  return __signed_greater_than_i64(a, b) & m;
}

static FORCEINLINE __vec16_i1 __signed_less_equal_i64(__vec16_i64 a, __vec16_i64 b) {
  __vec16_i1 ret_s_cmp = __signed_less_than_i64(a, b);
  __vec16_i1 ret_equal = __equal_i64(a, b);
  return ret_equal | ret_s_cmp;
}

static FORCEINLINE __vec16_i1 __signed_less_equal_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 m)
{
  return __signed_less_equal_i64(a, b) & m;
}

static FORCEINLINE __vec16_i1 __signed_greater_equal_i64(__vec16_i64 a, __vec16_i64 b) {
  __vec16_i1 ret_s_cmp = __signed_greater_than_i64(a, b);
  __vec16_i1 ret_equal = __equal_i64(a, b);
  return ret_equal | ret_s_cmp;
}

static FORCEINLINE __vec16_i1 __signed_greater_equal_i64_and_mask(__vec16_i64 a, __vec16_i64 b, __vec16_i1 m)
{
  return __signed_greater_equal_i64(a, b) & m;
}


template <class RetVecType> static RetVecType __smear_i64(const int64_t &l);
template <> FORCEINLINE  __vec16_i64 __smear_i64<__vec16_i64>(const int64_t &l) {
  const int *i = (const int*)&l;
  return __vec16_i64(_mm512_set1_epi32(i[0]), _mm512_set1_epi32(i[1]));
}

static FORCEINLINE __vec16_i64 __rotate_i64(__vec16_i64 v, int index) {
  __vec16_i32 idx = __smear_i32<__vec16_i32>(index);
  __vec16_i32 shuffle = _mm512_and_epi32(_mm512_add_epi32(__ispc_stride1, idx),  __smear_i32<__vec16_i32>(0xf));
  return __vec16_i64(_mm512_mask_permutevar_epi32(v.v_lo, 0xffff, shuffle, v.v_lo), 
                     _mm512_mask_permutevar_epi32(v.v_hi, 0xffff, shuffle, v.v_hi));
}

static FORCEINLINE __vec16_i64 __shuffle2_i64(__vec16_i64 v0, __vec16_i64 v1, __vec16_i32 index) {
  return __vec16_i64(__shuffle2_i32(v0.v_lo, v1.v_lo, index), __shuffle2_i32(v0.v_hi, v1.v_hi, index));
}


template <int ALIGN> static FORCEINLINE __vec16_i64 __load(const __vec16_i64 *p) {
  __vec16_i32 v1;
  __vec16_i32 v2;
  const uint8_t*ptr = (const uint8_t*)p;
  v2 = _mm512_extloadunpacklo_epi32(v2, ptr, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v2 = _mm512_extloadunpackhi_epi32(v2, ptr+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v1 = _mm512_extloadunpacklo_epi32(v1, ptr+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v1 = _mm512_extloadunpackhi_epi32(v1, ptr+128, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);

  __vec16_i64 ret;
  ret.v_hi = _mm512_mask_permutevar_epi32(ret.v_hi, 0xFF00,
      _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
      v1);
  ret.v_hi = _mm512_mask_permutevar_epi32(ret.v_hi, 0x00FF,
      _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
      v2);
  ret.v_lo = _mm512_mask_permutevar_epi32(ret.v_lo, 0xFF00,
      _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
      v1);
  ret.v_lo = _mm512_mask_permutevar_epi32(ret.v_lo, 0x00FF,
      _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
      v2);
  return ret;    
}

#if 0
template <> FORCEINLINE __vec16_i64 __load<64>(const __vec16_i64 *p) {
  __m512i v2 = _mm512_load_epi32(p);
  __m512i v1 = _mm512_load_epi32(((uint8_t*)p)+64);
  __vec16_i64 ret;
  ret.v_hi = _mm512_mask_permutevar_epi32(ret.v_hi, 0xFF00,
      _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
      v1);
  ret.v_hi = _mm512_mask_permutevar_epi32(ret.v_hi, 0x00FF,
      _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
      v2);
  ret.v_lo = _mm512_mask_permutevar_epi32(ret.v_lo, 0xFF00,
      _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
      v1);
  ret.v_lo = _mm512_mask_permutevar_epi32(ret.v_lo, 0x00FF,
      _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
      v2);
  return ret;    
}

template <> FORCEINLINE __vec16_i64 __load<128>(const __vec16_i64 *p) {
  return __load<64>(p);
}
#endif

template <int ALIGN> static FORCEINLINE void __store(__vec16_i64 *p, __vec16_i64 v) {
  __m512i v1;
  __m512i v2;
  v1 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
      _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
      v.v_hi);
  v1 = _mm512_mask_permutevar_epi32(v1, 0x5555,
      _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
      v.v_lo);
  v2 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
      _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
      v.v_hi);
  v2 = _mm512_mask_permutevar_epi32(v2, 0x5555,
      _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
      v.v_lo);
  _mm512_extpackstorelo_epi32(p, v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+64, v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorelo_epi32((uint8_t*)p+64, v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+128, v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
}
#if 0
template <> FORCEINLINE void __store<64>(__vec16_i64 *p, __vec16_i64 v) {
  __m512i v1;
  __m512i v2;
  v1 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
      _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
      v.v_hi);
  v1 = _mm512_mask_permutevar_epi32(v1, 0x5555,
      _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
      v.v_lo);
  v2 = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
      _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
      v.v_hi);
  v2 = _mm512_mask_permutevar_epi32(v2, 0x5555,
      _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
      v.v_lo);    
  _mm512_store_epi64(p, v2);
  _mm512_store_epi64(((uint8_t*)p)+64, v1);
}

template <> FORCEINLINE void __store<128>(__vec16_i64 *p, __vec16_i64 v) {
  __store<64>(p, v);
}
#endif

/*! gather vector of 64-bit ints from addresses pointing to uniform ints 

  (iw) WARNING: THIS CODE ONLY WORKS FOR GATHERS FROM ARRAYS OF
 ***UNIFORM*** INT64's/POINTERS.  (problem is that ispc doesn't
 expose whether it's from array of uniform or array of varying
 poitners, so in here there's no way to tell - only thing we can do
 is pick one...
 */
static FORCEINLINE __vec16_i64
__gather_base_offsets32_i64(uint8_t *base, uint32_t scale, __vec16_i32 offsets, 
    __vec16_i1 mask) {
  __vec16_i64 ret;
  ret.v_lo = _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask, offsets, 
      base, _MM_UPCONV_EPI32_NONE, scale,
      _MM_HINT_NONE);
  ret.v_hi = _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask, offsets, 
      base+4, _MM_UPCONV_EPI32_NONE, scale,
      _MM_HINT_NONE);
  return ret;
}

/*! gather vector of 64-bit ints from addresses pointing to uniform ints 

  (iw) WARNING: THIS CODE ONLY WORKS FOR GATHERS FROM ARRAYS OF
 ***UNIFORM*** INT64's/POINTERS.  (problem is that ispc doesn't
 expose whether it's from array of uniform or array of varying
 poitners, so in here there's no way to tell - only thing we can do
 is pick one...
 */
  static FORCEINLINE __vec16_i64
__gather64_i64(__vec16_i64 addr, __vec16_i1 mask) 
{
  __vec16_i64 ret;

  // There is no gather instruction with 64-bit offsets in KNC.
  // We have to manually iterate over the upper 32 bits ;-)
  __vec16_i1 still_to_do = mask;
  const __vec16_i32 signed_offsets = _mm512_add_epi32(addr.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint32_t &hi32 = ((uint*)&addr.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(still_to_do,addr.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((((unsigned long)hi32) << 32) + (unsigned long)(-(long)INT_MIN));
    ret.v_lo = _mm512_mask_i32extgather_epi32(ret.v_lo, match, signed_offsets, 
        base, _MM_UPCONV_EPI32_NONE, 1,
        _MM_HINT_NONE);
    ret.v_hi = _mm512_mask_i32extgather_epi32(ret.v_hi, match, signed_offsets, 
        base+4, _MM_UPCONV_EPI32_NONE, 1,
        _MM_HINT_NONE);

    still_to_do = _mm512_kxor(match, still_to_do);
  }

  return ret;
}



///////////////////////////////////////////////////////////////////////////
// float
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_f __add(__vec16_f a, __vec16_f b) { 
  return _mm512_add_ps(a, b);
}

static FORCEINLINE __vec16_f __sub(__vec16_f a, __vec16_f b) {
  return _mm512_sub_ps(a, b);
}

static FORCEINLINE __vec16_f __mul(__vec16_f a, __vec16_f b) {
  return _mm512_mul_ps(a, b);
}

static FORCEINLINE __vec16_f __div(__vec16_f a, __vec16_f b) {
  return _mm512_div_ps(a, b);
}

static FORCEINLINE __vec16_i1 __equal_float(__vec16_f a, __vec16_f b) {
  return _mm512_cmpeq_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __equal_float_and_mask(__vec16_f a, __vec16_f b,
    __vec16_i1 m) {
  return _mm512_mask_cmpeq_ps_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __not_equal_float(__vec16_f a, __vec16_f b) {
  return _mm512_cmpneq_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __not_equal_float_and_mask(__vec16_f a, __vec16_f b,
    __vec16_i1 m) {
  return _mm512_mask_cmpneq_ps_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __less_than_float(__vec16_f a, __vec16_f b) {
  return _mm512_cmplt_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __less_than_float_and_mask(__vec16_f a, __vec16_f b,
    __vec16_i1 m) {
  return _mm512_mask_cmplt_ps_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __less_equal_float(__vec16_f a, __vec16_f b) {
  return _mm512_cmple_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __less_equal_float_and_mask(__vec16_f a, __vec16_f b,
    __vec16_i1 m) {
  return _mm512_mask_cmple_ps_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __greater_than_float(__vec16_f a, __vec16_f b) {
  return _mm512_cmpnle_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __greater_than_float_and_mask(__vec16_f a, __vec16_f b,
    __vec16_i1 m) {
  return _mm512_mask_cmpnle_ps_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __greater_equal_float(__vec16_f a, __vec16_f b) {
  return _mm512_cmpnlt_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __greater_equal_float_and_mask(__vec16_f a, __vec16_f b,
    __vec16_i1 m) {
  return _mm512_mask_cmpnlt_ps_mask(m, a, b);
}

static FORCEINLINE __vec16_i1 __ordered_float(__vec16_f a, __vec16_f b) {
  return _mm512_cmpord_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __ordered_float_and_mask(__vec16_f a, __vec16_f b, __vec16_i1 mask) {
  return _mm512_mask_cmpord_ps_mask(mask, a, b);
}

static FORCEINLINE __vec16_i1 __unordered_float(__vec16_f a, __vec16_f b) {
  return _mm512_cmpunord_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __unordered_float_and_mask(__vec16_f a, __vec16_f b, __vec16_i1 mask) {
  return _mm512_mask_cmpunord_ps_mask(mask, a, b);
}

static FORCEINLINE __vec16_f __select(__vec16_i1 mask, __vec16_f a, __vec16_f b) {
  return _mm512_mask_mov_ps(b, mask, a);
}

static FORCEINLINE __vec16_f __select(bool cond, __vec16_f a, __vec16_f b) {
  return cond ? a : b;
}

static FORCEINLINE float __extract_element(__vec16_f v, uint32_t index) {
  return ((float *)&v)[index];
}

static FORCEINLINE void  __insert_element(__vec16_f *v, uint32_t index, float val) {
  ((float *)v)[index] = val;
}

template <class RetVecType> static RetVecType __smear_float(float f);
template <> FORCEINLINE __vec16_f __smear_float<__vec16_f>(float f) {
  return _mm512_set_1to16_ps(f);
}

template <class RetVecType> static RetVecType __setzero_float();
template <> FORCEINLINE __vec16_f __setzero_float<__vec16_f>() {
  return _mm512_setzero_ps();
}

template <class RetVecType> static RetVecType __undef_float();
template <> FORCEINLINE __vec16_f __undef_float<__vec16_f>() {
  return __vec16_f();
}

static FORCEINLINE __vec16_f __broadcast_float(__vec16_f v, int index) {
  int32_t val = __extract_element(v, index & 0xf);
  return _mm512_set1_ps(val);
}

static FORCEINLINE __vec16_f __shuffle_float(__vec16_f v, __vec16_i32 index) {
  return _mm512_castsi512_ps(_mm512_mask_permutevar_epi32(_mm512_castps_si512(v), 0xffff, index, _mm512_castps_si512(v)));
}

static FORCEINLINE __vec16_f __shuffle2_float(__vec16_f v0, __vec16_f v1, __vec16_i32 index) {
  __vec16_f ret;
  for (int i = 0; i < 16; ++i){
    if (__extract_element(index, i) < 16)
      __insert_element(&ret, i, __extract_element(v0, __extract_element(index, i) & 0xF));
    else
      __insert_element(&ret, i, __extract_element(v1, __extract_element(index, i) & 0xF));
  }
  return ret;
}

template <int ALIGN> static FORCEINLINE __vec16_f __load(const __vec16_f *p) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  return _mm512_load_ps(p);
#else
  __vec16_f v;
  v = _mm512_extloadunpacklo_ps(v, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  v = _mm512_extloadunpackhi_ps(v, (uint8_t*)p+64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  return v;
#endif
}
#if 0
template <> FORCEINLINE __vec16_f __load<64>(const __vec16_f *p) {
  return _mm512_load_ps(p);
}
#endif
template <int ALIGN> static FORCEINLINE void __store(__vec16_f *p, __vec16_f v) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  _mm512_store_ps(p, v);
#else
  _mm512_extpackstorelo_ps(p, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_ps((uint8_t*)p+64, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
}
#if 0
template <> FORCEINLINE void __store<64>(__vec16_f *p, __vec16_f v) {
  _mm512_store_ps(p, v);
}
#endif

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

static FORCEINLINE __vec16_d __add(__vec16_d a, __vec16_d b) { 
  __vec16_d ret;
  ret.v1 = _mm512_add_pd(a.v1, b.v1);
  ret.v2 = _mm512_add_pd(a.v2, b.v2);
  return ret;
}

static FORCEINLINE __vec16_d __sub(__vec16_d a, __vec16_d b) {
  __vec16_d ret;
  ret.v1 = _mm512_sub_pd(a.v1, b.v1);
  ret.v2 = _mm512_sub_pd(a.v2, b.v2);
  return ret;
}

static FORCEINLINE __vec16_d __mul(__vec16_d a, __vec16_d b) {
  __vec16_d ret;
  ret.v1 = _mm512_mul_pd(a.v1, b.v1);
  ret.v2 = _mm512_mul_pd(a.v2, b.v2);
  return ret;
}

static FORCEINLINE __vec16_d __div(__vec16_d a, __vec16_d b) {
  __vec16_d ret;
  ret.v1 = _mm512_div_pd(a.v1, b.v1);
  ret.v2 = _mm512_div_pd(a.v2, b.v2);
  return ret;
}

static FORCEINLINE __vec16_d __sqrt_varying_double(__vec16_d v) {
  return __vec16_d(_mm512_sqrt_pd(v.v1),_mm512_sqrt_pd(v.v2));
}

static FORCEINLINE __vec16_i1 __equal_double(__vec16_d a, __vec16_d b) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_cmpeq_pd_mask(a.v1, b.v1);
  ret2 = _mm512_cmpeq_pd_mask(a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __equal_double_and_mask(__vec16_d a, __vec16_d b,
    __vec16_i1 m) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_mask_cmpeq_pd_mask(m, a.v1, b.v1);
  __vec16_i1 tmp_m = m;
  ret2 = _mm512_mask_cmpeq_pd_mask(_mm512_kswapb(tmp_m,tmp_m), a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __not_equal_double(__vec16_d a, __vec16_d b) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_cmpneq_pd_mask(a.v1, b.v1);
  ret2 = _mm512_cmpneq_pd_mask(a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __not_equal_double_and_mask(__vec16_d a, __vec16_d b,
    __vec16_i1 m) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  __vec16_i1 tmp_m = m;
  ret1 = _mm512_mask_cmpneq_pd_mask(m, a.v1, b.v1);
  ret2 = _mm512_mask_cmpneq_pd_mask(_mm512_kswapb(tmp_m, tmp_m), a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __less_than_double(__vec16_d a, __vec16_d b) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_cmplt_pd_mask(a.v1, b.v1);
  ret2 = _mm512_cmplt_pd_mask(a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __less_than_double_and_mask(__vec16_d a, __vec16_d b,
    __vec16_i1 m) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  __vec16_i1 tmp_m = m;
  ret1 = _mm512_mask_cmplt_pd_mask(m, a.v1, b.v1);
  ret2 = _mm512_mask_cmplt_pd_mask(_mm512_kswapb(tmp_m, tmp_m), a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __less_equal_double(__vec16_d a, __vec16_d b) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_cmple_pd_mask(a.v1, b.v1);
  ret2 = _mm512_cmple_pd_mask(a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __less_equal_double_and_mask(__vec16_d a, __vec16_d b,
    __vec16_i1 m) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  __vec16_i1 tmp_m = m;
  ret1 = _mm512_mask_cmple_pd_mask(m, a.v1, b.v1);
  ret2 = _mm512_mask_cmple_pd_mask(_mm512_kswapb(tmp_m, tmp_m), a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __greater_than_double(__vec16_d a, __vec16_d b) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_cmpnle_pd_mask(a.v1, b.v1);
  ret2 = _mm512_cmpnle_pd_mask(a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __greater_than_double_and_mask(__vec16_d a, __vec16_d b,
    __vec16_i1 m) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  __vec16_i1 tmp_m = m;
  ret1 = _mm512_mask_cmpnle_pd_mask(m, a.v1, b.v1);
  ret2 = _mm512_mask_cmpnle_pd_mask(_mm512_kswapb(tmp_m, tmp_m), a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __greater_equal_double(__vec16_d a, __vec16_d b) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_cmpnlt_pd_mask(a.v1, b.v1);
  ret2 = _mm512_cmpnlt_pd_mask(a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __greater_equal_double_and_mask(__vec16_d a, __vec16_d b,
    __vec16_i1 m) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  __vec16_i1 tmp_m = m;
  ret1 = _mm512_mask_cmpnlt_pd_mask(m, a.v1, b.v1);
  ret2 = _mm512_mask_cmpnlt_pd_mask(_mm512_kswapb(tmp_m, tmp_m), a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __ordered_double(__vec16_d a, __vec16_d b) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_cmpord_pd_mask(a.v1, b.v1);
  ret2 = _mm512_cmpord_pd_mask(a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __unordered_double(__vec16_d a, __vec16_d b) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  ret1 = _mm512_cmpunord_pd_mask(a.v1, b.v1);
  ret2 = _mm512_cmpunord_pd_mask(a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_i1 __unordered_double_and_mask(__vec16_d a, __vec16_d b, __vec16_i1 mask) {
  __vec16_i1 ret1;
  __vec16_i1 ret2;
  __vec16_i1 tmp_m = mask;
  ret1 = _mm512_mask_cmpunord_pd_mask(mask, a.v1, b.v1);
  ret2 = _mm512_mask_cmpunord_pd_mask(_mm512_kswapb(tmp_m, tmp_m), a.v2, b.v2);
  return _mm512_kmovlhb(ret1, ret2);
}

static FORCEINLINE __vec16_d __select(__vec16_i1 mask, __vec16_d a, __vec16_d b) {
  __vec16_d ret;
  __vec16_i1 tmp_m = mask;
  ret.v1 = _mm512_mask_mov_pd(b.v1, mask, a.v1);
  ret.v2 = _mm512_mask_mov_pd(b.v2, _mm512_kswapb(tmp_m, tmp_m), a.v2);
  return ret;
}


static FORCEINLINE __vec16_d __select(bool cond, __vec16_d a, __vec16_d b) {
  return cond ? a : b;
}

static FORCEINLINE double __extract_element(__vec16_d v, uint32_t index) {
  return ((double *)&v)[index];
}

static FORCEINLINE void  __insert_element(__vec16_d *v, uint32_t index, double val) {
  ((double *)v)[index] = val;
}

template <class RetVecType> static RetVecType __smear_double(double d);
template <> FORCEINLINE __vec16_d __smear_double<__vec16_d>(double d) {
  __vec16_d ret;
  ret.v1 = _mm512_set1_pd(d);
  ret.v2 = _mm512_set1_pd(d);
  return ret;
}

template <class RetVecType> static RetVecType __setzero_double();
template <> FORCEINLINE __vec16_d __setzero_double<__vec16_d>() {
  __vec16_d ret;
  ret.v1 = _mm512_setzero_pd();
  ret.v2 = _mm512_setzero_pd();
  return ret;
}

template <class RetVecType> static RetVecType __undef_double();
template <> FORCEINLINE __vec16_d __undef_double<__vec16_d>() {
  return __vec16_d();
}

static FORCEINLINE __vec16_d __broadcast_double(__vec16_d v, int index) {
  __vec16_d ret;
  double val = __extract_element(v, index & 0xf);
  ret.v1 = _mm512_set1_pd(val);
  ret.v2 = _mm512_set1_pd(val);
  return ret;
}

static FORCEINLINE __vec16_d __shuffle2_double(__vec16_d v0, __vec16_d v1, __vec16_i32 index) {
  __vec16_d ret;
  for (int i = 0; i < 16; ++i){
    if (__extract_element(index, i) < 16)
      __insert_element(&ret, i, __extract_element(v0, __extract_element(index, i) & 0xF));
    else 
      __insert_element(&ret, i, __extract_element(v1, __extract_element(index, i) & 0xF));
  }
  return ret;
}

template <int ALIGN> static FORCEINLINE __vec16_d __load(const __vec16_d *p) {
  __vec16_d ret;
  ret.v1 = _mm512_extloadunpacklo_pd(ret.v1, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  ret.v1 = _mm512_extloadunpackhi_pd(ret.v1, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  ret.v2 = _mm512_extloadunpacklo_pd(ret.v2, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  ret.v2 = _mm512_extloadunpackhi_pd(ret.v2, (uint8_t*)p+128, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  return ret;
}
#if 0
template <> FORCEINLINE __vec16_d __load<64>(const __vec16_d *p) {
  __vec16_d ret;
  ret.v1 = _mm512_load_pd(p);
  ret.v2 = _mm512_load_pd(((uint8_t*)p)+64);
  return ret;
}

template <> FORCEINLINE __vec16_d __load<128>(const __vec16_d *p) {
  return __load<64>(p);
}
#endif
template <int ALIGN> static FORCEINLINE void __store(__vec16_d *p, __vec16_d v) {
  _mm512_extpackstorelo_pd(p, v.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_pd((uint8_t*)p+64, v.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorelo_pd((uint8_t*)p+64, v.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_pd((uint8_t*)p+128, v.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
}
#if 0
template <> FORCEINLINE void __store<64>(__vec16_d *p, __vec16_d v) {
  _mm512_store_pd(p, v.v1);
  _mm512_store_pd(((uint8_t*)p)+64, v.v2);
}

template <> FORCEINLINE void __store<128>(__vec16_d *p, __vec16_d v) {
  __store<64>(p, v);
}
#endif
///////////////////////////////////////////////////////////////////////////
// casts
///////////////////////////////////////////////////////////////////////////
static FORCEINLINE __vec16_i8 __cast_sext(const __vec16_i8 &, const __vec16_i1 &val)
{
  return __vec16_i8(-val[0], -val[1], -val[2],   -val[3],  -val[4],  -val[5],  -val[6],  -val[7], 
                    -val[8], -val[9], -val[10],  -val[11], -val[12], -val[13], -val[14], -val[15]);
}

static FORCEINLINE __vec16_i16 __cast_sext(const __vec16_i16 &, const __vec16_i1 &val)
{
  return __vec16_i16(-val[0], -val[1], -val[2],   -val[3],  -val[4],  -val[5],  -val[6],  -val[7], 
                     -val[8], -val[9], -val[10],  -val[11], -val[12], -val[13], -val[14], -val[15]);
}

static FORCEINLINE __vec16_i16 __cast_sext(const __vec16_i16 &, const __vec16_i8 &val)
{ 
  return __vec16_i16((int8_t)val[0],  (int8_t)val[1],  (int8_t)val[2],  (int8_t)val[3],  
                     (int8_t)val[4],  (int8_t)val[5],  (int8_t)val[6],  (int8_t)val[7], 
                     (int8_t)val[8],  (int8_t)val[9],  (int8_t)val[10], (int8_t)val[11], 
                     (int8_t)val[12], (int8_t)val[13], (int8_t)val[14], (int8_t)val[15]);
}

static FORCEINLINE __vec16_i32 __cast_sext(const __vec16_i32 &, const __vec16_i1 &val)
{
  __vec16_i32 ret = _mm512_setzero_epi32();
  __vec16_i32 one = _mm512_set1_epi32(-1);
  return _mm512_mask_mov_epi32(ret, val, one);
}

static FORCEINLINE __vec16_i32 __cast_sext(const __vec16_i32 &, const __vec16_i8 &val)
{
  //return _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 a = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  return a;
}

static FORCEINLINE __vec16_i32 __cast_sext(const __vec16_i32 &, const __vec16_i16 &val)
{
  return _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
}



static FORCEINLINE __vec16_i64 __cast_sext(const __vec16_i64 &, const __vec16_i1 &val)
{
  __vec16_i32 ret = _mm512_mask_mov_epi32(_mm512_setzero_epi32(), val, _mm512_set1_epi32(-1));
  return __vec16_i64(ret, ret);
}

static FORCEINLINE __vec16_i64 __cast_sext(const __vec16_i64 &, const __vec16_i8 &val)
{
  __vec16_i32 a = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  return __vec16_i64(a.v, _mm512_srai_epi32(a.v, 31));
}

static FORCEINLINE __vec16_i64 __cast_sext(const __vec16_i64 &, const __vec16_i32 &val)
{
  return __vec16_i64(val.v, _mm512_srai_epi32(val.v, 31));
}



static FORCEINLINE __vec16_i8 __cast_zext(const __vec16_i8 &, const __vec16_i1 &val)
{
  return __vec16_i8(val[0], val[1], val[2],   val[3],  val[4],  val[5],  val[6],  val[7], 
                    val[8], val[9], val[10],  val[11], val[12], val[13], val[14], val[15]);
}

static FORCEINLINE __vec16_i16 __cast_zext(const __vec16_i16 &, const __vec16_i1 &val)
{
  return __vec16_i16(val[0], val[1], val[2],   val[3],  val[4],  val[5],  val[6],  val[7], 
                     val[8], val[9], val[10],  val[11], val[12], val[13], val[14], val[15]);
}

static FORCEINLINE __vec16_i16 __cast_zext(const __vec16_i16 &, const __vec16_i8 &val)
{
  return __vec16_i16((uint8_t)val[0],  (uint8_t)val[1],  (uint8_t)val[2],  (uint8_t)val[3],  
                     (uint8_t)val[4],  (uint8_t)val[5],  (uint8_t)val[6],  (uint8_t)val[7], 
                     (uint8_t)val[8],  (uint8_t)val[9],  (uint8_t)val[10], (uint8_t)val[11], 
                     (uint8_t)val[12], (uint8_t)val[13], (uint8_t)val[14], (uint8_t)val[15]);
}

static FORCEINLINE __vec16_i32 __cast_zext(const __vec16_i32 &, const __vec16_i1 &val)
{
  __vec16_i32 ret = _mm512_setzero_epi32();
  __vec16_i32 one = _mm512_set1_epi32(1);
  return _mm512_mask_mov_epi32(ret, val, one);
}

static FORCEINLINE __vec16_i32 __cast_zext(const __vec16_i32 &, const __vec16_i8 &val)
{
  return _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
}

static FORCEINLINE __vec16_i32 __cast_zext(const __vec16_i32 &, const __vec16_i16 &val)
{
  return _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_UINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
}

static FORCEINLINE __vec16_i64 __cast_zext(const __vec16_i64 &, const __vec16_i1 &val)
{  
  __vec16_i32 ret_hi = _mm512_setzero_epi32();
  __vec16_i32 ret_lo = _mm512_setzero_epi32();
  __vec16_i32 one = _mm512_set1_epi32(1);
  ret_lo = _mm512_mask_mov_epi32(ret_lo, val, one);
  return __vec16_i64 (ret_lo, ret_hi);
}

static FORCEINLINE __vec16_i64 __cast_zext(const __vec16_i64 &, const __vec16_i8 &val)
{
  return __vec16_i64(__cast_zext(__vec16_i32(), val), _mm512_setzero_epi32());
}

static FORCEINLINE __vec16_i64 __cast_zext(const __vec16_i64 &, const __vec16_i16 &val)
{
  return __vec16_i64(__cast_zext(__vec16_i32(), val), _mm512_setzero_epi32());
}

static FORCEINLINE __vec16_i64 __cast_zext(const __vec16_i64 &, const __vec16_i32 &val)
{
  return __vec16_i64(val.v, _mm512_setzero_epi32());
}

static FORCEINLINE __vec16_f __cast_sitofp(__vec16_f, __vec16_i8 val) {
  return _mm512_extload_ps(&val, _MM_UPCONV_PS_SINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
}

static FORCEINLINE __vec16_f __cast_sitofp(__vec16_f, __vec16_i16 val) {
  return _mm512_extload_ps(&val, _MM_UPCONV_PS_SINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
}

static FORCEINLINE __vec16_f __cast_sitofp(__vec16_f, __vec16_i32 val) {
  return _mm512_cvtfxpnt_round_adjustepi32_ps(val, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE); 
}

static FORCEINLINE __vec16_f __cast_sitofp(__vec16_f, __vec16_i64 val) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(val, tmp1, tmp2);
  __vec16_f ret;

/*
  // Cycles don't work. It seems that it is icc bug.
  for (int i = 0; i < 8; i++) {
    ret[i] = (float)(((int64_t*)&tmp1)[i]);
  }
  for (int i = 0; i < 8; i++) {
    ((float*)&ret)[i + 8] = (float)(((int64_t*)&tmp2)[i]);
  }
*/

  ret[0] = (float)(((int64_t*)&tmp1)[0]);
  ret[1] = (float)(((int64_t*)&tmp1)[1]);
  ret[2] = (float)(((int64_t*)&tmp1)[2]);
  ret[3] = (float)(((int64_t*)&tmp1)[3]);
  ret[4] = (float)(((int64_t*)&tmp1)[4]);
  ret[5] = (float)(((int64_t*)&tmp1)[5]);
  ret[6] = (float)(((int64_t*)&tmp1)[6]);
  ret[7] = (float)(((int64_t*)&tmp1)[7]);

  ret[8] = (float)(((int64_t*)&tmp2)[0]);
  ret[9] = (float)(((int64_t*)&tmp2)[1]);
  ret[10] = (float)(((int64_t*)&tmp2)[2]);
  ret[11] = (float)(((int64_t*)&tmp2)[3]);
  ret[12] = (float)(((int64_t*)&tmp2)[4]);
  ret[13] = (float)(((int64_t*)&tmp2)[5]);
  ret[14] = (float)(((int64_t*)&tmp2)[6]);
  ret[15] = (float)(((int64_t*)&tmp2)[7]);

  return ret;
}

static FORCEINLINE __vec16_d __cast_sitofp(__vec16_d, __vec16_i8 val) {
  __vec16_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
  __vec16_d ret;
  ret.v1 = _mm512_cvtepi32lo_pd(vi);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(vi, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepi32lo_pd(other8);
  return ret;
}

static FORCEINLINE __vec16_d __cast_sitofp(__vec16_d, __vec16_i16 val) {
  __vec16_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
  __vec16_d ret;
  ret.v1 = _mm512_cvtepi32lo_pd(vi);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(vi, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepi32lo_pd(other8);
  return ret;
}

static FORCEINLINE __vec16_d __cast_sitofp(__vec16_d, __vec16_i32 val) {
  __vec16_d ret;
  ret.v1 = _mm512_cvtepi32lo_pd(val);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(val, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepi32lo_pd(other8);
  return ret;
}

static FORCEINLINE __vec16_d __cast_sitofp(__vec16_d, __vec16_i64 val) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(val, tmp1, tmp2);
  __vec16_d ret;
  for (int i = 0; i < 8; i++) {
    ((double*)&ret.v1)[i] = (double)(((int64_t*)&tmp1)[i]);
  }
  for (int i = 0; i < 8; i++) {
    ((double*)&ret.v2)[i] = (double)(((int64_t*)&tmp2)[i]);
  }
  return ret;
}


static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i1 v) 
{
  const __m512 ret = _mm512_setzero_ps();
  const __m512 one = _mm512_set1_ps(1.0);
  return _mm512_mask_mov_ps(ret, v, one);
}

static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, const __vec16_i8 &v) {
  return _mm512_extload_ps(v.v, _MM_UPCONV_PS_UINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
}

static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i16 val) {
  return _mm512_extload_ps(&val, _MM_UPCONV_PS_UINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
}

static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i32 v) {
  return _mm512_cvtfxpnt_round_adjustepu32_ps(v, _MM_FROUND_NO_EXC, _MM_EXPADJ_NONE); 
}

static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i64 val) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(val, tmp1, tmp2);
  __vec16_f ret;
  // Cycles don't work. It seems that it is icc bug.
  /*
  for (int i = 0; i < 8; i++) {
    ((float*)&ret)[i] = ((float)(((uint64_t*)&tmp1)[i]));
  }
  for (int i = 0; i < 8; i++) {
    ((float*)&ret)[i + 8] = ((float)(((uint64_t*)&tmp2)[i]));
  }
  */
  ret[0] = ((float)(((uint64_t*)&tmp1)[0]));
  ret[1] = ((float)(((uint64_t*)&tmp1)[1]));
  ret[2] = ((float)(((uint64_t*)&tmp1)[2]));
  ret[3] = ((float)(((uint64_t*)&tmp1)[3]));
  ret[4] = ((float)(((uint64_t*)&tmp1)[4]));
  ret[5] = ((float)(((uint64_t*)&tmp1)[5]));
  ret[6] = ((float)(((uint64_t*)&tmp1)[6]));
  ret[7] = ((float)(((uint64_t*)&tmp1)[7]));
  ret[8] = ((float)(((uint64_t*)&tmp2)[0]));
  ret[9] = ((float)(((uint64_t*)&tmp2)[1]));
  ret[10] = ((float)(((uint64_t*)&tmp2)[2]));
  ret[11] = ((float)(((uint64_t*)&tmp2)[3]));
  ret[12] = ((float)(((uint64_t*)&tmp2)[4]));
  ret[13] = ((float)(((uint64_t*)&tmp2)[5]));
  ret[14] = ((float)(((uint64_t*)&tmp2)[6]));
  ret[15] = ((float)(((uint64_t*)&tmp2)[7]));
  return ret;
}

static FORCEINLINE __vec16_d __cast_uitofp(__vec16_d, __vec16_i8 val) 
{
  __vec16_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_UINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
  __vec16_d ret;
  ret.v1 = _mm512_cvtepu32lo_pd(vi);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(vi, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepu32lo_pd(other8);
  return ret;
}

static FORCEINLINE __vec16_d __cast_uitofp(__vec16_d, __vec16_i16 val) 
{
  __vec16_i32 vi = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_UINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
  __vec16_d ret;
  ret.v1 = _mm512_cvtepu32lo_pd(vi);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(vi, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepu32lo_pd(other8);
  return ret;
}

static FORCEINLINE __vec16_d __cast_uitofp(__vec16_d, __vec16_i32 val) 
{
  __vec16_d ret;
  ret.v1 = _mm512_cvtepu32lo_pd(val);
  __vec16_i32 other8 = _mm512_permute4f128_epi32(val, _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtepu32lo_pd(other8);
  return ret;
}


static FORCEINLINE __vec16_d __cast_uitofp(__vec16_d, __vec16_i64 val) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(val, tmp1, tmp2);
  __vec16_d ret;
  for (int i = 0; i < 8; i++) {
    ((double*)&ret.v1)[i] = (double)(((uint64_t*)&tmp1)[i]);
  }
  for (int i = 0; i < 8; i++) {
    ((double*)&ret.v2)[i] = (double)(((uint64_t*)&tmp2)[i]);
  }
  return ret;
}


// float/double to signed int
static FORCEINLINE __vec16_i32 __cast_fptosi(__vec16_i32, __vec16_f val) {
  return _mm512_cvtfxpnt_round_adjustps_epi32(val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

static FORCEINLINE __vec16_i8 __cast_fptosi(__vec16_i8, __vec16_f val) {
  __vec16_i8 ret;
  __vec16_i32 tmp = __cast_fptosi(__vec16_i32(), val);
  _mm512_extstore_epi32(ret.v, tmp, _MM_DOWNCONV_EPI32_SINT8, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i16 __cast_fptosi(__vec16_i16, __vec16_f val) {
  __vec16_i16 ret;
  __vec16_i32 tmp = __cast_fptosi(__vec16_i32(), val);
  _mm512_extstore_epi32(ret.v, tmp, _MM_DOWNCONV_EPI32_SINT16, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i64 __cast_fptosi(__vec16_i64, __vec16_f val) {
  __m512i tmp1;
  for (int i = 0; i < 8; i++) {
    ((int64_t*)&tmp1)[i] = (int64_t)(((float*)&val)[i]);
  }
  __m512i tmp2;
  for (int i = 0; i < 8; i++) {
    ((int64_t*)&tmp2)[i] = (int64_t)(((float*)&val)[i + 8]);
  }
  return zmm2hilo(tmp1, tmp2);
}

static FORCEINLINE __vec16_i32 __cast_fptosi(__vec16_i32, __vec16_d val) { 
  __vec16_i32 tmp = _mm512_cvtfxpnt_roundpd_epi32lo(val.v2, _MM_ROUND_MODE_TOWARD_ZERO);
  __vec16_i32 ret_hi8 = _mm512_permute4f128_epi32(tmp, _MM_PERM_BADC);
  __vec16_i32 ret_lo8 = _mm512_cvtfxpnt_roundpd_epi32lo(val.v1, _MM_ROUND_MODE_TOWARD_ZERO);
  return _mm512_xor_epi32(ret_lo8, ret_hi8);
}

static FORCEINLINE __vec16_i8 __cast_fptosi(__vec16_i8, __vec16_d val) {
  __vec16_i8 ret;
  __vec16_i32 tmp = __cast_fptosi(__vec16_i32(), val);
  _mm512_extstore_epi32(ret.v, tmp, _MM_DOWNCONV_EPI32_SINT8, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i16 __cast_fptosi(__vec16_i16, __vec16_d val) {
  __vec16_i16 ret;
  __vec16_i32 tmp = __cast_fptosi(__vec16_i32(), val);
  _mm512_extstore_epi32(ret.v, tmp, _MM_DOWNCONV_EPI32_SINT16, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i64 __cast_fptosi(__vec16_i64, __vec16_d val) {
  __m512i tmp1;
  for (int i = 0; i < 8; i++) {
    ((int64_t*)&tmp1)[i] = (int64_t)(((double*)&val.v1)[i]);
  }
  __m512i tmp2;
  for (int i = 0; i < 8; i++) {
    ((int64_t*)&tmp2)[i] = (int64_t)(((double*)&val.v2)[i]);
  }
  return zmm2hilo(tmp1, tmp2);
}


static FORCEINLINE __vec16_i32 __cast_fptoui(__vec16_i32, __vec16_f val) {
  return _mm512_cvtfxpnt_round_adjustps_epu32(val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

static FORCEINLINE __vec16_i8 __cast_fptoui(__vec16_i8, __vec16_f val) {
  __vec16_i8 ret;
  __vec16_i32 tmp = __cast_fptoui(__vec16_i32(), val);
  _mm512_extstore_epi32(ret.v, tmp, _MM_DOWNCONV_EPI32_UINT8, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i16 __cast_fptoui(__vec16_i16, __vec16_f val) {
  __vec16_i16 ret;
  __vec16_i32 tmp = __cast_fptoui(__vec16_i32(), val);
  _mm512_extstore_epi32(ret.v, tmp, _MM_DOWNCONV_EPI32_UINT16, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i64 __cast_fptoui(__vec16_i64, __vec16_f val) {
  __m512i tmp1;
  for (int i = 0; i < 8; i++) {
    ((uint64_t*)&tmp1)[i] = (uint64_t)(((float*)&val)[i]);
  }
  __m512i tmp2;
  for (int i = 0; i < 8; i++) {
    ((uint64_t*)&tmp2)[i] = (uint64_t)(((float*)&val)[i + 8]);
  }
  return zmm2hilo(tmp1, tmp2);
}

static FORCEINLINE __vec16_i32 __cast_fptoui(__vec16_i32, __vec16_d val) {
  __vec16_i32 tmp = _mm512_cvtfxpnt_roundpd_epu32lo(val.v2, _MM_ROUND_MODE_TOWARD_ZERO);
  __vec16_i32 ret_hi8 = _mm512_permute4f128_epi32(tmp, _MM_PERM_BADC);
  __vec16_i32 ret_lo8 = _mm512_cvtfxpnt_roundpd_epu32lo(val.v1, _MM_ROUND_MODE_TOWARD_ZERO);
  return _mm512_xor_epi32(ret_lo8, ret_hi8);
}

static FORCEINLINE __vec16_i8 __cast_fptoui(__vec16_i8, __vec16_d val) {
  __vec16_i8 ret;
  __vec16_i32 tmp = __cast_fptoui(__vec16_i32(), val);
  _mm512_extstore_epi32(ret.v, tmp, _MM_DOWNCONV_EPI32_UINT8, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i16 __cast_fptoui(__vec16_i16, __vec16_d val) {
  __vec16_i16 ret;
  __vec16_i32 tmp = __cast_fptoui(__vec16_i32(), val);
  _mm512_extstore_epi32(ret.v, tmp, _MM_DOWNCONV_EPI32_UINT16, _MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i64 __cast_fptoui(__vec16_i64, __vec16_d val) {
  __m512i tmp1;
  for (int i = 0; i < 8; i++) {
    ((uint64_t*)&tmp1)[i] = (uint64_t)(((double*)&val.v1)[i]);
  }
  __m512i tmp2;
  for (int i = 0; i < 8; i++) {
    ((uint64_t*)&tmp2)[i] = (uint64_t)(((double*)&val.v2)[i]);
  }
  return zmm2hilo(tmp1, tmp2);
}







static FORCEINLINE __vec16_d __cast_fpext(__vec16_d, __vec16_f val) {
  __vec16_d ret;
  ret.v1 = _mm512_cvtpslo_pd(val.v);
  __vec16_f other8 = _mm512_permute4f128_epi32(_mm512_castps_si512(val.v), _MM_PERM_DCDC);
  ret.v2 = _mm512_cvtpslo_pd(other8);
  return ret;
}

static FORCEINLINE __vec16_f __cast_fptrunc(__vec16_f, __vec16_d val) {
  __m512i r0i = _mm512_castps_si512(_mm512_cvtpd_pslo(val.v2));
  __m512i r1i = _mm512_castps_si512(_mm512_cvtpd_pslo(val.v1));
  return _mm512_mask_permute4f128_epi32(r1i, 0xFF00, r0i, _MM_PERM_BABA);
}

static FORCEINLINE __vec16_f __cast_bits(__vec16_f, __vec16_i32 val) {
  return _mm512_castsi512_ps(val);
}

static FORCEINLINE __vec16_i32 __cast_bits(__vec16_i32, __vec16_f val) {
  return _mm512_castps_si512(val);
}

static FORCEINLINE __vec16_i32 __cast_bits(__vec16_i32, __vec16_i32 val) {
  return val;
}

static FORCEINLINE __vec16_i64 __cast_bits(__vec16_i64, __vec16_d val) {
  __vec16_i64 ret;
  ret.v_hi = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xFF00,
      _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
      _mm512_castpd_si512(val.v2));
  ret.v_hi = _mm512_mask_permutevar_epi32(ret.v_hi, 0x00FF,
      _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
      _mm512_castpd_si512(val.v1));
  ret.v_lo = _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xFF00,
      _mm512_set_16to16_pi(14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1),
      _mm512_castpd_si512(val.v2));
  ret.v_lo = _mm512_mask_permutevar_epi32(ret.v_lo, 0x00FF,
      _mm512_set_16to16_pi(15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0),
      _mm512_castpd_si512(val.v1));
  return ret;
}

static FORCEINLINE __vec16_d __cast_bits(__vec16_d, __vec16_i64 val) {
  __vec16_d ret;
  ret.v2 = _mm512_castsi512_pd(
      _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
        _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
	val.v_hi));
  ret.v2 = _mm512_castsi512_pd(
      _mm512_mask_permutevar_epi32(_mm512_castpd_si512(ret.v2), 0x5555,
        _mm512_set_16to16_pi(15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8),
	val.v_lo));
  ret.v1 = _mm512_castsi512_pd(
      _mm512_mask_permutevar_epi32(_mm512_undefined_epi32(), 0xAAAA,
        _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
	val.v_hi));
  ret.v1 = _mm512_castsi512_pd(
      _mm512_mask_permutevar_epi32(_mm512_castpd_si512(ret.v1), 0x5555,
        _mm512_set_16to16_pi(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0),
	val.v_lo));
  return ret;
}

static FORCEINLINE __vec16_i64 __cast_bits(__vec16_i64, __vec16_i64 val) {
  return val;
}

///////////////////////////////////////////////////////////////////////////
// templates for int8/16 operations
///////////////////////////////////////////////////////////////////////////
#define BINARY_OP(TYPE, NAME, OP)                                   \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
  TYPE ret;                                                         \
  for (int i = 0; i < 16; ++i)                                      \
    ret[i] = a[i] OP b[i];                                          \
  return ret;                                                       \
}

/* knc::macro::used */
#define BINARY_OP_CAST(TYPE, CAST, NAME, OP)                        \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
  TYPE ret;                                                         \
  for (int i = 0; i < 16; ++i)                                      \
    ret[i] = (CAST)(a[i]) OP (CAST)(b[i]);                          \
  return ret;                                                       \
}

#define CMP_OP(TYPE, SUFFIX, CAST, NAME, OP)                        \
static FORCEINLINE __vec16_i1 NAME##_##SUFFIX(TYPE a, TYPE b) {     \
  __vec16_i1 ret;                                                   \
  ret.v = 0;                                                        \
  for (int i = 0; i < 16; ++i)                                      \
    ret.v |= ((CAST)(a[i]) OP (CAST)(b[i])) << i;                   \
  return ret;                                                       \
}

#define SHIFT_UNIFORM(TYPE, CAST, NAME, OP)                         \
static FORCEINLINE TYPE NAME(TYPE a, int32_t b) {                   \
  TYPE ret;                                                         \
  for (int i = 0; i < 16; ++i)                                      \
    ret[i] = (CAST)(a[i]) OP b;                                     \
  return ret;                                                       \
}

#define SELECT(TYPE)                                                \
static FORCEINLINE TYPE __select(__vec16_i1 mask, TYPE a, TYPE b) { \
  TYPE ret;                                                         \
  for (int i = 0; i < 16; ++i)                                      \
    ret[i] = (mask.v & (1<<i)) ? a[i] : b[i];                       \
  return ret;                                                       \
}                                                                   \
static FORCEINLINE TYPE __select(bool cond, TYPE a, TYPE b) {       \
  return cond ? a : b;                                              \
}

#define REDUCE_ADD(TYPE, VTYPE, NAME)                               \
static FORCEINLINE TYPE NAME(VTYPE v) {                             \
  TYPE ret = v[0];                                                  \
  for (int i = 1; i < 16; ++i)                                      \
    ret = ret + v[i];                                               \
  return ret;                                                       \
}

///////////////////////////////////////////////////////////////////////////
// int8
///////////////////////////////////////////////////////////////////////////
template <class RetVecType> static RetVecType __setzero_i8();
template <> FORCEINLINE __vec16_i8 __setzero_i8<__vec16_i8>() {
      return __vec16_i8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

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
CMP_OP(__vec16_i8, i8, uint8_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i8, i8, int8_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i8, i8, uint8_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i8, i8, int8_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i8, i8, uint8_t, __unsigned_less_than, <)
CMP_OP(__vec16_i8, i8, int8_t,  __signed_less_than, <)
CMP_OP(__vec16_i8, i8, uint8_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i8, i8, int8_t,  __signed_greater_than, >)

SELECT(__vec16_i8)

static FORCEINLINE int8_t __extract_element(__vec16_i8 v, uint32_t index) {
    return v[index];
}

static FORCEINLINE void __insert_element(__vec16_i8 *v, uint32_t index, int8_t val) {
    ((int32_t *)v)[index] = val;
}

static FORCEINLINE __vec16_i8 __broadcast_i8(__vec16_i8 v, int index) {
    int32_t val = __extract_element(v, index & 0xf);
    __vec16_i32 tmp = _mm512_set1_epi32(val);
    __vec16_i8 ret;
    _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
    return ret;
}

static FORCEINLINE __vec16_i1 __not_equal_i8(__vec16_i8 a, __vec16_i8 b) {
    __vec16_i32 tmp_a = _mm512_extload_epi32(&a, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
    __vec16_i32 tmp_b = _mm512_extload_epi32(&b, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
    return __not_equal_i32(tmp_a, tmp_b);
}

static FORCEINLINE __vec16_i1 __equal_i8_and_mask(const __vec16_i8 &a, const __vec16_i8 &b, __vec16_i1 m) {
  __vec16_i32 tmp_a = _mm512_extload_epi32(&a, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp_b = _mm512_extload_epi32(&b, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  return __equal_i32_and_mask(tmp_a, tmp_b, m);
}

static FORCEINLINE __vec16_i1 __not_equal_i8_and_mask(__vec16_i8 a, __vec16_i8 b, __vec16_i1 m) {
  __vec16_i32 tmp_a = _mm512_extload_epi32(&a, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp_b = _mm512_extload_epi32(&b, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  return __not_equal_i32_and_mask(tmp_a, tmp_b, m);
}

static FORCEINLINE __vec16_i8 __rotate_i8(__vec16_i8 v, int index) {
    __vec16_i32 tmp_v = _mm512_extload_epi32(&v, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
    __vec16_i32 tmp   = __rotate_i32(tmp_v, index);
    __vec16_i8 ret;
    _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
    return ret;
}

static FORCEINLINE __vec16_i8 __shuffle_i8(__vec16_i8 v, __vec16_i32 index) {
    __vec16_i32 tmp_v = _mm512_extload_epi32(&v, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
    __vec16_i32 tmp   = __shuffle_i32(tmp_v, index);
    __vec16_i8 ret;
    _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
    return ret;

}

template <class RetVecType> static RetVecType __smear_i8(int8_t i);
template <> FORCEINLINE __vec16_i8 __smear_i8<__vec16_i8>(int8_t i) {
    __vec16_i32 tmp = __smear_i32<__vec16_i32>(i);
    __vec16_i8 ret;
    _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
    return ret;
}

static FORCEINLINE __vec16_i8 __shuffle2_i8(__vec16_i8 v0, __vec16_i8 v1, __vec16_i32 index) {
    __vec16_i32 tmp_v0 = _mm512_extload_epi32(&v0, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
    __vec16_i32 tmp_v1 = _mm512_extload_epi32(&v1, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
    __vec16_i32 tmp   = __shuffle2_i32(tmp_v0, tmp_v1, index);
    __vec16_i8 ret;
    _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
    return ret;
}
///////////////////////////////////////////////////////////////////////////
// int16
///////////////////////////////////////////////////////////////////////////


template <class RetVecType> static RetVecType __setzero_i16();
template <> FORCEINLINE __vec16_i16 __setzero_i16<__vec16_i16>() {
      return __vec16_i16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

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
CMP_OP(__vec16_i16, i16, uint16_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i16, i16, int16_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i16, i16, uint16_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i16, i16, int16_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i16, i16, uint16_t, __unsigned_less_than, <)
CMP_OP(__vec16_i16, i16, int16_t,  __signed_less_than, <)
CMP_OP(__vec16_i16, i16, uint16_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i16, i16, int16_t,  __signed_greater_than, >)

SELECT(__vec16_i16)

static FORCEINLINE int16_t __extract_element(__vec16_i16 v, uint32_t index) {
    return v[index];
}

static FORCEINLINE void __insert_element(__vec16_i16 *v, uint32_t index, int16_t val) {
    ((int16_t *)v)[index] = val;
}

static FORCEINLINE __vec16_i16 __broadcast_i16(__vec16_i16 v, int index) {
    int32_t val = __extract_element(v, index & 0xf);
    __vec16_i32 tmp = _mm512_set1_epi32(val);
    __vec16_i16 ret;
    _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT16,_MM_HINT_NONE);
    return ret;
}

static FORCEINLINE __vec16_i1 __not_equal_i16(__vec16_i16 a, __vec16_i16 b) {
  __vec16_i32 tmp_a = _mm512_extload_epi32(&a, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp_b = _mm512_extload_epi32(&b, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  return __not_equal_i32(tmp_a, tmp_b);
}

static FORCEINLINE __vec16_i1 __equal_i16_and_mask(const __vec16_i16 &a, const __vec16_i16 &b, __vec16_i1 m) {
  __vec16_i32 tmp_a = _mm512_extload_epi32(&a, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp_b = _mm512_extload_epi32(&b, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  return __equal_i32_and_mask(tmp_a, tmp_b, m);
}

static FORCEINLINE __vec16_i1 __not_equal_i16_and_mask(__vec16_i16 a, __vec16_i16 b, __vec16_i1 m) {
  __vec16_i32 tmp_a = _mm512_extload_epi32(&a, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp_b = _mm512_extload_epi32(&b, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  return __not_equal_i32_and_mask(tmp_a, tmp_b, m);
}

static FORCEINLINE __vec16_i16 __rotate_i16(__vec16_i16 v, int index) {
  __vec16_i32 tmp_v = _mm512_extload_epi32(&v, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp   = __rotate_i32(tmp_v, index);
  __vec16_i16 ret;
  _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT16,_MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i16 __shuffle_i16(__vec16_i16 v, __vec16_i32 index) {
  __vec16_i32 tmp_v = _mm512_extload_epi32(&v, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp   = __shuffle_i32(tmp_v, index);
  __vec16_i16 ret;
  _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT16,_MM_HINT_NONE);
  return ret;
}

template <class RetVecType> static RetVecType __smear_i16(int16_t i);
template <> FORCEINLINE __vec16_i16 __smear_i16<__vec16_i16>(int16_t i) {
  __vec16_i32 tmp = __smear_i32<__vec16_i32>(i);
  __vec16_i16 ret;
  _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT16,_MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i16 __shuffle2_i16(__vec16_i16 v0, __vec16_i16 v1, __vec16_i32 index) {
  __vec16_i32 tmp_v0 = _mm512_extload_epi32(&v0, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp_v1 = _mm512_extload_epi32(&v1, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  __vec16_i32 tmp   = __shuffle2_i32(tmp_v0, tmp_v1, index);
  __vec16_i16 ret;
  _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT16,_MM_HINT_NONE);
  return ret;
}

///////////////////////////////////////////////////////////////////////////
// various math functions
///////////////////////////////////////////////////////////////////////////

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

static FORCEINLINE __vec16_f __round_varying_float(__vec16_f v) {
  return _mm512_round_ps(v, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);
}

static FORCEINLINE __vec16_f __floor_varying_float(__vec16_f v) {
  return _mm512_floor_ps(v);
}

static FORCEINLINE __vec16_f __ceil_varying_float(__vec16_f v) {
  return _mm512_ceil_ps(v);
}

static FORCEINLINE __vec16_d __round_varying_double(__vec16_d v) {
  double tmp [16];
  for (int i = 0; i < 16; i++)
    tmp [i] = round(v [i]);
  return __vec16_d (tmp [8],  tmp [9],  tmp [10], tmp [11],
                    tmp [12], tmp [13], tmp [14], tmp [15],
                    tmp [0],  tmp [1],  tmp [2],  tmp [3],
                    tmp [4],  tmp [5],  tmp [6],  tmp [7]);
}

static FORCEINLINE __vec16_d __floor_varying_double(__vec16_d v) {
  __m512d tmp1 = _mm512_floor_pd(v.v1);
  __m512d tmp2 = _mm512_floor_pd(v.v2);
  return __vec16_d (tmp1, tmp2);
}

static FORCEINLINE __vec16_d __ceil_varying_double(__vec16_d v) {
  __m512d tmp1 = _mm512_ceil_pd(v.v1);
  __m512d tmp2 = _mm512_ceil_pd(v.v2);
  return __vec16_d (tmp1, tmp2);
}
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

static FORCEINLINE __vec16_f __max_varying_float (__vec16_f v1, __vec16_f v2) { return _mm512_gmax_ps(v1, v2); }
static FORCEINLINE __vec16_f __min_varying_float (__vec16_f v1, __vec16_f v2) { return _mm512_gmin_ps(v1, v2); }
static FORCEINLINE __vec16_d __max_varying_double(__vec16_d v1, __vec16_d v2) { return __vec16_d(_mm512_gmax_pd(v1.v1, v2.v1), _mm512_gmax_pd(v1.v2,v2.v2)); }
static FORCEINLINE __vec16_d __min_varying_double(__vec16_d v1, __vec16_d v2) { return __vec16_d(_mm512_gmin_pd(v1.v1, v2.v1), _mm512_gmin_pd(v1.v2,v2.v2)); }

static FORCEINLINE __vec16_i32 __max_varying_int32 (__vec16_i32 v1, __vec16_i32 v2) { return _mm512_max_epi32(v1, v2); }
static FORCEINLINE __vec16_i32 __min_varying_int32 (__vec16_i32 v1, __vec16_i32 v2) { return _mm512_min_epi32(v1, v2); }
static FORCEINLINE __vec16_i32 __max_varying_uint32(__vec16_i32 v1, __vec16_i32 v2) { return _mm512_max_epu32(v1, v2); }
static FORCEINLINE __vec16_i32 __min_varying_uint32(__vec16_i32 v1, __vec16_i32 v2) { return _mm512_min_epu32(v1, v2); }

static FORCEINLINE __vec16_i64 __max_varying_int64 (__vec16_i64 v1, __vec16_i64 v2) {
  __vec16_i64 ret;
  ret.v_hi = _mm512_max_epi32(v1.v_hi, v2.v_hi);
  __vec16_i1 mask = _mm512_cmpeq_epi32_mask(ret.v_hi, v2.v_hi);
  ret.v_lo = _mm512_mask_max_epi32(v1.v_lo, mask, v1.v_lo, v2.v_lo);  
  return ret;
}

static FORCEINLINE __vec16_i64 __min_varying_int64 (__vec16_i64 v1, __vec16_i64 v2) {
  __vec16_i64 ret;
  ret.v_hi = _mm512_min_epi32(v1.v_hi, v2.v_hi);
  __vec16_i1 mask = _mm512_cmpeq_epi32_mask(ret.v_hi, v2.v_hi);
  ret.v_lo = _mm512_mask_min_epi32(v1.v_lo, mask, v1.v_lo, v2.v_lo);
  return ret;
}

static FORCEINLINE __vec16_i64 __max_varying_uint64 (__vec16_i64 v1, __vec16_i64 v2) {
  __vec16_i64 ret;
  ret.v_hi = _mm512_max_epu32(v1.v_hi, v2.v_hi);
  __vec16_i1 mask = _mm512_cmpeq_epi32_mask(ret.v_hi, v2.v_hi);
  ret.v_lo = _mm512_mask_max_epu32(v1.v_lo, mask, v1.v_lo, v2.v_lo);
  return ret;
}

static FORCEINLINE __vec16_i64 __min_varying_uint64 (__vec16_i64 v1, __vec16_i64 v2) {
  __vec16_i64 ret;
  ret.v_hi = _mm512_min_epu32(v1.v_hi, v2.v_hi);
  __vec16_i1 mask = _mm512_cmpeq_epi32_mask(ret.v_hi, v2.v_hi);
  ret.v_lo = _mm512_mask_min_epu32(v1.v_lo, mask, v1.v_lo, v2.v_lo);
  return ret;
}

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

static FORCEINLINE __vec16_f __sqrt_varying_float(__vec16_f v) {
  return _mm512_sqrt_ps(v);
}

static FORCEINLINE __vec16_f __rcp_varying_float(__vec16_f v) {
#ifdef ISPC_FAST_MATH
  return _mm512_rcp23_ps(v); // Approximation with 23 bits of accuracy.
#else
  return _mm512_recip_ps(v);
#endif
}
static FORCEINLINE __vec16_d __rcp_varying_double(__vec16_d x) {
  __vec16_d y;
  for (int i = 0; i < 16; i++)
    __insert_element(&y, i, 1.0/__extract_element(x,i));
  return y;
}
static FORCEINLINE double __rcp_uniform_double(double v) 
{
  return 1.0/v;
}


static FORCEINLINE __vec16_f __rsqrt_varying_float(__vec16_f v) {
#ifdef ISPC_FAST_MATH
  return _mm512_rsqrt23_ps(v); // Approximation with 0.775ULP accuracy
#else 
  return _mm512_invsqrt_ps(v);
#endif
}

static FORCEINLINE __vec16_d __rsqrt_varying_double(__vec16_d x) {
  __vec16_d y;
  for (int i = 0; i < 16; i++)
    __insert_element(&y, i, 1.0/sqrt(__extract_element(x,i)));
  return y;
}

static FORCEINLINE double __rsqrt_uniform_double(double v) 
{
  return 1.0/v;
}


///////////////////////////////////////////////////////////////////////////
// bit ops
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE int32_t __popcnt_int32(const __vec1_i32 mask) {
  return _mm_countbits_32(mask);
}

static FORCEINLINE int32_t __popcnt_int64(const __vec1_i64 mask) {
  return _mm_countbits_64(mask);
}


static FORCEINLINE int32_t __count_trailing_zeros_i32(const __vec1_i32 mask) {
  return _mm_tzcnt_32(mask);
}

static FORCEINLINE int64_t __count_trailing_zeros_i64(const __vec1_i64 mask) {
  return _mm_tzcnt_64(mask);
}

static FORCEINLINE int32_t __count_leading_zeros_i32(__vec1_i32 mask) { 
  uint32_t n = 0;
  if (mask == 0) 
    return 32;
  while (1) {
    if (mask < 0) break;
      n ++;
      mask <<= 1;
  }
  return n;
}

static FORCEINLINE void __insert_element(__vec1_i32 *v, int index, uint32_t val) { 
    ((uint32_t *)v)[index] = val;                                        \
}

static FORCEINLINE int64_t __count_leading_zeros_i64(__vec1_i64 mask) {
  uint32_t n = 0;
  if (mask == 0)
    return 64;
  while (1) {
    if (mask < 0) break;
      n ++;
      mask <<= 1;
  }
  return n;
}

///////////////////////////////////////////////////////////////////////////
// reductions
///////////////////////////////////////////////////////////////////////////

REDUCE_ADD   ( int16_t, __vec16_i8,  __reduce_add_int8)
REDUCE_ADD   ( int32_t, __vec16_i16, __reduce_add_int16)

static FORCEINLINE int32_t __reduce_add_int32(__vec16_i32 v) {
  return _mm512_reduce_add_epi32(v);
}

static FORCEINLINE int32_t __reduce_min_int32(__vec16_i32 v) {
  return _mm512_reduce_min_epi32(v);
}

static FORCEINLINE int32_t __reduce_max_int32(__vec16_i32 v) {
  return _mm512_reduce_max_epi32(v);
}

static FORCEINLINE uint32_t __reduce_min_uint32(__vec16_i32 v) {
  return _mm512_reduce_min_epu32(v);
}

static FORCEINLINE uint32_t __reduce_max_uint32(__vec16_i32 v) {
  return _mm512_reduce_max_epu32(v);
}

static FORCEINLINE int64_t __reduce_add_int64(__vec16_i64 v) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(v, tmp1, tmp2);
#if __INTEL_COMPILER < 1500
  int64_t res1 = _mm512_reduce_add_epi64((__m512)tmp1);
  int64_t res2 = _mm512_reduce_add_epi64((__m512)tmp2);
#else
  int64_t res1 = _mm512_reduce_add_epi64(tmp1);
  int64_t res2 = _mm512_reduce_add_epi64(tmp2);
#endif
  return res1 + res2;
}

static FORCEINLINE int64_t __reduce_min_int64(__vec16_i64 v) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(v, tmp1, tmp2);
#if __INTEL_COMPILER < 1500
  int64_t res1 = _mm512_reduce_min_epi64((__m512)tmp1);
  int64_t res2 = _mm512_reduce_min_epi64((__m512)tmp2);
#else
  int64_t res1 = _mm512_reduce_min_epi64(tmp1);
  int64_t res2 = _mm512_reduce_min_epi64(tmp2);
#endif
  return (res1 < res2) ? res1 : res2;
}

static FORCEINLINE int64_t __reduce_max_int64(__vec16_i64 v) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(v, tmp1, tmp2);
#if __INTEL_COMPILER < 1500
  int64_t res1 = _mm512_reduce_max_epi64((__m512)tmp1);
  int64_t res2 = _mm512_reduce_max_epi64((__m512)tmp2);
#else
  int64_t res1 = _mm512_reduce_max_epi64(tmp1);
  int64_t res2 = _mm512_reduce_max_epi64(tmp2);
#endif
  return (res1 > res2) ? res1 : res2;
}

static FORCEINLINE uint64_t __reduce_min_uint64(__vec16_i64 v) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(v, tmp1, tmp2);
#if __INTEL_COMPILER < 1500
  uint64_t res1 = _mm512_reduce_min_epu64((__m512)tmp1);
  uint64_t res2 = _mm512_reduce_min_epu64((__m512)tmp2);
#else
  uint64_t res1 = _mm512_reduce_min_epu64(tmp1);
  uint64_t res2 = _mm512_reduce_min_epu64(tmp2);
#endif
  return (res1 < res2) ? res1 : res2;
}

static FORCEINLINE uint64_t __reduce_max_uint64(__vec16_i64 v) {
  __m512i tmp1;
  __m512i tmp2;
  hilo2zmm(v, tmp1, tmp2);
#if __INTEL_COMPILER < 1500
  uint64_t res1 = _mm512_reduce_max_epu64((__m512)tmp1);
  uint64_t res2 = _mm512_reduce_max_epu64((__m512)tmp2);
#else
  uint64_t res1 = _mm512_reduce_max_epu64(tmp1);
  uint64_t res2 = _mm512_reduce_max_epu64(tmp2);
#endif
  return (res1 > res2) ? res1 : res2;
}

static FORCEINLINE float __reduce_add_float(__vec16_f v) {
  return _mm512_reduce_add_ps(v);
}

static FORCEINLINE float __reduce_min_float(__vec16_f v) {
  return _mm512_reduce_min_ps(v);
}

static FORCEINLINE float __reduce_max_float(__vec16_f v) {
  return _mm512_reduce_max_ps(v);
}

static FORCEINLINE float __reduce_add_double(__vec16_d v) {
  return _mm512_reduce_add_pd(v.v1) + _mm512_reduce_add_pd(v.v2);
}

static FORCEINLINE float __reduce_min_double(__vec16_d v) {
  return std::min(_mm512_reduce_min_pd(v.v1), _mm512_reduce_min_pd(v.v2));
}

static FORCEINLINE float __reduce_max_double(__vec16_d v) {
  return std::max(_mm512_reduce_max_pd(v.v1), _mm512_reduce_max_pd(v.v2));
}

///////////////////////////////////////////////////////////////////////////
// masked load/store
///////////////////////////////////////////////////////////////////////////

// Currently, when a pseudo_gather is converted into a masked load, it has to be unaligned
static FORCEINLINE __vec16_i32 __masked_load_i32(void *p, __vec16_i1 mask) {
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

static FORCEINLINE __vec16_f __masked_load_float(void *p, __vec16_i1 mask) {
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

static FORCEINLINE __vec16_i64 __masked_load_i64(void *p, __vec16_i1 mask) {
  __vec16_i32 first8 = __masked_load_i32(p, mask);
  __vec16_i32 second8 = __masked_load_i32(p + 64, mask);
  return zmm2hilo(first8, second8);
}

static FORCEINLINE __vec16_d __masked_load_double(void *p, __vec16_i1 mask) {
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

static FORCEINLINE void __masked_store_i8(void *p, const __vec16_i8 &val, __vec16_i1 mask) { 
  __vec16_i32 tmp = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  _mm512_mask_extstore_epi32(p, mask, tmp, _MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
#else
  #if 0 // TODO: both implementations seem to work, need to test which one is faster
  _mm512_mask_i32extscatter_epi32 (p, mask, __vec16_i32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), tmp, _MM_DOWNCONV_EPI32_SINT8, sizeof(uint8_t), _MM_HINT_NONE);
  #else
  __vec16_i32 tmp_;
  tmp_.v = _mm512_extloadunpacklo_epi32(tmp_.v, p, _MM_UPCONV_EPI32_SINT8, _MM_HINT_NONE);
  tmp_.v = _mm512_extloadunpackhi_epi32(tmp_.v, (uint8_t*)p+64, _MM_UPCONV_EPI32_SINT8, _MM_HINT_NONE);
  tmp_.v = _mm512_mask_mov_epi32(tmp_.v, mask, tmp.v);
  _mm512_extpackstorelo_epi32(p, tmp_.v, _MM_DOWNCONV_EPI32_SINT8, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+64, tmp_.v, _MM_DOWNCONV_EPI32_SINT8, _MM_HINT_NONE);
  #endif // if 0
#endif
}

static FORCEINLINE __vec16_i8 __masked_load_i8(void *p, __vec16_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  __vec16_i32 tmp = _mm512_mask_extload_epi32(_mm512_undefined_epi32(), mask, p, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#else
  __vec16_i32 tmp;
  tmp.v = _mm512_mask_extloadunpacklo_epi32(tmp.v, 0xFFFF, p, _MM_UPCONV_EPI32_SINT8, _MM_HINT_NONE);
  tmp.v = _mm512_mask_extloadunpackhi_epi32(tmp.v, 0xFFFF, (uint8_t*)p+64, _MM_UPCONV_EPI32_SINT8, _MM_HINT_NONE);
#endif
  __vec16_i8 ret;
  _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT8, _MM_HINT_NONE);
  return ret;
}

template <int ALIGN> static FORCEINLINE __vec16_i8 __load(const __vec16_i8 *p) {
  return *p;
}
template <int ALIGN> static FORCEINLINE void __store(__vec16_i8 *p, __vec16_i8 v) {
  *p = v;
}

static FORCEINLINE void
__scatter_base_offsets32_i8(uint8_t *b, uint32_t scale, __vec16_i32 offsets,
    __vec16_i8 val, __vec16_i1 mask)
{
  __vec16_i32 tmp = _mm512_extload_epi32(&val,_MM_UPCONV_EPI32_SINT8, 
      _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  _mm512_mask_i32extscatter_epi32(b, mask, offsets, tmp, 
      _MM_DOWNCONV_EPI32_SINT8, scale, 
      _MM_HINT_NONE);
}


static FORCEINLINE void
__scatter_base_offsets32_i16(uint8_t *b, uint32_t scale, __vec16_i32 offsets,
    __vec16_i16 val, __vec16_i1 mask)
{
  __vec16_i32 tmp = _mm512_extload_epi32(&val,_MM_UPCONV_EPI32_SINT16, 
      _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  _mm512_mask_i32extscatter_epi32(b, mask, offsets, tmp, 
      _MM_DOWNCONV_EPI32_SINT16, scale, 
      _MM_HINT_NONE);
}


static FORCEINLINE void __masked_store_i16(void *p, const __vec16_i16 &val, __vec16_i1 mask) {
  __vec16_i32 tmp = _mm512_extload_epi32(&val, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  _mm512_mask_extstore_epi32(p, mask, tmp, _MM_DOWNCONV_EPI32_SINT16, _MM_HINT_NONE);
#else
  #if 0 // TODO: both implementations seem to work, need to test which one is faster
  _mm512_mask_i32extscatter_epi32 (p, mask, __vec16_i32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), tmp, _MM_DOWNCONV_EPI32_SINT16, sizeof(uint16_t), _MM_HINT_NONE);
  #else
  __vec16_i32 tmp_;
  tmp_.v = _mm512_extloadunpacklo_epi32(tmp_.v, p, _MM_UPCONV_EPI32_SINT16, _MM_HINT_NONE);
  tmp_.v = _mm512_extloadunpackhi_epi32(tmp_.v, (uint8_t*)p+64, _MM_UPCONV_EPI32_SINT16, _MM_HINT_NONE);
  tmp_.v = _mm512_mask_mov_epi32(tmp_.v, mask, tmp.v);
  _mm512_extpackstorelo_epi32(p, tmp_.v, _MM_DOWNCONV_EPI32_SINT16, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+64, tmp_.v, _MM_DOWNCONV_EPI32_SINT16, _MM_HINT_NONE);
  #endif // if 0
#endif
}

static FORCEINLINE __vec16_i16 __masked_load_i16(void *p, __vec16_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  __vec16_i32 tmp = _mm512_mask_extload_epi32(_mm512_undefined_epi32(), mask, p, _MM_UPCONV_EPI32_SINT16, _MM_BROADCAST32_NONE, _MM_HINT_NONE);
#else
  __vec16_i32 tmp;
  tmp.v = _mm512_mask_extloadunpacklo_epi32(tmp.v, 0xFFFF, p, _MM_UPCONV_EPI32_SINT16, _MM_HINT_NONE);
  tmp.v = _mm512_mask_extloadunpackhi_epi32(tmp.v, 0xFFFF, (uint8_t*)p+64, _MM_UPCONV_EPI32_SINT16, _MM_HINT_NONE);
#endif
  __vec16_i16 ret;
  _mm512_extstore_epi32(&ret, tmp, _MM_DOWNCONV_EPI32_SINT16, _MM_HINT_NONE);
  return ret;
}


template <int ALIGN> static FORCEINLINE __vec16_i16 __load(const __vec16_i16 *p) {
  return *p;
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_i16 *p, __vec16_i16 v) {
  *p = v;
}


static FORCEINLINE void __masked_store_i32(void *p, __vec16_i32 val, __vec16_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  _mm512_mask_store_epi32(p, mask, val.v);
#else
  __vec16_i32 tmp;
  tmp.v = _mm512_extloadunpacklo_epi32(tmp.v, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  tmp.v = _mm512_extloadunpackhi_epi32(tmp.v, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  tmp.v = _mm512_mask_mov_epi32(tmp.v, mask, val.v);
  _mm512_extpackstorelo_epi32(p, tmp.v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_epi32((uint8_t*)p+64, tmp.v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
#endif
}

static FORCEINLINE void __masked_store_float(void *p, __vec16_f val,
    __vec16_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  _mm512_mask_store_ps(p, mask, val.v);
#else
  __vec16_f tmp;
  tmp.v = _mm512_extloadunpacklo_ps(tmp.v, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  tmp.v = _mm512_extloadunpackhi_ps(tmp.v, (uint8_t*)p+64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
  tmp.v = _mm512_mask_mov_ps(tmp.v, mask, val.v);
  _mm512_extpackstorelo_ps(p, tmp.v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_ps((uint8_t*)p+64, tmp.v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
}

static FORCEINLINE void __masked_store_double(void *p, __vec16_d val,
    __vec16_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
  __vec16_i1 tmp_m = mask;
  tmp_m = _mm512_kswapb(tmp_m, tmp_m);
  _mm512_mask_store_pd(p, mask, val.v1);
  _mm512_mask_store_pd((uint8_t*)p+64, tmp_m, val.v2);
#else
  __vec16_d tmp;
  __vec16_i1 tmp_m = mask;
  tmp_m = _mm512_kswapb(tmp_m, tmp_m);
  tmp.v1 = _mm512_extloadunpacklo_pd(tmp.v1, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  tmp.v1 = _mm512_extloadunpackhi_pd(tmp.v1, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  tmp.v2 = _mm512_extloadunpacklo_pd(tmp.v2, (uint8_t*)p+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  tmp.v2 = _mm512_extloadunpackhi_pd(tmp.v2, (uint8_t*)p+128, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  tmp.v1 = _mm512_mask_mov_pd(tmp.v1, mask, val.v1);
  tmp.v2 = _mm512_mask_mov_pd(tmp.v2, tmp_m, val.v2);
  _mm512_extpackstorelo_pd(p, tmp.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_pd((uint8_t*)p+64, tmp.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorelo_pd((uint8_t*)p+64, tmp.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_pd((uint8_t*)p+128, tmp.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
#endif
}

static FORCEINLINE void __masked_store_blend_i32(void *p, __vec16_i32 val,
    __vec16_i1 mask) {
  __masked_store_i32(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_float(void *p, __vec16_f val,
    __vec16_i1 mask) {
  __masked_store_float(p, val, mask);
}

///////////////////////////////////////////////////////////////////////////
// gather/scatter
///////////////////////////////////////////////////////////////////////////

// offsets * offsetScale is in bytes (for all of these)

static FORCEINLINE __vec16_i8
__gather_base_offsets32_i8(uint8_t *base, uint32_t scale, __vec16_i32 offsets, 
    __vec16_i1 mask) {
  // (iw): need to temporarily store as int because gathers can only return ints.
  __vec16_i32 tmp = _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask, offsets, base, 
      _MM_UPCONV_EPI32_SINT8, scale,
      _MM_HINT_NONE);
  // now, downconverting to chars into temporary char vector
  __vec16_i8 ret;
  _mm512_extstore_epi32(ret.v,tmp,_MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i16
__gather_base_offsets32_i16(uint8_t *base, uint32_t scale, __vec16_i32 offsets, 
    __vec16_i1 mask) {
  // (iw): need to temporarily store as int because gathers can only return ints.
  __vec16_i32 tmp = _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask, offsets, base, 
      _MM_UPCONV_EPI32_SINT16, scale,
      _MM_HINT_NONE);
  // now, downconverting to chars into temporary char vector
  __vec16_i16 ret;
  _mm512_extstore_epi32(ret.v,tmp,_MM_DOWNCONV_EPI32_SINT16,_MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i32
__gather_base_offsets32_i32(uint8_t *base, uint32_t scale, __vec16_i32 offsets, 
    __vec16_i1 mask) {
  return _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask, offsets, 
      base, _MM_UPCONV_EPI32_NONE, scale,
      _MM_HINT_NONE);
}

static FORCEINLINE __vec16_f
__gather_base_offsets32_float(uint8_t *base, uint32_t scale, __vec16_i32 offsets,
    __vec16_i1 mask) { 
  return _mm512_mask_i32extgather_ps(_mm512_undefined_ps(), mask, offsets,
      base, _MM_UPCONV_PS_NONE, scale,
      _MM_HINT_NONE);
}

static FORCEINLINE __vec16_d
__gather_base_offsets32_double(uint8_t *base, uint32_t scale, __vec16_i32 offsets,
    __vec16_i1 mask) { 
  __vec16_d ret;
  ret.v1 = _mm512_mask_i32loextgather_pd(_mm512_undefined_pd(), mask, offsets,
      base, _MM_UPCONV_PD_NONE, scale,
      _MM_HINT_NONE); 
  __m512i shuffled_offsets = _mm512_permute4f128_epi32(offsets.v, _MM_PERM_DCDC);
  ret.v2 = _mm512_mask_i32loextgather_pd(_mm512_undefined_pd(), mask, shuffled_offsets,
      base, _MM_UPCONV_PD_NONE, scale,
      _MM_HINT_NONE); 
  return ret;
}

static FORCEINLINE __vec16_i32
__gather64_i32(__vec16_i64 addr, __vec16_i1 mask) 
{
  __vec16_i32 ret;

  // There is no gather instruction with 64-bit offsets in KNC.
  // We have to manually iterate over the upper 32 bits ;-)
  __vec16_i1 still_to_do = mask;
  const __vec16_i32 signed_offsets = _mm512_add_epi32(addr.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&addr.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(still_to_do, addr.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((((unsigned long)hi32) << 32) + (unsigned long)(-(long)INT_MIN));

    ret.v = _mm512_mask_i32extgather_epi32(ret.v, match, signed_offsets, 
        base, _MM_UPCONV_EPI32_NONE, 1,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match, still_to_do);
  }

  return ret;
}

static FORCEINLINE __vec16_f
__gather64_float(__vec16_i64 addr, __vec16_i1 mask) 
{
  __vec16_f ret;

  // There is no gather instruction with 64-bit offsets in KNC.
  // We have to manually iterate over the upper 32 bits ;-)
  __vec16_i1 still_to_do = mask;
  const __vec16_i32 signed_offsets = _mm512_add_epi32(addr.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&addr.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(still_to_do,addr.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((((unsigned long)hi32) << 32) + (unsigned long)(-(long)INT_MIN));

    ret.v = _mm512_mask_i32extgather_ps(ret.v, match, signed_offsets, 
        base, _MM_UPCONV_PS_NONE, 1,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match, still_to_do);
  }

  return ret;
}

static FORCEINLINE __vec16_d
__gather64_double(__vec16_i64 addr, __vec16_i1 mask) 
{
  __vec16_d ret;
  
  __vec16_i32 addr_lo, addr_hi;
  hilo2zmm(addr, addr_lo.v, addr_hi.v);

#if __INTEL_COMPILER < 1500
  ret.v1 = (__m512d)_mm512_i64extgather_pd ((__m512)addr_lo.v, 0, _MM_UPCONV_PD_NONE, 1, _MM_HINT_NONE);
  ret.v2 = (__m512d)_mm512_i64extgather_pd ((__m512)addr_hi.v, 0, _MM_UPCONV_PD_NONE, 1, _MM_HINT_NONE);
#else
  ret.v1 = _mm512_i64extgather_pd (addr_lo, 0, _MM_UPCONV_PD_NONE, 1, _MM_HINT_NONE);
  ret.v2 = _mm512_i64extgather_pd (addr_hi, 0, _MM_UPCONV_PD_NONE, 1, _MM_HINT_NONE);
#endif
  return ret;
}



/*! gather with 64-bit offsets.

  \todo add optimization that falls back to 32-bit offset gather if
  upper 32 bits are all 0es (in practice, offsets are usually array
  indices, and _usually_ <4G even if the compiler cannot statically
  figure out that this is the case */

static FORCEINLINE __vec16_f
__gather_base_offsets64_float(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_i1 mask) {

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

static FORCEINLINE __vec16_d
__gather_base_offsets64_double(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_i1 mask) {

  const __vec16_i32 signed_offsets = _mm512_add_epi32(offsets.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  const __m512i shuffled_signed_offsets = _mm512_permute4f128_epi32(signed_offsets.v, _MM_PERM_DCDC);

  // There is no gather instruction with 64-bit offsets in KNC.
  // We have to manually iterate over the upper 32 bits ;-)
  __vec16_i1 still_to_do = mask;
  __vec16_d ret;
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);
    void * base = (void*)((unsigned long)_base  +
        ((scale*(unsigned long)hi32) << 32) + scale*(unsigned long)(-(long)INT_MIN));

    // Extracting double
    ret.v1 = _mm512_mask_i32loextgather_pd(ret.v1, match, signed_offsets, base,
        _MM_UPCONV_PD_NONE, scale,
        _MM_HINT_NONE);

    ret.v2 = _mm512_mask_i32loextgather_pd(ret.v2, match, shuffled_signed_offsets, base, 
        _MM_UPCONV_PD_NONE, scale,
        _MM_HINT_NONE); 
    // --

    // Updating mask
    still_to_do = _mm512_kxor(match, still_to_do);
  }

  return ret;
}

static FORCEINLINE __vec16_i8 __gather_base_offsets64_i8(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_i1 mask) 
{ 
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
  _mm512_extstore_epi32(ret.v,tmp,_MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i8
__gather64_i8(__vec16_i64 addr, __vec16_i1 mask) 
{
  return __gather_base_offsets64_i8(0, 1, addr, mask);
}

static FORCEINLINE __vec16_i16 __gather_base_offsets64_i16(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_i1 mask) 
{ 
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
        _MM_UPCONV_EPI32_SINT16, scale,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match,still_to_do);
  }
  __vec16_i16 ret;
  _mm512_extstore_epi32(ret.v,tmp,_MM_DOWNCONV_EPI32_SINT16,_MM_HINT_NONE);
  return ret;
}

static FORCEINLINE __vec16_i16
__gather64_i16(__vec16_i64 addr, __vec16_i1 mask) 
{
  return __gather_base_offsets64_i16(0, 1, addr, mask);
}

static FORCEINLINE void __scatter_base_offsets64_float(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_f value,
    __vec16_i1 mask) { 

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

static FORCEINLINE void __scatter_base_offsets64_i32(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_i32 value,
    __vec16_i1 mask) { 

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

static FORCEINLINE void __scatter_base_offsets64_i64(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_i64 value,
    __vec16_i1 mask) { 

  const __vec16_i32 signed_offsets = _mm512_add_epi32(offsets.v_lo, __smear_i32<__vec16_i32>((int32_t)INT_MIN));
  __vec16_i1 still_to_do = mask;
  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((unsigned long)_base  + ((scale*(unsigned long)hi32) << 32) + scale*(unsigned long)(-(long)INT_MIN));  
   
    _mm512_mask_i32extscatter_epi32(base, match, signed_offsets, value.v_lo, _MM_DOWNCONV_EPI32_NONE, scale, _MM_HINT_NONE);
    _mm512_mask_i32extscatter_epi32(base + sizeof(uint32_t), match, signed_offsets, value.v_hi, _MM_DOWNCONV_EPI32_NONE, scale, _MM_HINT_NONE);

    still_to_do = _mm512_kxor(match,still_to_do);
  }
}

static FORCEINLINE void // TODO
__scatter_base_offsets64_i8(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_i8 value,
    __vec16_i1 mask) { 
  __vec16_i1 still_to_do = mask;

  __vec16_i32 tmp = _mm512_extload_epi32(&value, _MM_UPCONV_EPI32_SINT8,
      _MM_BROADCAST32_NONE, _MM_HINT_NONE);
  // _mm512_mask_extstore_epi32(p, mask, tmp, _MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);

  while (still_to_do) {
    int first_active_lane = _mm_tzcnt_32((int)still_to_do);
    const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
    __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
        __smear_i32<__vec16_i32>((int32_t)hi32),
        _MM_CMPINT_EQ);

    void * base = (void*)((unsigned long)_base  +
        ((scale*(unsigned long)hi32) << 32));    
    _mm512_mask_i32extscatter_epi32(base, match, offsets.v_lo, 
        tmp,
        _MM_DOWNCONV_EPI32_SINT8, scale,
        _MM_HINT_NONE);
    still_to_do = _mm512_kxor(match,still_to_do);
  }
}


static FORCEINLINE __vec16_i32
__gather_base_offsets64_i32(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
    __vec16_i1 mask) 
{
  __vec16_f r = __gather_base_offsets64_float(_base,scale,offsets,mask);
  return (__vec16_i32&)r;
}

// scatter

static FORCEINLINE void __scatter_base_offsets32_i32(uint8_t *b, uint32_t scale, __vec16_i32 offsets, __vec16_i32 val, __vec16_i1 mask)
{
  _mm512_mask_i32extscatter_epi32(b, mask, offsets, val, _MM_DOWNCONV_EPI32_NONE, scale, _MM_HINT_NONE);
}

static FORCEINLINE void __scatter_base_offsets32_i64(uint8_t *b, uint32_t scale, __vec16_i32 offsets, __vec16_i64 val, __vec16_i1 mask)
{
  _mm512_mask_i32extscatter_epi32(b, mask, offsets, val.v_lo, _MM_DOWNCONV_EPI32_NONE, scale, _MM_HINT_NONE);
  _mm512_mask_i32extscatter_epi32(b + sizeof(uint32_t), mask, offsets, val.v_hi, _MM_DOWNCONV_EPI32_NONE, scale, _MM_HINT_NONE);
}

static FORCEINLINE void __scatter_base_offsets32_float(void *base, uint32_t scale, __vec16_i32 offsets, __vec16_f val, __vec16_i1 mask) 
{ 
  _mm512_mask_i32extscatter_ps(base, mask, offsets, val, _MM_DOWNCONV_PS_NONE, scale, _MM_HINT_NONE);
}

static FORCEINLINE void __scatter_base_offsets32_double(void *base, uint32_t scale, __vec16_i32 offsets, __vec16_d val, __vec16_i1 mask) 
{ 
  _mm512_mask_i32loextscatter_pd(base, mask, offsets, val.v1, _MM_DOWNCONV_PD_NONE, scale, _MM_HINT_NONE);
  __m512i shuffled_offsets = _mm512_permute4f128_epi32(offsets.v, _MM_PERM_DCDC);
  const __mmask8 mask8 = 0x00FF & (mask >> 8);
  _mm512_mask_i32loextscatter_pd(base, mask8, shuffled_offsets, val.v2, _MM_DOWNCONV_PD_NONE, scale, _MM_HINT_NONE);
}

static FORCEINLINE void __scatter64_float(__vec16_i64 ptrs, __vec16_f val, __vec16_i1 mask){
#if __INTEL_COMPILER < 1500
  #warning "__scatter64_float is slow due to outdated compiler"
  __scatter_base_offsets64_float(0, 1, ptrs, val, mask);
#else
  __vec16_i32 first8ptrs, second8ptrs;
  hilo2zmm(ptrs, first8ptrs.v, second8ptrs.v);
  _mm512_mask_i64scatter_pslo (0, mask, first8ptrs, val, 1);
  const __mmask8 mask_hi = 0x00FF & (mask >> 8);
  _mm512_mask_i64scatter_pslo (0, mask_hi, second8ptrs, _mm512_permute4f128_ps(val.v, _MM_PERM_DCDC), 1);
#endif
}

static FORCEINLINE void __scatter64_i32(__vec16_i64 ptrs, __vec16_i32 val, __vec16_i1 mask) {
#if __INTEL_COMPILER < 1500
  #warning "__scatter64_i32 is slow due to outdated compiler"
  __scatter_base_offsets64_i32(0, 1, ptrs, val, mask);
#else
  __vec16_i32 first8ptrs, second8ptrs;
  hilo2zmm(ptrs, first8ptrs.v, second8ptrs.v);
  _mm512_mask_i64scatter_epi32lo (0, mask, first8ptrs, val, 1);
  const __mmask8 mask_hi = 0x00FF & (mask >> 8);
  _mm512_mask_i64scatter_epi32lo (0, mask_hi, second8ptrs, _mm512_permute4f128_epi32(val.v, _MM_PERM_DCDC), 1);
#endif
}

static FORCEINLINE void __scatter64_i64(__vec16_i64 ptrs, __vec16_i64 val, __vec16_i1 mask) {
#if __INTEL_COMPILER < 1500
  #warning "__scatter64_i64 is slow due to outdated compiler"
  __scatter_base_offsets64_i64(0, 1, ptrs, val, mask);
#else
  __vec16_i32 first8ptrs, second8ptrs;
  hilo2zmm(ptrs, first8ptrs.v, second8ptrs.v);
  __vec16_i32 first8vals, second8vals;
  hilo2zmm(val, first8vals.v, second8vals.v);
  _mm512_mask_i64extscatter_epi64 (0, mask, first8ptrs, first8vals, _MM_DOWNCONV_EPI64_NONE, 1, _MM_HINT_NONE);
  const __mmask8 mask8 = 0x00FF & (mask >> 8);
  _mm512_mask_i64extscatter_epi64 (0, mask8, second8ptrs, second8vals, _MM_DOWNCONV_EPI64_NONE, 1, _MM_HINT_NONE);
#endif
}


///////////////////////////////////////////////////////////////////////////
// packed load/store
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE int32_t __packed_load_active(uint32_t *p, __vec16_i32 *val, __vec16_i1 mask) {
  __vec16_i32 v = __load<64>(val);
  v = _mm512_mask_extloadunpacklo_epi32(v, mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  v = _mm512_mask_extloadunpackhi_epi32(v, mask, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
  __store<64>(val, v);
  return _mm_countbits_32(uint32_t(mask));
}

static FORCEINLINE int32_t __packed_store_active(uint32_t *p, __vec16_i32 val, __vec16_i1 mask) {
  _mm512_mask_extpackstorelo_epi32(p, mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  _mm512_mask_extpackstorehi_epi32((uint8_t*)p+64, mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
  return _mm_countbits_32(uint32_t(mask));
}

static FORCEINLINE int32_t __packed_store_active2(uint32_t *p, __vec16_i32 val, __vec16_i1 mask)
{
  return __packed_store_active(p, val, mask);
}


///////////////////////////////////////////////////////////////////////////
// aos/soa
///////////////////////////////////////////////////////////////////////////


static FORCEINLINE void __soa_to_aos3_float(__vec16_f v0, __vec16_f v1, __vec16_f v2,
                                            float *ptr) {
  _mm512_i32scatter_ps(ptr, __vec16_i32(0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180), v0, 1);
  _mm512_i32scatter_ps(ptr+1, __vec16_i32(0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180), v1, 1);
  _mm512_i32scatter_ps(ptr+2, __vec16_i32(0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180), v2, 1);
}

static FORCEINLINE void __aos_to_soa3_float(float *ptr, __vec16_f *out0, __vec16_f *out1,
                                            __vec16_f *out2) {
  *out0 = _mm512_i32gather_ps(__vec16_i32(0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180), ptr, 1);
  *out1 = _mm512_i32gather_ps(__vec16_i32(0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180), ptr+1, 1);
  *out2 = _mm512_i32gather_ps(__vec16_i32(0,12,24,36,48,60,72,84,96,108,120,132,144,156,168,180), ptr+2, 1);
}

static FORCEINLINE void __soa_to_aos4_float(__vec16_f v0, __vec16_f v1, __vec16_f v2,
                                            __vec16_f v3, float *ptr) {
  /*
  __vec16_f v0 (1,5, 9,13,17,21,25,29,33,37,41,45,49,53,57,61);
  __vec16_f v1 (2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62);
  __vec16_f v2 (3,7,11,15,19,23,27,31,35,39,44,47,51,55,59,63);
  __vec16_f v3 (4,8,12,16,20,24,28,32,36,40,45,48,52,56,60,64);


  // v0 = A1 ... A16, v1 = B1 ..., v3 = D1 ... D16
  __vec16_f tmp00 = _mm512_mask_swizzle_ps (v0, 0xCCCC, v1, _MM_SWIZ_REG_BADC); // A1A2B1B2 A5A6B5B6 ...
  __vec16_f tmp01 = _mm512_mask_swizzle_ps (v0, 0x3333, v1, _MM_SWIZ_REG_BADC); // B3B4A3A4 B7B8A7A8 ...
  __vec16_f tmp02 = _mm512_mask_swizzle_ps (v2, 0xCCCC, v3, _MM_SWIZ_REG_BADC); // C1C2D1D2 ...
  __vec16_f tmp03 = _mm512_mask_swizzle_ps (v2, 0x3333, v3, _MM_SWIZ_REG_BADC); // D3D4C3C4 ...

  __vec16_f tmp10 = _mm512_mask_swizzle_ps (tmp00, 0xAAAA, tmp02, _MM_SWIZ_REG_CDAB); // A1C1B1D1 A5C5B5D5 ...
  __vec16_f tmp11 = _mm512_mask_swizzle_ps (tmp00, 0x5555, tmp02, _MM_SWIZ_REG_CDAB); // C2A2D2B2 C6A6D6B6 ...
  __vec16_f tmp12 = _mm512_mask_swizzle_ps (tmp01, 0xAAAA, tmp03, _MM_SWIZ_REG_CDAB); // DBCA ...
  __vec16_f tmp13 = _mm512_mask_swizzle_ps (tmp01, 0x5555, tmp03, _MM_SWIZ_REG_CDAB); // BDAC ...  
  */


  _mm512_i32scatter_ps(ptr, __vec16_i32(0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240), v0, 1);
  _mm512_i32scatter_ps(ptr+1, __vec16_i32(0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240), v1, 1);
  _mm512_i32scatter_ps(ptr+2, __vec16_i32(0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240), v2, 1);
  _mm512_i32scatter_ps(ptr+3, __vec16_i32(0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240), v3, 1);
}


static FORCEINLINE void __aos_to_soa4_float(float *ptr, __vec16_f *out0, __vec16_f *out1,
                                            __vec16_f *out2, __vec16_f *out3) {
  *out0 = _mm512_i32gather_ps(__vec16_i32(0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240), ptr, 1);
  *out1 = _mm512_i32gather_ps(__vec16_i32(0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240), ptr+1, 1);
  *out2 = _mm512_i32gather_ps(__vec16_i32(0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240), ptr+2, 1);
  *out3 = _mm512_i32gather_ps(__vec16_i32(0,16,32,48,64,80,96,112,128,144,160,176,192,208,224,240), ptr+3, 1);
}


///////////////////////////////////////////////////////////////////////////
// prefetch
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE void __prefetch_read_uniform_1(uint8_t *p) {
  _mm_prefetch((const char*) p, _MM_HINT_T0); // prefetch into L1$
}

static FORCEINLINE void __prefetch_read_uniform_2(uint8_t *p) {
  _mm_prefetch((const char*) p, _MM_HINT_T1); // prefetch into L2$
}

static FORCEINLINE void __prefetch_read_uniform_3(uint8_t *p) {
  // There is no L3$ on KNC, but we prefetch into L2$ instead.
  _mm_prefetch((const char*) p, _MM_HINT_T1); // prefetch into L2$
}

static FORCEINLINE void __prefetch_read_uniform_nt(uint8_t *p) {
  _mm_prefetch((const char*) p, _MM_HINT_T2); // prefetch into L2$ with non-temporal hint
  // _mm_prefetch(p, _MM_HINT_NTA); // prefetch into L1$ with non-temporal hint
}

#define PREFETCH_READ_VARYING(CACHE_NUM, HINT)                                                              \
static FORCEINLINE void __prefetch_read_varying_##CACHE_NUM##_native(uint8_t *base, uint32_t scale,         \
                                                                   __vec16_i32 offsets, __vec16_i1 mask) {  \
    _mm512_mask_prefetch_i32gather_ps (offsets, mask, base, scale, HINT);                                   \
}                                                                                                           \
static FORCEINLINE void __prefetch_read_varying_##CACHE_NUM(__vec16_i64 addr, __vec16_i1 mask) {}           \

PREFETCH_READ_VARYING(1, _MM_HINT_T0)
PREFETCH_READ_VARYING(2, _MM_HINT_T1)
// L3 prefetch is mapped to L2 cache
PREFETCH_READ_VARYING(3, _MM_HINT_T1)
PREFETCH_READ_VARYING(nt, _MM_HINT_T2)

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
  return _InterlockedAnd((LONG volatile *)p, v);
#else
  return __sync_fetch_and_and(p, v);
#endif
}

static FORCEINLINE uint32_t __atomic_or(uint32_t *p, uint32_t v) {
#ifdef _MSC_VER
  return _InterlockedOr((LONG volatile *)p, v);
#else
  return __sync_fetch_and_or(p, v);
#endif
}

static FORCEINLINE uint32_t __atomic_xor(uint32_t *p, uint32_t v) {
#ifdef _MSC_VER
  return _InterlockedXor((LONG volatile *)p, v);
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
