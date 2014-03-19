/*
  Copyright (c) 2012, Intel Corporation
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
#include <unistd.h>
#include <algorithm>

#include <immintrin.h>
#include <zmmintrin.h>

#include <iostream> // for operator<<(m512[i])
#include <iomanip>  // for operator<<(m512[i])


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

typedef __mmask16 POST_ALIGN(2) __vec16_i1;

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
    __m512i v_hi;
    __m512i v_lo;
} POST_ALIGN(64) __vec16_i64;

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
    T v[16]; 
};

PRE_ALIGN(16) struct __vec16_i8   : public vec16<int8_t> { 
    FORCEINLINE __vec16_i8() { }
    FORCEINLINE __vec16_i8(const __vec16_i8 &o);
    FORCEINLINE __vec16_i8& operator =(const __vec16_i8 &o);
    FORCEINLINE __vec16_i8(int8_t v0, int8_t v1, int8_t v2, int8_t v3, 
                           int8_t v4, int8_t v5, int8_t v6, int8_t v7,
                           int8_t v8, int8_t v9, int8_t v10, int8_t v11, 
                           int8_t v12, int8_t v13, int8_t v14, int8_t v15)
        : vec16<int8_t>(v0, v1, v2, v3, v4, v5, v6, v7,
                        v8, v9, v10, v11, v12, v13, v14, v15) { }
} POST_ALIGN(16);

PRE_ALIGN(32) struct __vec16_i16  : public vec16<int16_t> { 
    FORCEINLINE __vec16_i16() { }
    FORCEINLINE __vec16_i16(const __vec16_i16 &o);
    FORCEINLINE __vec16_i16& operator =(const __vec16_i16 &o);
    FORCEINLINE __vec16_i16(int16_t v0, int16_t v1, int16_t v2, int16_t v3, 
                            int16_t v4, int16_t v5, int16_t v6, int16_t v7,
                            int16_t v8, int16_t v9, int16_t v10, int16_t v11, 
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
    out << (i?",":"") << std::dec << std::setw(8) << ((int*)&v)[i] << std::dec;
    // out << (i?",":"") << std::hex << std::setw(8) << ((int*)&v)[i] << std::dec;
  
  out << "]" << std::flush;
  return out;
}

inline std::ostream &operator<<(std::ostream &out, const __m512 &v)
{
  out << "[";
  for (int i=0;i<16;i++)  
    out << (i?",":"") << ((float*)&v)[i];
  
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
    return _mm512_knot( _mm512_kandn(a, b));
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

template <class RetVecType> RetVecType __smear_i1(int i);
template <> static FORCEINLINE __vec16_i1 __smear_i1<__vec16_i1>(int i) {
    return i?0xFFFF:0x0;
}

template <class RetVecType> RetVecType __setzero_i1();
template <> static FORCEINLINE __vec16_i1 __setzero_i1<__vec16_i1>() {
    return 0;
}

template <class RetVecType> RetVecType __undef_i1();
template <> static FORCEINLINE __vec16_i1 __undef_i1<__vec16_i1>() {
    return __vec16_i1();
}

///////////////////////////////////////////////////////////////////////////
// int8
///////////////////////////////////////////////////////////////////////////

/*

TODO

*/


///////////////////////////////////////////////////////////////////////////
// int16
///////////////////////////////////////////////////////////////////////////

/*

TODO

*/

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

template <class RetVecType> RetVecType __smear_i32(int32_t i);
template <> static FORCEINLINE __vec16_i32 __smear_i32<__vec16_i32>(int32_t i) {
    return _mm512_set1_epi32(i);
}

static const __vec16_i32 __ispc_one = __smear_i32<__vec16_i32>(1);
static const __vec16_i32 __ispc_thirty_two = __smear_i32<__vec16_i32>(32);
static const __vec16_i32 __ispc_ffffffff = __smear_i32<__vec16_i32>(-1);
static const __vec16_i32 __ispc_stride1(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

template <class RetVecType> RetVecType __setzero_i32();
template <> static FORCEINLINE __vec16_i32 __setzero_i32<__vec16_i32>() {
    return _mm512_setzero_epi32();
}

template <class RetVecType> RetVecType __undef_i32();
template <> static FORCEINLINE __vec16_i32 __undef_i32<__vec16_i32>() {
    return __vec16_i32();
}

static FORCEINLINE __vec16_i32 __broadcast_i32(__vec16_i32 v, int index) {
    int32_t val = __extract_element(v, index & 0xf);
    return _mm512_set1_epi32(val);
}

static FORCEINLINE __vec16_i32 __rotate_i32(__vec16_i32 v, int index) {
    __vec16_i32 idx = __smear_i32<__vec16_i32>(index);
    __vec16_i32 shuffle = _mm512_and_epi32(_mm512_add_epi32(__ispc_stride1, idx),  __smear_i32<__vec16_i32>(0x7));
    return _mm512_mask_permutevar_epi32(v, 0xffff, shuffle, v);
}

static FORCEINLINE __vec16_i32 __shuffle_i32(__vec16_i32 v, __vec16_i32 index) {
    return _mm512_mask_permutevar_epi32(v, 0xffff, index, v);
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

template <> static FORCEINLINE __vec16_i32 __load<64>(const __vec16_i32 *p) {
    return _mm512_load_epi32(p);
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_i32 *p, __vec16_i32 v) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    _mm512_store_epi32(p, v);
#else
    _mm512_extpackstorelo_epi32(p, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_epi32((uint8_t*)p+64, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
#endif
}

template <> static FORCEINLINE void __store<64>(__vec16_i32 *p, __vec16_i32 v) {
    _mm512_store_epi32(p, v);
}

///////////////////////////////////////////////////////////////////////////
// int64
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE int64_t __extract_element(const __vec16_i64 &v, uint32_t index)
{
    uint *src = (uint *)&v;
    return src[index+16] | (int64_t(src[index]) << 32);
}

static FORCEINLINE void __insert_element(__vec16_i64 *v, uint32_t index, int64_t val) {
    ((int32_t *)&v->v_hi)[index] = val>>32;
    ((int32_t *)&v->v_lo)[index] = val;
}


template <class RetVecType> RetVecType __setzero_i64();
template <> static FORCEINLINE __vec16_i64 __setzero_i64<__vec16_i64>() {
    __vec16_i64 ret;
    ret.v_lo = _mm512_setzero_epi32();
    ret.v_hi = _mm512_setzero_epi32();
    return ret;
}

template <class RetVecType> RetVecType __undef_i64();
template <> static FORCEINLINE __vec16_i64 __undef_i64<__vec16_i64>() {
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

static FORCEINLINE __vec16_i64 __mul(const __vec16_i64 &a, const __vec16_i64 &b)
{
    __vec16_i32 lo = _mm512_mullo_epi32(a.v_lo,b.v_lo);
    __vec16_i32 hi_m1 = _mm512_mulhi_epi32(a.v_lo, b.v_lo);
    __vec16_i32 hi_m2 = _mm512_mullo_epi32(a.v_hi, b.v_lo);
    __vec16_i32 hi_m3 = _mm512_mullo_epi32(a.v_lo, b.v_hi);
    __mmask16 carry = 0;
    __vec16_i32 hi_p23 = _mm512_addsetc_epi32(hi_m2, hi_m1, &carry);
    __vec16_i32 hi = _mm512_adc_epi32(hi_m3, carry, hi_p23, &carry);
    return __vec16_i64(lo, hi);
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
    __vec16_i32 xfer = _mm512_srlv_epi32(a.v_lo, _mm512_sub_epi32(__ispc_thirty_two, b.v_lo));
    __vec16_i32 hi = _mm512_or_epi32(_mm512_sllv_epi32(a.v_hi, b.v_lo), xfer);
    __vec16_i32 lo = _mm512_sllv_epi32(a.v_lo, b.v_lo);
    return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i64 __lshr(__vec16_i64 a, __vec16_i64 b) {
    __vec16_i32 shift = _mm512_sub_epi32(__ispc_thirty_two, b.v_lo);
    __vec16_i32 xfer = _mm512_and_epi32(_mm512_sllv_epi32(__ispc_ffffffff, shift), _mm512_sllv_epi32(a.v_hi, shift));
    //__vec16_i32 xfer = _mm512_sllv_epi32(_mm512_and_epi32(a.v_hi, 
    //                                                      _mm512_sub_epi32(_mm512_sllv_epi32(__ispc_one, b.v_lo), __ispc_one)), 
    //                                     _mm512_sub_epi32(__ispc_thirty_two, b.v_lo));
    __vec16_i32 hi = _mm512_srlv_epi32(a.v_hi, b.v_lo);
    __vec16_i32 lo = _mm512_or_epi32(xfer, _mm512_srlv_epi32(a.v_lo, b.v_lo));
    return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i64 __ashr(__vec16_i64 a, __vec16_i64 b) {
    __vec16_i32 xfer = _mm512_sllv_epi32(_mm512_and_epi32(a.v_hi, 
                                                          _mm512_sub_epi32(_mm512_sllv_epi32(__ispc_one, b.v_lo), __ispc_one)), 
                                         _mm512_sub_epi32(__ispc_thirty_two, b.v_lo));
    __vec16_i32 hi = _mm512_srav_epi32(a.v_hi, b.v_lo);
    __vec16_i32 lo = _mm512_or_epi32(xfer, _mm512_srlv_epi32(a.v_lo, b.v_lo));
    return __vec16_i64(lo, hi);
}

static FORCEINLINE __vec16_i1 __equal_i64(const __vec16_i64 &a, const __vec16_i64 &b) {
    const __mmask16 lo_match = _mm512_cmpeq_epi32_mask(a.v_lo,b.v_lo);
    return _mm512_mask_cmpeq_epi32_mask(lo_match,a.v_hi,b.v_hi);
}

static FORCEINLINE __vec16_i1 __equal_i64_and_mask(const __vec16_i64 &a, const __vec16_i64 &b,
                                                   __vec16_i1 mask) {
    __mmask16 lo_match = _mm512_cmpeq_epi32_mask(a.v_lo,b.v_lo);
    __mmask16 full_match = _mm512_mask_cmpeq_epi32_mask(lo_match,a.v_hi,b.v_hi);
    return _mm512_kand(full_match, (__mmask16)mask);
}

static FORCEINLINE __vec16_i1 __not_equal_i64(const __vec16_i64 &a, const __vec16_i64 &b) {
    return __not(__equal_i64(a,b));
}

static FORCEINLINE __vec16_i1 __not_equal_i64_and_mask(const __vec16_i64 &a, const __vec16_i64 &b,
                                                       __vec16_i1 mask) {
    return __and(__not(__equal_i64(a,b)), mask);
}

static FORCEINLINE __vec16_i64 __select(__vec16_i1 mask,
                                        __vec16_i64 a, __vec16_i64 b) {
    __vec16_i64 ret;
    ret.v_hi = _mm512_mask_mov_epi32(b.v_hi, mask, a.v_hi);
    ret.v_lo = _mm512_mask_mov_epi32(b.v_lo, mask, a.v_lo);
    return ret;
}

template <class RetVecType> RetVecType __smear_i64(const int64_t &l);
template <> FORCEINLINE  __vec16_i64 __smear_i64<__vec16_i64>(const int64_t &l) {
    const int *i = (const int*)&l;
    return __vec16_i64(_mm512_set1_epi32(i[0]), _mm512_set1_epi32(i[1]));
}

template <int ALIGN> static FORCEINLINE __vec16_i64 __load(const __vec16_i64 *p) {
    __vec16_i32 v1;
    __vec16_i32 v2;
    v2 = _mm512_extloadunpacklo_epi32(v2, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v2 = _mm512_extloadunpackhi_epi32(v2, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v1 = _mm512_extloadunpacklo_epi32(v1, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v1 = _mm512_extloadunpackhi_epi32(v1, (uint8_t*)p+128, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);

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

template <> static FORCEINLINE __vec16_i64 __load<64>(const __vec16_i64 *p) {
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

template <> static FORCEINLINE __vec16_i64 __load<128>(const __vec16_i64 *p) {
    return __load<64>(p);
}

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

template <> static FORCEINLINE void __store<64>(__vec16_i64 *p, __vec16_i64 v) {
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

template <> static FORCEINLINE void __store<128>(__vec16_i64 *p, __vec16_i64 v) {
    __store<64>(p, v);
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

static FORCEINLINE __vec16_i1 __unordered_float(__vec16_f a, __vec16_f b) {
    return _mm512_cmpunord_ps_mask(a, b);
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

template <class RetVecType> RetVecType __smear_float(float f);
template <> static FORCEINLINE __vec16_f __smear_float<__vec16_f>(float f) {
    return _mm512_set_1to16_ps(f);
}

template <class RetVecType> RetVecType __setzero_float();
template <> static FORCEINLINE __vec16_f __setzero_float<__vec16_f>() {
    return _mm512_setzero_ps();
}

template <class RetVecType> RetVecType __undef_float();
template <> static FORCEINLINE __vec16_f __undef_float<__vec16_f>() {
    return __vec16_f();
}

static FORCEINLINE __vec16_f __broadcast_float(__vec16_f v, int index) {
    int32_t val = __extract_element(v, index & 0xf);
    return _mm512_set1_ps(val);
}

static FORCEINLINE __vec16_f __shuffle_float(__vec16_f v, __vec16_i32 index) {
    return _mm512_castsi512_ps(_mm512_mask_permutevar_epi32(_mm512_castps_si512(v), 0xffff, index, _mm512_castps_si512(v)));
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

template <> static FORCEINLINE __vec16_f __load<64>(const __vec16_f *p) {
    return _mm512_load_ps(p);
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_f *p, __vec16_f v) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    _mm512_store_ps(p, v);
#else
    _mm512_extpackstorelo_ps(p, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_ps((uint8_t*)p+64, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
}

template <> static FORCEINLINE void __store<64>(__vec16_f *p, __vec16_f v) {
    _mm512_store_ps(p, v);
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

template <class RetVecType> RetVecType __smear_double(double d);
template <> static FORCEINLINE __vec16_d __smear_double<__vec16_d>(double d) {
    __vec16_d ret;
    ret.v1 = _mm512_set1_pd(d);
    ret.v2 = _mm512_set1_pd(d);
    return ret;
}

template <class RetVecType> RetVecType __setzero_double();
template <> static FORCEINLINE __vec16_d __setzero_double<__vec16_d>() {
    __vec16_d ret;
    ret.v1 = _mm512_setzero_pd();
    ret.v2 = _mm512_setzero_pd();
    return ret;
}

template <class RetVecType> RetVecType __undef_double();
template <> static FORCEINLINE __vec16_d __undef_double<__vec16_d>() {
    return __vec16_d();
}

static FORCEINLINE __vec16_d __broadcast_double(__vec16_d v, int index) {
    __vec16_d ret;
    double val = __extract_element(v, index & 0xf);
    ret.v1 = _mm512_set1_pd(val);
    ret.v2 = _mm512_set1_pd(val);
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

template <> static FORCEINLINE __vec16_d __load<64>(const __vec16_d *p) {
    __vec16_d ret;
    ret.v1 = _mm512_load_pd(p);
    ret.v2 = _mm512_load_pd(((uint8_t*)p)+64);
    return ret;
}

template <> static FORCEINLINE __vec16_d __load<128>(const __vec16_d *p) {
    return __load<64>(p);
}
 
template <int ALIGN> static FORCEINLINE void __store(__vec16_d *p, __vec16_d v) {
    _mm512_extpackstorelo_pd(p, v.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_pd((uint8_t*)p+64, v.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_pd((uint8_t*)p+64, v.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_pd((uint8_t*)p+128, v.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
}

template <> static FORCEINLINE void __store<64>(__vec16_d *p, __vec16_d v) {
    _mm512_store_pd(p, v.v1);
    _mm512_store_pd(((uint8_t*)p)+64, v.v2);
}

template <> static FORCEINLINE void __store<128>(__vec16_d *p, __vec16_d v) {
    __store<64>(p, v);
}

///////////////////////////////////////////////////////////////////////////
// casts
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE __vec16_i64 __cast_sext(const __vec16_i64 &, const __vec16_i32 &val)
{
    return __vec16_i64(val.v,_mm512_srai_epi32(val.v,31));
}

static FORCEINLINE __vec16_i64 __cast_zext(const __vec16_i64 &, const __vec16_i32 &val)
{
    return __vec16_i64(val.v, _mm512_setzero_epi32());
}

static FORCEINLINE __vec16_i32 __cast_sext(const __vec16_i32 &, const __vec16_i1 &val)
{
  __vec16_i32 ret = _mm512_setzero_epi32();
  __vec16_i32 one = _mm512_set1_epi32(-1);
  return _mm512_mask_mov_epi32(ret, val, one);
}

static FORCEINLINE __vec16_i32 __cast_zext(const __vec16_i32 &, const __vec16_i1 &val)
{
    __vec16_i32 ret = _mm512_setzero_epi32();
    __vec16_i32 one = _mm512_set1_epi32(1);
    return _mm512_mask_mov_epi32(ret, val, one);
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

static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, const __vec16_i8 &v) {
  return _mm512_extload_ps(v.v,_MM_UPCONV_PS_UINT8,_MM_BROADCAST32_NONE,_MM_HINT_NONE);
}

static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i32 v) {
  return _mm512_cvtfxpnt_round_adjustepu32_ps(v, _MM_FROUND_NO_EXC,_MM_EXPADJ_NONE); 
}

// float/double to signed int

static FORCEINLINE __vec16_i32 __cast_fptosi(__vec16_i32, __vec16_f val) {
  return _mm512_cvtfxpnt_round_adjustps_epi32(val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

static FORCEINLINE __vec16_i32 __cast_fptoui(__vec16_i32, __vec16_f val) {
  return _mm512_cvtfxpnt_round_adjustps_epu32(val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

static FORCEINLINE __vec16_d __cast_fpext(__vec16_d, __vec16_f val) {
    __vec16_d ret;
    ret.v2 = _mm512_cvtpslo_pd(val.v);
    __vec16_f other8 = _mm512_permute4f128_epi32(_mm512_castps_si512(val.v), _MM_PERM_DCDC);
    ret.v1 = _mm512_cvtpslo_pd(other8);
    return ret;
}

static FORCEINLINE __vec16_f __cast_fptrunc(__vec16_f, __vec16_d val) {
    __m512i r0i = _mm512_castps_si512(_mm512_cvtpd_pslo(val.v1));
    __m512i r1i = _mm512_castps_si512(_mm512_cvtpd_pslo(val.v2));

    return _mm512_mask_permute4f128_epi32(r1i, 0xFF00, r0i, _MM_PERM_BABA);
}

static FORCEINLINE __vec16_f __cast_bits(__vec16_f, __vec16_i32 val) {
    return _mm512_castsi512_ps(val);
}

static FORCEINLINE __vec16_i32 __cast_bits(__vec16_i32, __vec16_f val) {
    return _mm512_castps_si512(val);
}


static FORCEINLINE __vec16_i64 __cast_bits(__vec16_i64, __vec16_d val) {
    return *(__vec16_i64*)&val;
}

static FORCEINLINE __vec16_d __cast_bits(__vec16_d, __vec16_i64 val) {
    return *(__vec16_d*)&val;
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

static FORCEINLINE __vec16_f __max_varying_float(__vec16_f v1, __vec16_f v2) {
  return _mm512_gmax_ps(v1, v2);
}

static FORCEINLINE __vec16_f __min_varying_float(__vec16_f v1, __vec16_f v2) {
  return _mm512_gmin_ps(v1, v2);
}

static FORCEINLINE __vec16_i32 __max_varying_int32(__vec16_i32 v1, __vec16_i32 v2) {
  return _mm512_max_epi32(v1, v2);
}

static FORCEINLINE __vec16_i32 __min_varying_int32(__vec16_i32 v1, __vec16_i32 v2) {
  return _mm512_min_epi32(v1, v2);
}

static FORCEINLINE __vec16_i32 __max_varying_uint32(__vec16_i32 v1, __vec16_i32 v2) {
  return _mm512_max_epu32(v1, v2);
}

static FORCEINLINE __vec16_i32 __min_varying_uint32(__vec16_i32 v1, __vec16_i32 v2) {
  return _mm512_min_epu32(v1, v2);
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

///////////////////////////////////////////////////////////////////////////
// reductions
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE int16_t __reduce_add_i8(__vec16_i8 v) {
  // TODO: improve this!
  int16_t ret = 0;
  for (int i = 0; i < 16; ++i)
    ret += v.v[i];
  return ret;
}

static FORCEINLINE int32_t __reduce_add_i16(__vec16_i16 v) {
  // TODO: improve this!
  int32_t ret = 0;
  for (int i = 0; i < 16; ++i)
    ret += v.v[i];
  return ret;
}

static FORCEINLINE uint32_t __reduce_add_i32(__vec16_i32 v) {
    return _mm512_reduce_add_epi32(v);
}

static FORCEINLINE uint32_t __reduce_min_i32(__vec16_i32 v) {
    return _mm512_reduce_min_epi32(v);
}

static FORCEINLINE uint32_t __reduce_max_i32(__vec16_i32 v) {
    return _mm512_reduce_max_epi32(v);
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
    ret.v2 = _mm512_mask_i32loextgather_pd(_mm512_undefined_pd(), mask, offsets,
                                       base, _MM_UPCONV_PD_NONE, scale,
                                       _MM_HINT_NONE); 
    __m512i shuffled_offsets = _mm512_permute4f128_epi32(offsets.v, _MM_PERM_DCDC);
    ret.v1 = _mm512_mask_i32loextgather_pd(_mm512_undefined_pd(), mask, shuffled_offsets,
                                       base, _MM_UPCONV_PD_NONE, scale,
                                       _MM_HINT_NONE); 
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
                              ((scale*(unsigned long)hi32) << 32));
        ret = _mm512_mask_i32extgather_ps(ret, match, offsets.v_lo, base,
                                          _MM_UPCONV_PS_NONE, scale,
                                          _MM_HINT_NONE);
        still_to_do = _mm512_kxor(match, still_to_do);
    }

    return ret;
}


static FORCEINLINE __vec16_i8
__gather_base_offsets64_i8(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
                           __vec16_i1 mask) 
{ 
    __vec16_i1 still_to_do = mask;
    __vec16_i32 tmp;
    while (still_to_do) {
        int first_active_lane = _mm_tzcnt_32((int)still_to_do);
        const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
        __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
                                                      __smear_i32<__vec16_i32>((int32_t)hi32),
                                                      _MM_CMPINT_EQ);
    
        void * base = (void*)((unsigned long)_base  +
                              ((scale*(unsigned long)hi32) << 32));    
        tmp = _mm512_mask_i32extgather_epi32(tmp, match, offsets.v_lo, base,
                                             _MM_UPCONV_EPI32_SINT8, scale,
                                             _MM_HINT_NONE);
        still_to_do = _mm512_kxor(match,still_to_do);
    }
    __vec16_i8 ret;
    _mm512_extstore_epi32(ret.v,tmp,_MM_DOWNCONV_EPI32_SINT8,_MM_HINT_NONE);
    return ret;
}


static FORCEINLINE void
__scatter_base_offsets64_float(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
                               __vec16_f value,
                               __vec16_i1 mask) { 
    __vec16_i1 still_to_do = mask;
    while (still_to_do) {
        int first_active_lane = _mm_tzcnt_32((int)still_to_do);
        const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
        __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
                                                      __smear_i32<__vec16_i32>((int32_t)hi32),
                                                      _MM_CMPINT_EQ);

        void * base = (void*)((unsigned long)_base  +
                              ((scale*(unsigned long)hi32) << 32));    
        _mm512_mask_i32extscatter_ps(base, match, offsets.v_lo, 
                                     value,
                                     _MM_DOWNCONV_PS_NONE, scale,
                                     _MM_HINT_NONE);
        still_to_do = _mm512_kxor(match,still_to_do);
    }
}

static FORCEINLINE void
__scatter_base_offsets64_i32(uint8_t *_base, uint32_t scale, __vec16_i64 offsets,
                             __vec16_i32 value,
                             __vec16_i1 mask) { 
    __vec16_i1 still_to_do = mask;
    while (still_to_do) {
        int first_active_lane = _mm_tzcnt_32((int)still_to_do);
        const uint &hi32 = ((uint*)&offsets.v_hi)[first_active_lane];
        __vec16_i1 match = _mm512_mask_cmp_epi32_mask(mask,offsets.v_hi,
                                                      __smear_i32<__vec16_i32>((int32_t)hi32),
                                                      _MM_CMPINT_EQ);
    
        void * base = (void*)((unsigned long)_base  +
                              ((scale*(unsigned long)hi32) << 32));    
        _mm512_mask_i32extscatter_epi32(base, match, offsets.v_lo, 
                                        value,
                                        _MM_DOWNCONV_EPI32_NONE, scale,
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

static FORCEINLINE void
__scatter_base_offsets32_i32(uint8_t *b, uint32_t scale, __vec16_i32 offsets,
                             __vec16_i32 val, __vec16_i1 mask)
{
    _mm512_mask_i32extscatter_epi32(b, mask, offsets, val, 
                                    _MM_DOWNCONV_EPI32_NONE, scale, 
                                    _MM_HINT_NONE);
}

static FORCEINLINE void 
__scatter_base_offsets32_float(void *base, uint32_t scale, __vec16_i32 offsets,
                               __vec16_f val, __vec16_i1 mask) 
{ 
    _mm512_mask_i32extscatter_ps(base, mask, offsets, val, 
                                 _MM_DOWNCONV_PS_NONE, scale,
                                 _MM_HINT_NONE);
}

///////////////////////////////////////////////////////////////////////////
// packed load/store
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE int32_t __packed_load_active(uint32_t *p, __vec16_i32 *val,
                                                __vec16_i1 mask) {
    __vec16_i32 v;
    v = _mm512_mask_extloadunpacklo_epi32(v, mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v = _mm512_mask_extloadunpackhi_epi32(_mm512_undefined_epi32(), mask, (uint8_t*)p+64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    __store<64>(val, v);
    return _mm_countbits_32(uint32_t(mask));
}

static FORCEINLINE int32_t __packed_store_active(uint32_t *p, __vec16_i32 val,
                                                 __vec16_i1 mask) {
    _mm512_mask_extpackstorelo_epi32(p, mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_mask_extpackstorehi_epi32((uint8_t*)p+64, mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    return _mm_countbits_32(uint32_t(mask));
}

static FORCEINLINE int32_t __packed_store_active2(uint32_t *p, __vec16_i32 val, __vec16_i1 mask)
{
  return __packed_store_active(p, val, mask);
}

///////////////////////////////////////////////////////////////////////////
// prefetch
///////////////////////////////////////////////////////////////////////////

static FORCEINLINE void __prefetch_read_uniform_1(const char *p) {
    _mm_prefetch(p, _MM_HINT_T0); // prefetch into L1$
}

static FORCEINLINE void __prefetch_read_uniform_2(const char *p) {
    _mm_prefetch(p, _MM_HINT_T1); // prefetch into L2$
}

static FORCEINLINE void __prefetch_read_uniform_3(const char *p) {
    // There is no L3$ on KNC, don't want to pollute L2$ unecessarily
}

static FORCEINLINE void __prefetch_read_uniform_nt(const char *p) {
    _mm_prefetch(p, _MM_HINT_T2); // prefetch into L2$ with non-temporal hint
    // _mm_prefetch(p, _MM_HINT_NTA); // prefetch into L1$ with non-temporal hint
}

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
