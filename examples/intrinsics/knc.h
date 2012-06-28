/*
  Copyright (c) 2010-2011, Intel Corporation
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

#include <immintrin.h>
#include <zmmintrin.h>

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#define PRE_ALIGN(x)  /*__declspec(align(x))*/
#define POST_ALIGN(x)  
#define roundf(x) (floorf(x + .5f))
#define round(x) (floor(x + .5))
#else
#define FORCEINLINE __attribute__((always_inline))
#define PRE_ALIGN(x)
#define POST_ALIGN(x)  __attribute__ ((aligned(x)))
#endif

#define KNC 1
extern "C" {
  int printf(const unsigned char *, ...);
}


typedef float __vec1_f;
typedef double __vec1_d;
typedef int8_t __vec1_i8;
typedef int16_t __vec1_i16;
typedef int32_t __vec1_i32;
typedef int64_t __vec1_i64;

struct __vec16_i32;


typedef struct PRE_ALIGN(2) __vec16_i1 {
    operator __mmask16() const { return m; }
    __vec16_i1() { }
    __vec16_i1(const __mmask16& in) { m = in; }
    __vec16_i1(const __vec16_i32& in);
    __vec16_i1(uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03, 
               uint32_t v04, uint32_t v05, uint32_t v06, uint32_t v07,
               uint32_t v08, uint32_t v09, uint32_t v10, uint32_t v11,
               uint32_t v12, uint32_t v13, uint32_t v14, uint32_t v15) {
        m = (v00) |
            ((v01) <<  1) |
            ((v02) <<  2) |
            ((v03) <<  3) |
            ((v04) <<  4) |
            ((v05) <<  5) |
            ((v06) <<  6) |
            ((v07) <<  7) |
            ((v08) <<  8) |
            ((v09) <<  9) |
            ((v10) << 10) |
            ((v11) << 11) |
            ((v12) << 12) |
            ((v13) << 13) |
            ((v14) << 14) |
  	    ((v15) << 15);
    }

    union {
        __mmask16 m;
        struct {
            __mmask8 m1;
            __mmask8 m2;
        } m8;
    };
} POST_ALIGN(2) __vec16_i1;

typedef struct PRE_ALIGN(64) __vec16_f {
    operator __m512() const { return v; }
    __vec16_f() { }
    __vec16_f(const __m512& in) { v = in; }
    __vec16_f(float v00, float v01, float v02, float v03, 
              float v04, float v05, float v06, float v07,
              float v08, float v09, float v10, float v11,
              float v12, float v13, float v14, float v15) {
        v = _mm512_set_16to16_ps(v15, v14, v13, v12, v11, v10, v09, v08, v07, v06, v05, v04, v03, v02, v01, v00);
    }
    __m512 v;
} POST_ALIGN(64) __vec16_f;

typedef struct PRE_ALIGN(64) __vec16_d {
    __vec16_d() { }
    __vec16_d(double v00, double v01, double v02, double v03, 
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
    operator __m512i() const { return v; }
    __vec16_i32() { }
    __vec16_i32(const __m512i& in) { v = in; }
    __vec16_i32(const __vec16_i32& in) { v = in.v; }
    __vec16_i32(int32_t v00, int32_t v01, int32_t v02, int32_t v03, 
                int32_t v04, int32_t v05, int32_t v06, int32_t v07,
                int32_t v08, int32_t v09, int32_t v10, int32_t v11,
                int32_t v12, int32_t v13, int32_t v14, int32_t v15) {
        v = _mm512_set_16to16_pi(v15, v14, v13, v12, v11, v10, v09, v08, v07, v06, v05, v04, v03, v02, v01, v00);
    }
    __m512i v;
} POST_ALIGN(64) __vec16_i32;

typedef struct PRE_ALIGN(64) __vec16_i64 {
    __vec16_i64() { }
    __vec16_i64(int64_t v00, int64_t v01, int64_t v02, int64_t v03, 
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

FORCEINLINE __vec16_i1::__vec16_i1(const __vec16_i32& in) {
    m = _mm512_test_epi32_mask(in, in);
}

template <typename T>
struct vec16 {
    vec16() { }
    vec16(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
          T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15) {
        v[0] = v0;        v[1] = v1;        v[2] = v2;        v[3] = v3;
        v[4] = v4;        v[5] = v5;        v[6] = v6;        v[7] = v7;
        v[8] = v8;        v[9] = v9;        v[10] = v10;      v[11] = v11;
        v[12] = v12;      v[13] = v13;      v[14] = v14;      v[15] = v15;
    }
    T v[16]; 
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


///////////////////////////////////////////////////////////////////////////
// macros...

#define UNARY_OP(TYPE, NAME, OP) 
/*
static FORCEINLINE TYPE NAME(TYPE v) {      \
    TYPE ret;                               \
    for (int i = 0; i < 16; ++i)            \
        ret.v[i] = OP(v.v[i]);              \
    return ret;                             \
}
*/

#define BINARY_OP(TYPE, NAME, OP)
/*
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                  \
    TYPE ret;                                                   \
   for (int i = 0; i < 16; ++i)                                 \
       ret.v[i] = a.v[i] OP b.v[i];                             \
   return ret;                                                   \
}
*/

#define BINARY_OP_CAST(TYPE, CAST, NAME, OP)
/*
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v[i] = (CAST)(a.v[i]) OP (CAST)(b.v[i]);                 \
   return ret;                                                      \
}
*/


#define BINARY_OP_FUNC(TYPE, NAME, FUNC)
/*
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v[i] = FUNC(a.v[i], b.v[i]);                             \
   return ret;                                                      \
}
*/

#define CMP_OP(TYPE, CAST, NAME, OP)
/*
static FORCEINLINE __vec16_i1 NAME(TYPE a, TYPE b) {                \
   __vec16_i1 ret;                                                  \
   ret.v = 0;                                                       \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v |= ((CAST)(a.v[i]) OP (CAST)(b.v[i])) << i;            \
   return ret;                                                      \
}
*/

#define INSERT_EXTRACT(VTYPE, STYPE)
/*
static FORCEINLINE STYPE __extract_element(VTYPE v, int index) {      \
    return ((STYPE *)&v)[index];                                      \
}                                                                     \
static FORCEINLINE void __insert_element(VTYPE *v, int index, STYPE val) { \
    ((STYPE *)v)[index] = val;                                        \
}
*/

#define LOAD_STORE(VTYPE, STYPE)
/*
static FORCEINLINE VTYPE __load(VTYPE *p, int align) { \
    STYPE *ptr = (STYPE *)p;                           \
    VTYPE ret;                                         \
    for (int i = 0; i < 16; ++i)                       \
        ret.v[i] = ptr[i];                             \
    return ret;                                        \
}                                                      \
static FORCEINLINE void __store(VTYPE *p, VTYPE v, int align) {    \
    STYPE *ptr = (STYPE *)p;                           \
    for (int i = 0; i < 16; ++i)                       \
        ptr[i] = v.v[i];                               \
}
*/

#define REDUCE_ADD(TYPE, VTYPE, NAME)
/*
static FORCEINLINE TYPE NAME(VTYPE v) {         \
     TYPE ret = v.v[0];                         \
     for (int i = 1; i < 16; ++i)               \
         ret = ret + v.v[i];                    \
     return ret;                                \
}
*/

#define REDUCE_MINMAX(TYPE, VTYPE, NAME, OP)
/*
static FORCEINLINE TYPE NAME(VTYPE v) {                         \
    TYPE ret = v.v[0];                                          \
    for (int i = 1; i < 16; ++i)                                \
        ret = (ret OP (TYPE)v.v[i]) ? ret : (TYPE)v.v[i];       \
    return ret;                                                 \
}
*/

#define SELECT(TYPE)
/*
static FORCEINLINE TYPE __select(__vec16_i1 mask, TYPE a, TYPE b) { \
    TYPE ret;                                                       \
    for (int i = 0; i < 16; ++i)                                    \
        ret.v[i] = (mask.v & (1<<i)) ? a.v[i] : b.v[i];             \
    return ret;                                                     \
}                                                                   \
static FORCEINLINE TYPE __select(bool cond, TYPE a, TYPE b) {       \
    return cond ? a : b;                                            \
}
*/

#define SHIFT_UNIFORM(TYPE, CAST, NAME, OP)
/*
static FORCEINLINE TYPE NAME(TYPE a, int32_t b) {                   \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v[i] = (CAST)(a.v[i]) OP b;                              \
   return ret;                                                      \
}
*/


#define SMEAR(VTYPE, NAME, STYPE)
/*
static FORCEINLINE VTYPE __smear_##NAME(STYPE v) {        \
    VTYPE ret;                                  \
    for (int i = 0; i < 16; ++i)                \
        ret.v[i] = v;                           \
    return ret;                                 \
}                                               \
*/

#define BROADCAST(VTYPE, NAME, STYPE)
/*
static FORCEINLINE VTYPE __broadcast_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i)                      \
        ret.v[i] = v.v[index & 0xf];                  \
    return ret;                                       \
}                                                     \
*/

#define ROTATE(VTYPE, NAME, STYPE)
/*
static FORCEINLINE VTYPE __rotate_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i)                      \
        ret.v[i] = v.v[(i+index) & 0xf];              \
    return ret;                                       \
}                                                     \
*/

#define SHUFFLES(VTYPE, NAME, STYPE)
/*
static FORCEINLINE VTYPE __shuffle_##NAME(VTYPE v, __vec16_i32 index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i)                      \
        ret.v[i] = v.v[__extract_element(index, i) & 0xf];      \
    return ret;                                       \
}                                                     \
static FORCEINLINE VTYPE __shuffle2_##NAME(VTYPE v0, VTYPE v1, __vec16_i32 index) {     \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i) {                    \
        int ii = __extract_element(index, i) & 0x1f;    \
        ret.v[i] = (ii < 16) ? v0.v[ii] : v1.v[ii-16];  \
    }                                                 \
    return ret;                                       \
}
*/

///////////////////////////////////////////////////////////////////////////

INSERT_EXTRACT(__vec1_i8, int8_t)
INSERT_EXTRACT(__vec1_i16, int16_t)
INSERT_EXTRACT(__vec1_i32, int32_t)
INSERT_EXTRACT(__vec1_i64, int64_t)
INSERT_EXTRACT(__vec1_f, float)
INSERT_EXTRACT(__vec1_d, double)

///////////////////////////////////////////////////////////////////////////
// mask ops

static FORCEINLINE __vec16_i1 __movmsk(__vec16_i1 mask) {
    return _mm512_kmov(mask);
}

static FORCEINLINE __vec16_i1 __equal(__vec16_i1 a, __vec16_i1 b) {
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
    return ((a.m & mask.m) | (b.m & ~mask.m));
    //return __or(__and(a, mask), __andnr(b, mask));
}

static FORCEINLINE __vec16_i1 __select(bool cond, __vec16_i1 a, __vec16_i1 b) {
    return cond ? a : b;
}

/*
static FORCEINLINE bool __extract_element(__vec16_i1 vec, int index) {
    return (vec.v & (1 << index)) ? true : false;
}

static FORCEINLINE void __insert_element(__vec16_i1 *vec, int index, 
                                         bool val) {
    if (val == false)
        vec->v &= ~(1 << index);
    else
        vec->v |= (1 << index);
}
*/

template <int ALIGN> static FORCEINLINE __vec16_i1 __load(__vec16_i1 *p) {
    uint16_t *ptr = (uint16_t *)p;
    __vec16_i1 r;
    r.m = *ptr;
    return r;
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_i1 *p, __vec16_i1 v) {
    uint16_t *ptr = (uint16_t *)p;
    *ptr = v.m;
}

static FORCEINLINE __vec16_i1 __smear_i1(__vec16_i1, int i) {
    return i?0xFFFF:0x0;
}

///////////////////////////////////////////////////////////////////////////
// int8

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

CMP_OP(__vec16_i8, int8_t,  __equal, ==)
CMP_OP(__vec16_i8, int8_t,  __not_equal, !=)
CMP_OP(__vec16_i8, uint8_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i8, int8_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i8, uint8_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i8, int8_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i8, uint8_t, __unsigned_less_than, <)
CMP_OP(__vec16_i8, int8_t,  __signed_less_than, <)
CMP_OP(__vec16_i8, uint8_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i8, int8_t,  __signed_greater_than, >)

SELECT(__vec16_i8)
INSERT_EXTRACT(__vec16_i8, int8_t)
SMEAR(__vec16_i8, i8, int8_t)
BROADCAST(__vec16_i8, i8, int8_t)
ROTATE(__vec16_i8, i8, int8_t)
SHUFFLES(__vec16_i8, i8, int8_t)
LOAD_STORE(__vec16_i8, int8_t)

///////////////////////////////////////////////////////////////////////////
// int16

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

CMP_OP(__vec16_i16, int16_t,  __equal, ==)
CMP_OP(__vec16_i16, int16_t,  __not_equal, !=)
CMP_OP(__vec16_i16, uint16_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i16, int16_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i16, uint16_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i16, int16_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i16, uint16_t, __unsigned_less_than, <)
CMP_OP(__vec16_i16, int16_t,  __signed_less_than, <)
CMP_OP(__vec16_i16, uint16_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i16, int16_t,  __signed_greater_than, >)

SELECT(__vec16_i16)
INSERT_EXTRACT(__vec16_i16, int16_t)
SMEAR(__vec16_i16, i16, int16_t)
BROADCAST(__vec16_i16, i16, int16_t)
ROTATE(__vec16_i16, i16, int16_t)
SHUFFLES(__vec16_i16, i16, int16_t)
LOAD_STORE(__vec16_i16, int16_t)

///////////////////////////////////////////////////////////////////////////
// int32

static FORCEINLINE __vec16_i32 __add(__vec16_i32 a, __vec16_i32 b) {
  return _mm512_add_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __sub(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_sub_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __mul(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_mullo_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __udiv(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_div_epu32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __sdiv(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_div_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __urem(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_rem_epu32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __srem(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_rem_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __or(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_or_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __and(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_and_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __xor(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_xor_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __shl(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_sllv_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __lshr(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_srlv_epi32((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __ashr(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_srav_epi32((__m512i)a, (__m512i)b); 
}

static FORCEINLINE __vec16_i32 __shl(__vec16_i32 a, int32_t n) {
    return _mm512_slli_epi32((__m512i)a, n);
}

static FORCEINLINE __vec16_i32 __lshr(__vec16_i32 a, int32_t n) {
    return _mm512_srli_epi32((__m512i)a, n); 
}

static FORCEINLINE __vec16_i32 __ashr(__vec16_i32 a, int32_t n) {
    return _mm512_srai_epi32((__m512i)a, n); 
}

static FORCEINLINE __vec16_i1 __equal(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmpeq_epi32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __not_equal(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmpneq_epi32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __unsigned_less_equal(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmple_epu32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __signed_less_equal(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmple_epi32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __unsigned_greater_equal(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmpge_epu32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __signed_greater_equal(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmpge_epi32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __unsigned_less_than(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmplt_epu32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __signed_less_than(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmplt_epi32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __unsigned_greater_than(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmpgt_epu32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i1 __signed_greater_than(__vec16_i32 a, __vec16_i32 b) {
    return _mm512_cmpgt_epi32_mask((__m512i)a, (__m512i)b);
}

static FORCEINLINE __vec16_i32 __select(__vec16_i1 mask,
                                        __vec16_i32 a, __vec16_i32 b) {
    return _mm512_mask_mov_epi32(b.v, mask.m, a.v);
} 

static FORCEINLINE __vec16_i32 __select(bool cond, __vec16_i32 a, __vec16_i32 b) {
    return cond ? a : b;
}

static FORCEINLINE int32_t __extract_element(__vec16_i32 v, int index) { return ((int32_t *)&v)[index]; }
static FORCEINLINE void __insert_element(__vec16_i32 *v, int index, int32_t val) { ((int32_t *)v)[index] = val; }

static FORCEINLINE __vec16_i32 __smear_i32(__vec16_i32, int32_t i) {
    return _mm512_set_1to16_epi32(i);
}

static FORCEINLINE __vec16_i32 __broadcast_i32(__vec16_i32 v, int index) {
    int32_t val = __extract_element(v, index & 0xf);
    return _mm512_set_1to16_epi32(val);
}

static FORCEINLINE __vec16_i32 __rotate_i32(__vec16_i32 v, int index) {
    __vec16_i32 shuffle((0+index)%8, (1+index)%8, (2+index)%8, (3+index)%8, 
                        (4+index)%8, (5+index)%8, (6+index)%8, (7+index)%8, 
                        (8+index)%8, (9+index)%8, (10+index)%8, (11+index)%8, 
                        (12+index)%8, (13+index)%8, (14+index), (15+index)%8);
    return _mm512_mask_permutevar_epi32(v, 0xffff, shuffle, v);
}

static FORCEINLINE __vec16_i32 __shuffle_i32(__vec16_i32 v, __vec16_i32 index) {
    return _mm512_mask_permutevar_epi32(v, 0xffff, index, v);
}

/*
static FORCEINLINE __vec16_i32 __shuffle2_i32(__vec16_i32 v0, __vec16_i32 v1, __vec16_i32 index) {
    __vec16_i32 ret; for (int i = 0; i < 16; ++i) { int ii = __extract_element(index, i) & 0x1f; ret.v[i] = (ii < 16) ? v0.v[ii] : v1.v[ii-16]; } return ret;
}
*/

template <int ALIGN> static FORCEINLINE __vec16_i32 __load(__vec16_i32 *p) {
    __vec16_i32 v;
    v = _mm512_extloadunpackhi_epi32(v, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v = _mm512_extloadunpacklo_epi32(v, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    return v;
}

template <> static FORCEINLINE __vec16_i32 __load<64>(__vec16_i32 *p) {
    return _mm512_load_epi32(p);
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_i32 *p, __vec16_i32 v) {
    _mm512_extpackstorehi_epi32(p, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_epi32(p, v, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
}

template <> static FORCEINLINE void __store<64>(__vec16_i32 *p, __vec16_i32 v) {
    _mm512_store_epi32(p, v);
}

///////////////////////////////////////////////////////////////////////////
// int64

BINARY_OP(__vec16_i64, __add, +)
BINARY_OP(__vec16_i64, __sub, -)
BINARY_OP(__vec16_i64, __mul, *)

BINARY_OP(__vec16_i64, __or, |)
BINARY_OP(__vec16_i64, __and, &)
BINARY_OP(__vec16_i64, __xor, ^)
BINARY_OP(__vec16_i64, __shl, <<)

BINARY_OP_CAST(__vec16_i64, uint64_t, __udiv, /)
BINARY_OP_CAST(__vec16_i64, int64_t,  __sdiv, /)

BINARY_OP_CAST(__vec16_i64, uint64_t, __urem, %)
BINARY_OP_CAST(__vec16_i64, int64_t,  __srem, %)
BINARY_OP_CAST(__vec16_i64, uint64_t, __lshr, >>)
BINARY_OP_CAST(__vec16_i64, int64_t,  __ashr, >>)

SHIFT_UNIFORM(__vec16_i64, uint64_t, __lshr, >>)
SHIFT_UNIFORM(__vec16_i64, int64_t, __ashr, >>)
SHIFT_UNIFORM(__vec16_i64, int64_t, __shl, <<)

CMP_OP(__vec16_i64, int64_t,  __equal, ==)
CMP_OP(__vec16_i64, int64_t,  __not_equal, !=)
CMP_OP(__vec16_i64, uint64_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i64, int64_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i64, uint64_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i64, int64_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i64, uint64_t, __unsigned_less_than, <)
CMP_OP(__vec16_i64, int64_t,  __signed_less_than, <)
CMP_OP(__vec16_i64, uint64_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i64, int64_t,  __signed_greater_than, >)

SELECT(__vec16_i64)
INSERT_EXTRACT(__vec16_i64, int64_t)
SMEAR(__vec16_i64, i64, int64_t)
BROADCAST(__vec16_i64, i64, int64_t)
ROTATE(__vec16_i64, i64, int64_t)
SHUFFLES(__vec16_i64, i64, int64_t)
LOAD_STORE(__vec16_i64, int64_t)


template <int ALIGN> static FORCEINLINE __vec16_i64 __load(__vec16_i64 *p) {
    __m512i v1;
    __m512i v2;
    v1 = _mm512_extloadunpackhi_epi32(v1, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v1 = _mm512_extloadunpacklo_epi32(v1, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v2 = _mm512_extloadunpackhi_epi32(v2, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v2 = _mm512_extloadunpacklo_epi32(v2, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);

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

template <> static FORCEINLINE __vec16_i64 __load<64>(__vec16_i64 *p) {
    __m512i v1 = _mm512_load_epi32(p);
    __m512i v2 = _mm512_load_epi32(((uint8_t*)p)+64);
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
    _mm512_extpackstorehi_epi32(p, v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_epi32(p, v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_epi32(((uint8_t*)p)+64, v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_epi32(((uint8_t*)p)+64, v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
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
    _mm512_store_epi64(p, v1);
    _mm512_store_epi64(((uint8_t*)p)+64, v2);
}



///////////////////////////////////////////////////////////////////////////
// float

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

static FORCEINLINE __vec16_i1 __equal(__vec16_f a, __vec16_f b) {
    return _mm512_cmpeq_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __not_equal(__vec16_f a, __vec16_f b) {
    return _mm512_cmpneq_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __less_than(__vec16_f a, __vec16_f b) {
    return _mm512_cmplt_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __less_equal(__vec16_f a, __vec16_f b) {
    return _mm512_cmple_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __greater_than(__vec16_f a, __vec16_f b) {
    return _mm512_cmpnle_ps_mask(a, b);
}

static FORCEINLINE __vec16_i1 __greater_equal(__vec16_f a, __vec16_f b) {
    return _mm512_cmpnlt_ps_mask(a, b);
}

/*
static FORCEINLINE __vec16_i1 __ordered(__vec16_f a, __vec16_f b) {
    __vec16_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 16; ++i)
        ret.v |= ((a.v[i] == a.v[i]) && (b.v[i] == b.v[i])) ? (1 << i) : 0;
    return ret;
}
*/


static FORCEINLINE __vec16_f __select(__vec16_i1 mask, __vec16_f a, __vec16_f b) {
    return _mm512_mask_mov_ps(b, mask, a);
}

static FORCEINLINE __vec16_f __select(bool cond, __vec16_f a, __vec16_f b) {
    return cond ? a : b;
}

static FORCEINLINE float __extract_element(__vec16_f v, int index) {
    return ((float *)&v)[index];
}

static FORCEINLINE void  __insert_element(__vec16_f *v, int index, float val) {
    ((float *)v)[index] = val;
}

static FORCEINLINE __vec16_f __smear_float(__vec16_f, float f) {
    return _mm512_set_1to16_ps(f);
}

static FORCEINLINE __vec16_f __broadcast_float(__vec16_f v, int index) {
    int32_t val = __extract_element(v, index & 0xf);
    return _mm512_set_1to16_ps(val);
}

/*
static FORCEINLINE __vec16_f __rotate_float(__vec16_f v, int index) {
    __vec16_f ret; for (int i = 0; i < 16; ++i) ret.v[i] = v.v[(i+index) & 0xf]; return ret;
}
*/

static FORCEINLINE __vec16_f __shuffle_float(__vec16_f v, __vec16_i32 index) {
    return _mm512_castsi512_ps(_mm512_mask_permutevar_epi32(_mm512_castps_si512(v), 0xffff, index, _mm512_castps_si512(v)));
}

/*
static FORCEINLINE __vec16_f __shuffle2_float(__vec16_f v0, __vec16_f v1, __vec16_i32 index) {
    __vec16_f ret; for (int i = 0; i < 16; ++i) { int ii = __extract_element(index, i) & 0x1f; ret.v[i] = (ii < 16) ? v0.v[ii] : v1.v[ii-16]; } return ret;
}
*/

template <int ALIGN> static FORCEINLINE __vec16_f __load(__vec16_f *p) {
    __vec16_f v;
    v = _mm512_extloadunpackhi_ps(v, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    v = _mm512_extloadunpacklo_ps(v, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    return v;
}

template <> static FORCEINLINE __vec16_f __load<64>(__vec16_f *p) {
    return _mm512_load_ps(p);
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_f *p, __vec16_f v) {
    _mm512_extpackstorehi_ps(p, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_ps(p, v, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
}

template <> static FORCEINLINE void __store<64>(__vec16_f *p, __vec16_f v) {
    _mm512_store_ps(p, v);
}


///////////////////////////////////////////////////////////////////////////
// double

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

static FORCEINLINE __vec16_i1 __equal(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.m8.m1 = _mm512_cmpeq_pd_mask(a.v1, b.v1);
    ret.m8.m2 = _mm512_cmpeq_pd_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec16_i1 __not_equal(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.m8.m1 = _mm512_cmpneq_pd_mask(a.v1, b.v1);
    ret.m8.m2 = _mm512_cmpneq_pd_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec16_i1 __less_than(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.m8.m1 = _mm512_cmplt_pd_mask(a.v1, b.v1);
    ret.m8.m2 = _mm512_cmplt_pd_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec16_i1 __less_equal(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.m8.m1 = _mm512_cmple_pd_mask(a.v1, b.v1);
    ret.m8.m2 = _mm512_cmple_pd_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec16_i1 __greater_than(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.m8.m1 = _mm512_cmpnle_pd_mask(a.v1, b.v1);
    ret.m8.m2 = _mm512_cmpnle_pd_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec16_i1 __greater_equal(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.m8.m1 = _mm512_cmpnlt_pd_mask(a.v1, b.v1);
    ret.m8.m2 = _mm512_cmpnlt_pd_mask(a.v2, b.v2);
    return ret;
}


/*
static FORCEINLINE __vec16_i1 __ordered(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 16; ++i)
        ret.v |= ((a.v[i] == a.v[i]) && (b.v[i] == b.v[i])) ? (1 << i) : 0;
    return ret;
}
*/

static FORCEINLINE __vec16_d __select(__vec16_i1 mask, __vec16_d a, __vec16_d b) {
    __vec16_d ret;
    ret.v1 = _mm512_mask_mov_pd(b.v1, mask.m8.m1, a.v1);
    ret.v2 = _mm512_mask_mov_pd(b.v2, mask.m8.m2, a.v2);
    return ret;
}


static FORCEINLINE __vec16_d __select(bool cond, __vec16_d a, __vec16_d b) {
    return cond ? a : b;
}

static FORCEINLINE double __extract_element(__vec16_d v, int index) {
    return ((double *)&v)[index];
}

static FORCEINLINE void  __insert_element(__vec16_d *v, int index, double val) {
    ((double *)v)[index] = val;
}

static FORCEINLINE __vec16_d __smear_double(__vec16_d, double d) {
    __vec16_d ret;
    ret.v1 = _mm512_extload_pd(&d, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    ret.v2 = ret.v1;
    return ret;
}

static FORCEINLINE __vec16_d __broadcast_double(__vec16_d v, int index) {
    __vec16_d ret;
    int32_t val = __extract_element(v, index & 0xf);
    ret.v1 = _mm512_extload_pd(&val, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    ret.v2 = ret.v1;
    return ret;
}

/*
static FORCEINLINE __vec16_d __rotate_double(__vec16_d v, int index) {
    __vec16_d ret; for (int i = 0; i < 16; ++i) ret.v[i] = v.v[(i+index) & 0xf]; return ret;
}

static FORCEINLINE __vec16_d __shuffle_double(__vec16_d v, __vec16_i32 index) {
    return _mm512_castsi512_ps(
        _mm512_mask_permutevar_epi32(_mm512_castps_si512(v), 0xffff, index, _mm512_castps_si512(v)));
}

static FORCEINLINE __vec16_f __shuffle2_float(__vec16_d v0, __vec16_d v1, __vec16_i32 index) {
    __vec16_f ret; for (int i = 0; i < 16; ++i) { int ii = __extract_element(index, i) & 0x1f; ret.v[i] = (ii < 16) ? v0.v[ii] : v1.v[ii-16]; } return ret;
}
*/

template <int ALIGN> static FORCEINLINE __vec16_d __load(__vec16_d *p) {
    __vec16_d ret;
    ret.v1 = _mm512_extloadunpackhi_pd(ret.v1, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
    ret.v1 = _mm512_extloadunpacklo_pd(ret.v1, p, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
    ret.v2 = _mm512_extloadunpackhi_pd(ret.v2, ((uint8_t*)p)+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
    ret.v2 = _mm512_extloadunpacklo_pd(ret.v2, ((uint8_t*)p)+64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
    return ret;
}

template <> static FORCEINLINE __vec16_d __load<64>(__vec16_d *p) {
    __vec16_d ret;
    ret.v1 = _mm512_load_pd(p);
    ret.v2 = _mm512_load_pd(((uint8_t*)p)+64);
    return ret;
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_d *p, __vec16_d v) {
    _mm512_extpackstorehi_pd(p, v.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_pd(p, v.v1, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_pd(((uint8_t*)p)+64, v.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_pd(((uint8_t*)p)+64, v.v2, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
}

template <> static FORCEINLINE void __store<64>(__vec16_d *p, __vec16_d v) {
    _mm512_store_pd(p, v.v1);
    _mm512_store_pd(((uint8_t*)p)+64, v.v2);
}

///////////////////////////////////////////////////////////////////////////
// casts


#define CAST(TO, STO, FROM, SFROM, FUNC)
/*
static FORCEINLINE TO FUNC(TO, FROM val) {      \
    TO ret;                                     \
    for (int i = 0; i < 16; ++i)                \
        ret.v[i] = (STO)((SFROM)(val.v[i]));    \
    return ret;                                 \
}
*/
// sign extension conversions
CAST(__vec16_i64, int64_t, __vec16_i32, int32_t, __cast_sext)
CAST(__vec16_i64, int64_t, __vec16_i16, int16_t, __cast_sext)
CAST(__vec16_i64, int64_t, __vec16_i8,  int8_t,  __cast_sext)
CAST(__vec16_i32, int32_t, __vec16_i16, int16_t, __cast_sext)
CAST(__vec16_i32, int32_t, __vec16_i8,  int8_t,  __cast_sext)
CAST(__vec16_i16, int16_t, __vec16_i8,  int8_t,  __cast_sext)

#define CAST_SEXT_I1(TYPE)
/*
static FORCEINLINE TYPE __cast_sext(TYPE, __vec16_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 16; ++i) {                    \
        ret.v[i] = 0;                                 \
        if (v.v & (1 << i))                           \
            ret.v[i] = ~ret.v[i];                     \
    }                                                 \
    return ret;                                       \
}
*/
CAST_SEXT_I1(__vec16_i8)
CAST_SEXT_I1(__vec16_i16)
CAST_SEXT_I1(__vec16_i32)
CAST_SEXT_I1(__vec16_i64)

// zero extension
CAST(__vec16_i64, uint64_t, __vec16_i32, uint32_t, __cast_zext)
CAST(__vec16_i64, uint64_t, __vec16_i16, uint16_t, __cast_zext)
CAST(__vec16_i64, uint64_t, __vec16_i8,  uint8_t,  __cast_zext)
CAST(__vec16_i32, uint32_t, __vec16_i16, uint16_t, __cast_zext)
CAST(__vec16_i32, uint32_t, __vec16_i8,  uint8_t,  __cast_zext)
CAST(__vec16_i16, uint16_t, __vec16_i8,  uint8_t,  __cast_zext)

#define CAST_ZEXT_I1(TYPE)
/*
static FORCEINLINE TYPE __cast_zext(TYPE, __vec16_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 16; ++i)                      \
        ret.v[i] = (v.v & (1 << i)) ? 1 : 0;          \
    return ret;                                       \
}
*/
CAST_ZEXT_I1(__vec16_i8)
CAST_ZEXT_I1(__vec16_i16)
CAST_ZEXT_I1(__vec16_i32)
CAST_ZEXT_I1(__vec16_i64)

// truncations
CAST(__vec16_i32, int32_t, __vec16_i64, int64_t, __cast_trunc)
CAST(__vec16_i16, int16_t, __vec16_i64, int64_t, __cast_trunc)
CAST(__vec16_i8,  int8_t,  __vec16_i64, int64_t, __cast_trunc)
CAST(__vec16_i16, int16_t, __vec16_i32, int32_t, __cast_trunc)
CAST(__vec16_i8,  int8_t,  __vec16_i32, int32_t, __cast_trunc)
CAST(__vec16_i8,  int8_t,  __vec16_i16, int16_t, __cast_trunc)

// signed int to float/double
CAST(__vec16_f, float, __vec16_i64,  int64_t, __cast_sitofp)
CAST(__vec16_d, double, __vec16_i64, int64_t, __cast_sitofp)


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


/*
static FORCEINLINE __vec16_f __cast_sitofp(__vec16_f, __vec16_i64 val) {
    __vec16_f ret; for (int i = 0; i < 16; ++i) ret.v[i] = (float)((int64_t)(val.v[i])); return ret;
}


static FORCEINLINE __vec16_d __cast_sitofp(__vec16_d, __vec16_i64 val) {
    __vec16_d ret; for (int i = 0; i < 16; ++i) ret.v[i] = (double)((int64_t)(val.v[i])); return ret;
}
*/

// unsigned int to float/double
CAST(__vec16_f, float, __vec16_i8,   uint8_t,  __cast_uitofp)
CAST(__vec16_f, float, __vec16_i16,  uint16_t, __cast_uitofp)
CAST(__vec16_f, float, __vec16_i32,  uint32_t, __cast_uitofp)
CAST(__vec16_f, float, __vec16_i64,  uint64_t, __cast_uitofp)
CAST(__vec16_d, double, __vec16_i8,  uint8_t,  __cast_uitofp)
CAST(__vec16_d, double, __vec16_i16, uint16_t, __cast_uitofp)
CAST(__vec16_d, double, __vec16_i32, uint32_t, __cast_uitofp)
CAST(__vec16_d, double, __vec16_i64, uint64_t, __cast_uitofp)
/*
static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i1 v) {
    __vec16_f ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = (v.v & (1 << i)) ? 1. : 0.;
    return ret;
}
*/
// float/double to signed int

static FORCEINLINE __vec16_i32 __cast_fptosi(__vec16_i32, __vec16_f val) {
  return _mm512_cvtfxpnt_round_adjustps_epi32(val, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

CAST(__vec16_i8,  int8_t,  __vec16_f, float, __cast_fptosi)
CAST(__vec16_i16, int16_t, __vec16_f, float, __cast_fptosi)
CAST(__vec16_i32, int32_t, __vec16_f, float, __cast_fptosi)
CAST(__vec16_i64, int64_t, __vec16_f, float, __cast_fptosi)
CAST(__vec16_i8,  int8_t,  __vec16_d, double, __cast_fptosi)
CAST(__vec16_i16, int16_t, __vec16_d, double, __cast_fptosi)
CAST(__vec16_i32, int32_t, __vec16_d, double, __cast_fptosi)
CAST(__vec16_i64, int64_t, __vec16_d, double, __cast_fptosi)

// float/double to unsigned int
CAST(__vec16_i8,  uint8_t,  __vec16_f, float, __cast_fptoui)
CAST(__vec16_i16, uint16_t, __vec16_f, float, __cast_fptoui)
CAST(__vec16_i32, uint32_t, __vec16_f, float, __cast_fptoui)
CAST(__vec16_i64, uint64_t, __vec16_f, float, __cast_fptoui)
CAST(__vec16_i8,  uint8_t,  __vec16_d, double, __cast_fptoui)
CAST(__vec16_i16, uint16_t, __vec16_d, double, __cast_fptoui)
CAST(__vec16_i32, uint32_t, __vec16_d, double, __cast_fptoui)
CAST(__vec16_i64, uint64_t, __vec16_d, double, __cast_fptoui)

// float/double conversions
CAST(__vec16_f, float,  __vec16_d, double, __cast_fptrunc)
CAST(__vec16_d, double, __vec16_f, float,  __cast_fpext)

/*
typedef union {
    int32_t i32;
    float f;
    int64_t i64;
    double d;
} BitcastUnion;
*/
#define CAST_BITS(TO, TO_TYPE, TO_ELT, FROM, FROM_ELT)
/*
static FORCEINLINE TO __cast_bits(TO, FROM val) {      \
  BitcastUnion u;				       \
  u.FROM_ELT = val;				       \
  return u.TO_ELT;				       \
}
*/

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


#define CAST_BITS_SCALAR(TO, FROM)
/*
static FORCEINLINE TO __cast_bits(TO, FROM v) {     \
    union {                                         \
    TO to;                                          \
    FROM from;                                      \
    } u;                                            \
    u.from = v;                                     \
    return u.to;                                    \
}
*/
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

/*
static FORCEINLINE void __fastmath() {
}
*/

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

UNARY_OP(__vec16_d, __round_varying_double, round)
UNARY_OP(__vec16_d, __floor_varying_double, floor)
UNARY_OP(__vec16_d, __ceil_varying_double, ceil)

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

BINARY_OP_FUNC(__vec16_d, __max_varying_double, __max_uniform_double)
BINARY_OP_FUNC(__vec16_d, __min_varying_double, __min_uniform_double)

BINARY_OP_FUNC(__vec16_i32, __max_varying_int32, __max_uniform_int32)
BINARY_OP_FUNC(__vec16_i32, __min_varying_int32, __min_uniform_int32)
BINARY_OP_FUNC(__vec16_i32, __max_varying_uint32, __max_uniform_uint32)
BINARY_OP_FUNC(__vec16_i32, __min_varying_uint32, __min_uniform_uint32)

BINARY_OP_FUNC(__vec16_i64, __max_varying_int64, __max_uniform_int64)
BINARY_OP_FUNC(__vec16_i64, __min_varying_int64, __min_uniform_int64)
BINARY_OP_FUNC(__vec16_i64, __max_varying_uint64, __max_uniform_uint64)
BINARY_OP_FUNC(__vec16_i64, __min_varying_uint64, __min_uniform_uint64)

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
    return _mm512_recip_ps(v);
    //return _mm512_rcp23_ps(v); // 23-bits of accuracy
}

static FORCEINLINE __vec16_f __rsqrt_varying_float(__vec16_f v) {
  return _mm512_rsqrt23_ps(v); // to 0.775ULP accuracy
}

static FORCEINLINE __vec16_f __exp_varying_float(__vec16_f v) {
  return _mm512_exp_ps(v);
}

static FORCEINLINE __vec16_f __log_varying_float(__vec16_f v) {
  return _mm512_log_ps(v);
}

static FORCEINLINE __vec16_f __pow_varying_float(__vec16_f a, __vec16_f b) {
  return _mm512_pow_ps(a, b);
}

UNARY_OP(__vec16_d, __sqrt_varying_double, __sqrt_uniform_double)

///////////////////////////////////////////////////////////////////////////
// bit ops

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

/*
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
*/
///////////////////////////////////////////////////////////////////////////
// reductions

REDUCE_ADD(float, __vec16_f, __reduce_add_float)
REDUCE_MINMAX(float, __vec16_f, __reduce_min_float, <)
REDUCE_MINMAX(float, __vec16_f, __reduce_max_float, >)

REDUCE_ADD(double, __vec16_d, __reduce_add_double)
REDUCE_MINMAX(double, __vec16_d, __reduce_min_double, <)
REDUCE_MINMAX(double, __vec16_d, __reduce_max_double, >)

REDUCE_ADD(uint32_t, __vec16_i32, __reduce_add_int32)
REDUCE_MINMAX(int32_t, __vec16_i32, __reduce_min_int32, <)
REDUCE_MINMAX(int32_t, __vec16_i32, __reduce_max_int32, >)

REDUCE_ADD(uint32_t, __vec16_i32, __reduce_add_uint32)
REDUCE_MINMAX(uint32_t, __vec16_i32, __reduce_min_uint32, <)
REDUCE_MINMAX(uint32_t, __vec16_i32, __reduce_max_uint32, >)

REDUCE_ADD(uint64_t, __vec16_i64, __reduce_add_int64)
REDUCE_MINMAX(int64_t, __vec16_i64, __reduce_min_int64, <)
REDUCE_MINMAX(int64_t, __vec16_i64, __reduce_max_int64, >)

REDUCE_ADD(uint64_t, __vec16_i64, __reduce_add_uint64)
REDUCE_MINMAX(uint64_t, __vec16_i64, __reduce_min_uint64, <)
REDUCE_MINMAX(uint64_t, __vec16_i64, __reduce_max_uint64, >)

///////////////////////////////////////////////////////////////////////////
// masked load/store
/*
static FORCEINLINE __vec16_i8 __masked_load_i8(void *p,
                                              __vec16_i1 mask) {
    __vec16_i8 ret;
    int8_t *ptr = (int8_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret.v[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec16_i16 __masked_load_i16(void *p,
                                                __vec16_i1 mask) {
    __vec16_i16 ret;
    int16_t *ptr = (int16_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret.v[i] = ptr[i];
    return ret;
}
*/

template <int ALIGN> static FORCEINLINE __vec16_i32 __masked_load_i32(void *p, __vec16_i1 mask) {
    __vec16_i32 ret;
    ret = _mm512_mask_extloadunpackhi_epi32(ret, mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    ret = _mm512_mask_extloadunpacklo_epi32(ret, mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    return ret;
}

template <> static FORCEINLINE __vec16_i32 __masked_load_i32<64>(void *p, __vec16_i1 mask) {
    return _mm512_mask_load_epi32(_mm512_undefined_epi32(), mask, p);
}

static FORCEINLINE __vec16_i32 __masked_load_i32(void *p, __vec16_i1 mask) {
    __vec16_i32 ret;
    ret = _mm512_mask_extloadunpackhi_epi32(ret, mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    ret = _mm512_mask_extloadunpacklo_epi32(ret, mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    return ret;
}

/*
static FORCEINLINE __vec16_i64 __masked_load_i64(void *p,
                                                __vec16_i1 mask) {
    union {
        __vec16_i64 v64;
        __vec16_i32 v32[2];
    } ret;

    ret.v32[0] = _mm512_undefined_epi32();
    ret.v32[1] = _mm512_undefined_epi32();
    ret.v32[0] = _mm512_mask_loadq(ret, mask, p, _MM_FULLUPC64_NONE, _MM_BROADCAST_8X8, _MM_HINT_NONE);
    ret.v32[1] = _mm512_mask_loadq(ret, mask, p, _MM_FULLUPC64_NONE, _MM_BROADCAST_8X8, _MM_HINT_NONE);

    return ret.v64;
}
*/

static FORCEINLINE __vec16_f __masked_load_float(void *p, __vec16_i1 mask) {
    __vec16_f ret;
    return _mm512_mask_load_ps(ret, mask, p);
}

static FORCEINLINE __vec16_d __masked_load_double(void *p, __vec16_i1 mask) {
    __vec16_d ret;
    ret.v1 = _mm512_mask_load_pd(ret.v1, mask.m8.m1, p);
    ret.v2 = _mm512_mask_load_pd(ret.v2, mask.m8.m2, p);
    return ret;
}

/*
static FORCEINLINE void __masked_store_i8(void *p, __vec16_i8 val,
                                         __vec16_i1 mask) {
    int8_t *ptr = (int8_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
}

static FORCEINLINE void __masked_store_i16(void *p, __vec16_i16 val,
                                          __vec16_i1 mask) {
    int16_t *ptr = (int16_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
}
*/

static FORCEINLINE void __masked_store_i32(void *p, __vec16_i32 val,
                                          __vec16_i1 mask) {
    _mm512_mask_store_epi32(p, mask, val);
}

/*
static FORCEINLINE void __masked_store_i64(void *p, __vec16_i64 val,
                                          __vec16_i1 mask) {
    // TODO: this needs to change
    _mm512_mask_store_epi64(p, mask, val.v1);
    _mm512_mask_store_epi64((uint8_t*)p+64, mask, val.v2);
}
*/

static FORCEINLINE void __masked_store_float(void *p, __vec16_f val,
                                             __vec16_i1 mask) {
    _mm512_mask_store_ps(p, mask, val);
}

static FORCEINLINE void __masked_store_double(void *p, __vec16_d val,
                                              __vec16_i1 mask) {
    _mm512_mask_store_pd(p, mask.m8.m1, val.v1);
    _mm512_mask_store_pd(((uint8_t*)p)+64, mask.m8.m2, val.v2);
}

/*
static FORCEINLINE void __masked_store_blend_i8(void *p, __vec16_i8 val,
                                               __vec16_i1 mask) {
    __masked_store_i8(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i16(void *p, __vec16_i16 val,
                                                __vec16_i1 mask) {
    __masked_store_i16(p, val, mask);
}
*/

static FORCEINLINE void __masked_store_blend_i32(void *p, __vec16_i32 val,
                                                __vec16_i1 mask) {
    __masked_store_i32(p, val, mask);
}

/*
static FORCEINLINE void __masked_store_blend_i64(void *p, __vec16_i64 val,
                                                __vec16_i1 mask) {
    __masked_store_i64(p, val, mask);
}
*/

static FORCEINLINE void __masked_store_blend_float(void *p, __vec16_f val,
                                                  __vec16_i1 mask) {
        __masked_store_float(p, val, mask);
}

///////////////////////////////////////////////////////////////////////////
// gather/scatter

// offsets * offsetScale is in bytes (for all of these)

#define GATHER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)
/*
static FORCEINLINE VTYPE FUNC(unsigned char *b, OTYPE varyingOffset,    \
                              uint32_t scale, OTYPE constOffset, \
                              __vec16_i1 mask) {                        \
    VTYPE ret;                                                          \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 16; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * varyingOffset.v[i] +  \
                                   constOffset.v[i]);                   \
            ret.v[i] = *ptr;                                            \
        }                                                               \
    return ret;                                                         \
}
*/

static FORCEINLINE __vec16_i32
__gather_base_offsets32_i32(uint8_t *base, __vec16_i32 varyingOffset, 
			    uint32_t scale, __vec16_i32 constOffset, 
			    __vec16_i1 mask) { 
    __vec16_i32 vscale = _mm512_extload_epi32(&scale, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
    __vec16_i32 offsets = __add(__mul(vscale, varyingOffset), constOffset);
    __vec16_i32 tmp;

    // Loop is generated by intrinsic
    __vec16_i32 ret = _mm512_mask_i32extgather_epi32(tmp, mask, offsets, base, 
                                                     _MM_UPCONV_EPI32_NONE, 1,
                                                     _MM_HINT_NONE);
    return ret;
}

static FORCEINLINE __vec16_f
__gather_base_offsets32_float(uint8_t *base, __vec16_i32 varyingOffset, 
                              uint32_t scale, __vec16_i32 constOffset, 
                              __vec16_i1 mask) { 
    __vec16_i32 vscale = _mm512_extload_epi32(&scale, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
    __vec16_i32 offsets = __add(__mul(vscale, varyingOffset), constOffset);
    __vec16_f tmp;

    // Loop is generated by intrinsic
    __vec16_f ret = _mm512_mask_i32extgather_ps(tmp, mask, offsets, base, 
                                                _MM_UPCONV_PS_NONE, 1,
                                                _MM_HINT_NONE);
    return ret;
}

GATHER_BASE_OFFSETS(__vec16_i8, int8_t, __vec16_i32, __gather_base_offsets32_i8)
GATHER_BASE_OFFSETS(__vec16_i8, int8_t, __vec16_i64, __gather_base_offsets64_i8)
GATHER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i32, __gather_base_offsets32_i16)
GATHER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i64, __gather_base_offsets64_i16)
//GATHER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i32, __gather_base_offsets32_i32)
GATHER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i64, __gather_base_offsets64_i32)
GATHER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i32, __gather_base_offsets32_i64)
GATHER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i64, __gather_base_offsets64_i64)

#define GATHER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)
/*
static FORCEINLINE VTYPE FUNC(PTRTYPE ptrs, __vec16_i1 mask) {   \
    VTYPE ret;                                              \
    for (int i = 0; i < 16; ++i)                            \
        if ((mask.v & (1 << i)) != 0) {                     \
            STYPE *ptr = (STYPE *)ptrs.v[i];                \
            ret.v[i] = *ptr;                                \
        }                                                   \
    return ret;                                             \
}
*/

GATHER_GENERAL(__vec16_i8, int8_t, __vec16_i32, __gather32_i8)
GATHER_GENERAL(__vec16_i8, int8_t, __vec16_i64, __gather64_i8)
GATHER_GENERAL(__vec16_i16, int16_t, __vec16_i32, __gather32_i16)
GATHER_GENERAL(__vec16_i16, int16_t, __vec16_i64, __gather64_i16)
GATHER_GENERAL(__vec16_i32, int32_t, __vec16_i32, __gather32_i32)
GATHER_GENERAL(__vec16_i32, int32_t, __vec16_i64, __gather64_i32)
GATHER_GENERAL(__vec16_i64, int64_t, __vec16_i32, __gather32_i64)
GATHER_GENERAL(__vec16_i64, int64_t, __vec16_i64, __gather64_i64)

/*
static __forceinline __vec16_i32 __gather64_i32(__vec16_i64 ptrs, __vec16_i1 mask) {
    __vec16_i32 ret;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0) {
            int32_t *ptr = (int32_t *)ptrs.v[i];
            ret.v[i] = *ptr; 
        } 
    return ret;
}
*/
/*
static FORCEINLINE __vec16_i32 __gather64_i32(__vec16_i64 ptrs, __vec16_i1 mask) {
    // Loop is generated by intrinsic
    __vec16_i32 ret = _mm512_mask_i32extgather_epi32(tmp, mask, offsets, base, 
                                                     _MM_UPCONV_EPI32_NONE, 1,
                                                     _MM_HINT_NONE);
    return ret;
}
*/
// scatter

#define SCATTER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)
/*
static FORCEINLINE void FUNC(unsigned char *b, OTYPE varyingOffset,     \
                             uint32_t scale, OTYPE constOffset,         \
                             VTYPE val, __vec16_i1 mask) {              \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 16; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * varyingOffset.v[i] +  \
                                   constOffset.v[i]);                   \
            *ptr = val.v[i];                                            \
        }                                                               \
}
*/

SCATTER_BASE_OFFSETS(__vec16_i8, int8_t, __vec16_i32, __scatter_base_offsets32_i8)
SCATTER_BASE_OFFSETS(__vec16_i8, int8_t, __vec16_i64, __scatter_base_offsets64_i8)
SCATTER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i32, __scatter_base_offsets32_i16)
SCATTER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i64, __scatter_base_offsets64_i16)
//SCATTER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i32, __scatter_base_offsets32_i32)
SCATTER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i64, __scatter_base_offsets64_i32)
SCATTER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i32, __scatter_base_offsets32_i64)
SCATTER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i64, __scatter_base_offsets64_i64)

static FORCEINLINE void
__scatter_base_offsets32_i32(uint8_t *b, __vec16_i32 varyingOffset,
                             uint32_t scale, __vec16_i32 constOffset,
                             __vec16_i32 val, __vec16_i1 mask) {
    __vec16_i32 vscale = _mm512_extload_epi32(&scale, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
    __vec16_i32 offsets = __add(__mul(vscale, varyingOffset), constOffset);

    // Loop is generated by the intrinsic
    _mm512_mask_i32extscatter_epi32(b, mask, offsets, val, _MM_DOWNCONV_EPI32_NONE, 1, _MM_HINT_NONE);
}

#define SCATTER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)
/*
static FORCEINLINE void FUNC(PTRTYPE ptrs, VTYPE val, __vec16_i1 mask) {  \
    VTYPE ret;                                                       \
    for (int i = 0; i < 16; ++i)                                     \
        if ((mask.v & (1 << i)) != 0) {                              \
            STYPE *ptr = (STYPE *)ptrs.v[i];                         \
            *ptr = val.v[i];                                         \
        }                                                            \
}
*/
SCATTER_GENERAL(__vec16_i8, int8_t, __vec16_i32, __scatter32_i8)
SCATTER_GENERAL(__vec16_i8, int8_t, __vec16_i64, __scatter64_i8)
SCATTER_GENERAL(__vec16_i16, int16_t, __vec16_i32, __scatter32_i16)
SCATTER_GENERAL(__vec16_i16, int16_t, __vec16_i64, __scatter64_i16)
SCATTER_GENERAL(__vec16_i32, int32_t, __vec16_i32, __scatter32_i32)
SCATTER_GENERAL(__vec16_i32, int32_t, __vec16_i64, __scatter64_i32)
SCATTER_GENERAL(__vec16_i64, int64_t, __vec16_i32, __scatter32_i64)
SCATTER_GENERAL(__vec16_i64, int64_t, __vec16_i64, __scatter64_i64)

///////////////////////////////////////////////////////////////////////////
// packed load/store

static FORCEINLINE int32_t __packed_load_active(uint32_t *p, __vec16_i32 *val,
                                                __vec16_i1 mask) {
    __vec16_i32 v;
    v = _mm512_mask_extloadunpackhi_epi32(_mm512_undefined_epi32(), mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    v = _mm512_mask_extloadunpacklo_epi32(v, mask, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    __store<64>(val, v);
    return _mm_countbits_32(uint32_t(mask));
}

static FORCEINLINE int32_t __packed_store_active(uint32_t *p, __vec16_i32 val,
                                                 __vec16_i1 mask) {
    _mm512_mask_extpackstorehi_epi32(p, mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_mask_extpackstorelo_epi32(p, mask, val, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    return _mm_countbits_32(uint32_t(mask.m));
}

///////////////////////////////////////////////////////////////////////////
// aos/soa

/*
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
*/

///////////////////////////////////////////////////////////////////////////
// prefetch

static FORCEINLINE void __prefetch_read_uniform_1(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_2(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_3(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_nt(unsigned char *) {
}

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

#undef FORCEINLINE
#undef PRE_ALIGN
#undef POST_ALIGN

