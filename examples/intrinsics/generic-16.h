/*
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

#include <stdint.h>
#include <math.h>

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

typedef float __vec1_f;
typedef double __vec1_d;
typedef int8_t __vec1_i8;
typedef int16_t __vec1_i16;
typedef int32_t __vec1_i32;
typedef int64_t __vec1_i64;

struct __vec16_i1 {
    __vec16_i1() { }
    __vec16_i1(const uint16_t &vv) : v(vv) { }
    __vec16_i1(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
               uint32_t v4, uint32_t v5, uint32_t v6, uint32_t v7,
               uint32_t v8, uint32_t v9, uint32_t v10, uint32_t v11,
               uint32_t v12, uint32_t v13, uint32_t v14, uint32_t v15) {
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
             
    uint16_t v;
};


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

PRE_ALIGN(64) struct __vec16_f : public vec16<float> { 
    __vec16_f() { }
    __vec16_f(float v0, float v1, float v2, float v3, 
              float v4, float v5, float v6, float v7,
              float v8, float v9, float v10, float v11, 
              float v12, float v13, float v14, float v15) 
        : vec16<float>(v0, v1, v2, v3, v4, v5, v6, v7,
                       v8, v9, v10, v11, v12, v13, v14, v15) { }

} POST_ALIGN(64);

PRE_ALIGN(128) struct __vec16_d : public vec16<double> { 
    __vec16_d() { }
    __vec16_d(double v0, double v1, double v2, double v3, 
              double v4, double v5, double v6, double v7,
              double v8, double v9, double v10, double v11, 
              double v12, double v13, double v14, double v15) 
        : vec16<double>(v0, v1, v2, v3, v4, v5, v6, v7,
                        v8, v9, v10, v11, v12, v13, v14, v15) { }

} POST_ALIGN(128);

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

PRE_ALIGN(64) struct __vec16_i32  : public vec16<int32_t> { 
    __vec16_i32() { }
    __vec16_i32(int32_t v0, int32_t v1, int32_t v2, int32_t v3, 
                int32_t v4, int32_t v5, int32_t v6, int32_t v7,
                int32_t v8, int32_t v9, int32_t v10, int32_t v11, 
                int32_t v12, int32_t v13, int32_t v14, int32_t v15) 
        : vec16<int32_t>(v0, v1, v2, v3, v4, v5, v6, v7,
                         v8, v9, v10, v11, v12, v13, v14, v15) { }
} POST_ALIGN(64);

static inline int32_t __extract_element(__vec16_i32, int);

PRE_ALIGN(128) struct __vec16_i64  : public vec16<int64_t> { 
    __vec16_i64() { }
    __vec16_i64(int64_t v0, int64_t v1, int64_t v2, int64_t v3, 
                int64_t v4, int64_t v5, int64_t v6, int64_t v7,
                int64_t v8, int64_t v9, int64_t v10, int64_t v11, 
                int64_t v12, int64_t v13, int64_t v14, int64_t v15) 
        : vec16<int64_t>(v0, v1, v2, v3, v4, v5, v6, v7,
                         v8, v9, v10, v11, v12, v13, v14, v15) { }
} POST_ALIGN(128);

///////////////////////////////////////////////////////////////////////////
// macros...

#define UNARY_OP(TYPE, NAME, OP)            \
static FORCEINLINE TYPE NAME(TYPE v) {      \
    TYPE ret;                               \
    for (int i = 0; i < 16; ++i)            \
        ret.v[i] = OP(v.v[i]);              \
    return ret;                             \
}

#define BINARY_OP(TYPE, NAME, OP)                               \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                  \
    TYPE ret;                                                   \
   for (int i = 0; i < 16; ++i)                                 \
       ret.v[i] = a.v[i] OP b.v[i];                             \
   return ret;                                                   \
}

#define BINARY_OP_CAST(TYPE, CAST, NAME, OP)                        \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v[i] = (CAST)(a.v[i]) OP (CAST)(b.v[i]);                 \
   return ret;                                                      \
}

#define BINARY_OP_FUNC(TYPE, NAME, FUNC)                            \
static FORCEINLINE TYPE NAME(TYPE a, TYPE b) {                      \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v[i] = FUNC(a.v[i], b.v[i]);                             \
   return ret;                                                      \
}

#define CMP_OP(TYPE, SUFFIX, CAST, NAME, OP)                        \
static FORCEINLINE __vec16_i1 NAME##_##SUFFIX(TYPE a, TYPE b) {     \
   __vec16_i1 ret;                                                  \
   ret.v = 0;                                                       \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v |= ((CAST)(a.v[i]) OP (CAST)(b.v[i])) << i;            \
   return ret;                                                      \
}                                                                   \
static FORCEINLINE __vec16_i1 NAME##_##SUFFIX##_and_mask(TYPE a, TYPE b,       \
                                              __vec16_i1 mask) {    \
   __vec16_i1 ret;                                                  \
   ret.v = 0;                                                       \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v |= ((CAST)(a.v[i]) OP (CAST)(b.v[i])) << i;            \
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
    for (int i = 0; i < 16; ++i)                       \
        ret.v[i] = ptr[i];                             \
    return ret;                                        \
}                                                      \
template <int ALIGN>                                   \
static FORCEINLINE void __store(VTYPE *p, VTYPE v) {   \
    STYPE *ptr = (STYPE *)p;                           \
    for (int i = 0; i < 16; ++i)                       \
        ptr[i] = v.v[i];                               \
}

#define REDUCE_ADD(TYPE, VTYPE, NAME)           \
static FORCEINLINE TYPE NAME(VTYPE v) {         \
     TYPE ret = v.v[0];                         \
     for (int i = 1; i < 16; ++i)               \
         ret = ret + v.v[i];                    \
     return ret;                                \
}

#define REDUCE_MINMAX(TYPE, VTYPE, NAME, OP)                    \
static FORCEINLINE TYPE NAME(VTYPE v) {                         \
    TYPE ret = v.v[0];                                          \
    for (int i = 1; i < 16; ++i)                                \
        ret = (ret OP (TYPE)v.v[i]) ? ret : (TYPE)v.v[i];       \
    return ret;                                                 \
}

#define SELECT(TYPE)                                                \
static FORCEINLINE TYPE __select(__vec16_i1 mask, TYPE a, TYPE b) { \
    TYPE ret;                                                       \
    for (int i = 0; i < 16; ++i)                                    \
        ret.v[i] = (mask.v & (1<<i)) ? a.v[i] : b.v[i];             \
    return ret;                                                     \
}                                                                   \
static FORCEINLINE TYPE __select(bool cond, TYPE a, TYPE b) {       \
    return cond ? a : b;                                            \
}

#define SHIFT_UNIFORM(TYPE, CAST, NAME, OP)                         \
static FORCEINLINE TYPE NAME(TYPE a, int32_t b) {                   \
   TYPE ret;                                                        \
   for (int i = 0; i < 16; ++i)                                     \
       ret.v[i] = (CAST)(a.v[i]) OP b;                              \
   return ret;                                                      \
}

#define SMEAR(VTYPE, NAME, STYPE)                                  \
template <class RetVecType> VTYPE __smear_##NAME(STYPE);           \
template <> FORCEINLINE VTYPE __smear_##NAME<VTYPE>(STYPE v) {     \
    VTYPE ret;                                                     \
    for (int i = 0; i < 16; ++i)                                   \
        ret.v[i] = v;                                              \
    return ret;                                                    \
}

#define SETZERO(VTYPE, NAME)                                       \
template <class RetVecType> VTYPE __setzero_##NAME();              \
template <> FORCEINLINE VTYPE __setzero_##NAME<VTYPE>() {          \
    VTYPE ret;                                                     \
    for (int i = 0; i < 16; ++i)                                   \
        ret.v[i] = 0;                                              \
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
    for (int i = 0; i < 16; ++i)                      \
        ret.v[i] = v.v[index & 0xf];                  \
    return ret;                                       \
}                                                     \

#define ROTATE(VTYPE, NAME, STYPE)                    \
static FORCEINLINE VTYPE __rotate_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i)                      \
        ret.v[i] = v.v[(i+index) & 0xf];              \
    return ret;                                       \
}                                                     \

#define SHIFT(VTYPE, NAME, STYPE)                    \
static FORCEINLINE VTYPE __shift_##NAME(VTYPE v, int index) {   \
    VTYPE ret;                                        \
    for (int i = 0; i < 16; ++i) {                    \
      int modIndex = i+index;                         \
      STYPE val = ((modIndex >= 0) && (modIndex < 16)) ? v.v[modIndex] : 0; \
      ret.v[i] = val;                                 \
    }                                                 \
    return ret;                                       \
}                                                     \

#define SHUFFLES(VTYPE, NAME, STYPE)                 \
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

///////////////////////////////////////////////////////////////////////////

INSERT_EXTRACT(__vec1_i8, int8_t)
INSERT_EXTRACT(__vec1_i16, int16_t)
INSERT_EXTRACT(__vec1_i32, int32_t)
INSERT_EXTRACT(__vec1_i64, int64_t)
INSERT_EXTRACT(__vec1_f, float)
INSERT_EXTRACT(__vec1_d, double)

///////////////////////////////////////////////////////////////////////////
// mask ops

static FORCEINLINE uint64_t __movmsk(__vec16_i1 mask) {
    return (uint64_t)mask.v;
}

static FORCEINLINE bool __any(__vec16_i1 mask) {
    return (mask.v!=0);
}

static FORCEINLINE bool __all(__vec16_i1 mask) {
    return (mask.v==0xFFFF);
}

static FORCEINLINE bool __none(__vec16_i1 mask) {
    return (mask.v==0);
}

static FORCEINLINE __vec16_i1 __equal_i1(__vec16_i1 a, __vec16_i1 b) {
    __vec16_i1 r;
    r.v = (a.v & b.v) | (~a.v & ~b.v);
    return r;
}

static FORCEINLINE __vec16_i1 __and(__vec16_i1 a, __vec16_i1 b) {
    __vec16_i1 r;
    r.v = a.v & b.v;
    return r;
}

static FORCEINLINE __vec16_i1 __xor(__vec16_i1 a, __vec16_i1 b) {
    __vec16_i1 r;
    r.v = a.v ^ b.v;
    return r;
}

static FORCEINLINE __vec16_i1 __or(__vec16_i1 a, __vec16_i1 b) {
    __vec16_i1 r;
    r.v = a.v | b.v;
    return r;
}

static FORCEINLINE __vec16_i1 __not(__vec16_i1 v) {
    __vec16_i1 r;
    r.v = ~v.v;
    return r;
}

static FORCEINLINE __vec16_i1 __and_not1(__vec16_i1 a, __vec16_i1 b) {
    __vec16_i1 r;
    r.v = ~a.v & b.v;
    return r;
}

static FORCEINLINE __vec16_i1 __and_not2(__vec16_i1 a, __vec16_i1 b) {
    __vec16_i1 r;
    r.v = a.v & ~b.v;
    return r;
}

static FORCEINLINE __vec16_i1 __select(__vec16_i1 mask, __vec16_i1 a, 
                                       __vec16_i1 b) {
    __vec16_i1 r;
    r.v = (a.v & mask.v) | (b.v & ~mask.v);
    return r;
}

static FORCEINLINE __vec16_i1 __select(bool cond, __vec16_i1 a, __vec16_i1 b) {
    return cond ? a : b;
}

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

template <int ALIGN> static FORCEINLINE __vec16_i1 __load(const __vec16_i1 *p) {
    uint16_t *ptr = (uint16_t *)p;
    __vec16_i1 r;
    r.v = *ptr;
    return r;
}

template <int ALIGN> static FORCEINLINE void __store(__vec16_i1 *p, __vec16_i1 v) {
    uint16_t *ptr = (uint16_t *)p;
    *ptr = v.v;
}

template <class RetVecType> __vec16_i1 __smear_i1(int i);
template <> FORCEINLINE __vec16_i1 __smear_i1<__vec16_i1>(int v) {
    return __vec16_i1(v, v, v, v, v, v, v, v, 
                      v, v, v, v, v, v, v, v);
}

template <class RetVecType> __vec16_i1 __setzero_i1();
template <> FORCEINLINE __vec16_i1 __setzero_i1<__vec16_i1>() {
    return __vec16_i1(0, 0, 0, 0, 0, 0, 0, 0, 
                      0, 0, 0, 0, 0, 0, 0, 0);
}

template <class RetVecType> __vec16_i1 __undef_i1();
template <> FORCEINLINE __vec16_i1 __undef_i1<__vec16_i1>() {
    return __vec16_i1();
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

BINARY_OP(__vec16_i32, __add, +)
BINARY_OP(__vec16_i32, __sub, -)
BINARY_OP(__vec16_i32, __mul, *)

BINARY_OP(__vec16_i32, __or, |)
BINARY_OP(__vec16_i32, __and, &)
BINARY_OP(__vec16_i32, __xor, ^)
BINARY_OP(__vec16_i32, __shl, <<)

BINARY_OP_CAST(__vec16_i32, uint32_t, __udiv, /)
BINARY_OP_CAST(__vec16_i32, int32_t,  __sdiv, /)

BINARY_OP_CAST(__vec16_i32, uint32_t, __urem, %)
BINARY_OP_CAST(__vec16_i32, int32_t,  __srem, %)
BINARY_OP_CAST(__vec16_i32, uint32_t, __lshr, >>)
BINARY_OP_CAST(__vec16_i32, int32_t,  __ashr, >>)

SHIFT_UNIFORM(__vec16_i32, uint32_t, __lshr, >>)
SHIFT_UNIFORM(__vec16_i32, int32_t, __ashr, >>)
SHIFT_UNIFORM(__vec16_i32, int32_t, __shl, <<)

CMP_OP(__vec16_i32, i32, int32_t,  __equal, ==)
CMP_OP(__vec16_i32, i32, int32_t,  __not_equal, !=)
CMP_OP(__vec16_i32, i32, uint32_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i32, i32, int32_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i32, i32, uint32_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i32, i32, int32_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i32, i32, uint32_t, __unsigned_less_than, <)
CMP_OP(__vec16_i32, i32, int32_t,  __signed_less_than, <)
CMP_OP(__vec16_i32, i32, uint32_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i32, i32, int32_t,  __signed_greater_than, >)

SELECT(__vec16_i32)
INSERT_EXTRACT(__vec16_i32, int32_t)
SMEAR(__vec16_i32, i32, int32_t)
SETZERO(__vec16_i32, i32)
UNDEF(__vec16_i32, i32)
BROADCAST(__vec16_i32, i32, int32_t)
ROTATE(__vec16_i32, i32, int32_t)
SHIFT(__vec16_i32, i32, int32_t)
SHUFFLES(__vec16_i32, i32, int32_t)
LOAD_STORE(__vec16_i32, int32_t)

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

CMP_OP(__vec16_i64, i64, int64_t,  __equal, ==)
CMP_OP(__vec16_i64, i64, int64_t,  __not_equal, !=)
CMP_OP(__vec16_i64, i64, uint64_t, __unsigned_less_equal, <=)
CMP_OP(__vec16_i64, i64, int64_t,  __signed_less_equal, <=)
CMP_OP(__vec16_i64, i64, uint64_t, __unsigned_greater_equal, >=)
CMP_OP(__vec16_i64, i64, int64_t,  __signed_greater_equal, >=)
CMP_OP(__vec16_i64, i64, uint64_t, __unsigned_less_than, <)
CMP_OP(__vec16_i64, i64, int64_t,  __signed_less_than, <)
CMP_OP(__vec16_i64, i64, uint64_t, __unsigned_greater_than, >)
CMP_OP(__vec16_i64, i64, int64_t,  __signed_greater_than, >)

SELECT(__vec16_i64)
INSERT_EXTRACT(__vec16_i64, int64_t)
SMEAR(__vec16_i64, i64, int64_t)
SETZERO(__vec16_i64, i64)
UNDEF(__vec16_i64, i64)
BROADCAST(__vec16_i64, i64, int64_t)
ROTATE(__vec16_i64, i64, int64_t)
SHIFT(__vec16_i64, i64, int64_t)
SHUFFLES(__vec16_i64, i64, int64_t)
LOAD_STORE(__vec16_i64, int64_t)

///////////////////////////////////////////////////////////////////////////
// float

BINARY_OP(__vec16_f, __add, +)
BINARY_OP(__vec16_f, __sub, -)
BINARY_OP(__vec16_f, __mul, *)
BINARY_OP(__vec16_f, __div, /)

CMP_OP(__vec16_f, float, float, __equal, ==)
CMP_OP(__vec16_f, float, float, __not_equal, !=)
CMP_OP(__vec16_f, float, float, __less_than, <)
CMP_OP(__vec16_f, float, float, __less_equal, <=)
CMP_OP(__vec16_f, float, float, __greater_than, >)
CMP_OP(__vec16_f, float, float, __greater_equal, >=)

static FORCEINLINE __vec16_i1 __ordered_float(__vec16_f a, __vec16_f b) {
    __vec16_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 16; ++i)
        ret.v |= ((a.v[i] == a.v[i]) && (b.v[i] == b.v[i])) ? (1 << i) : 0;
    return ret;
}

static FORCEINLINE __vec16_i1 __unordered_float(__vec16_f a, __vec16_f b) {
    __vec16_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 16; ++i)
        ret.v |= ((a.v[i] != a.v[i]) || (b.v[i] != b.v[i])) ? (1 << i) : 0;
    return ret;
}

#if 0
      case Instruction::FRem: intrinsic = "__frem"; break;
#endif

SELECT(__vec16_f)
INSERT_EXTRACT(__vec16_f, float)
SMEAR(__vec16_f, float, float)
SETZERO(__vec16_f, float)
UNDEF(__vec16_f, float)
BROADCAST(__vec16_f, float, float)
ROTATE(__vec16_f, float, float)
SHIFT(__vec16_f, float, float)
SHUFFLES(__vec16_f, float, float)
LOAD_STORE(__vec16_f, float)

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


static FORCEINLINE __vec16_f __half_to_float_varying(__vec16_i16 v) {
    __vec16_f ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = __half_to_float_uniform(v.v[i]);
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


static FORCEINLINE __vec16_i16 __float_to_half_varying(__vec16_f v) {
    __vec16_i16 ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = __float_to_half_uniform(v.v[i]);
    return ret;
}


///////////////////////////////////////////////////////////////////////////
// double

BINARY_OP(__vec16_d, __add, +)
BINARY_OP(__vec16_d, __sub, -)
BINARY_OP(__vec16_d, __mul, *)
BINARY_OP(__vec16_d, __div, /)

CMP_OP(__vec16_d, double, double, __equal, ==)
CMP_OP(__vec16_d, double, double, __not_equal, !=)
CMP_OP(__vec16_d, double, double, __less_than, <)
CMP_OP(__vec16_d, double, double, __less_equal, <=)
CMP_OP(__vec16_d, double, double, __greater_than, >)
CMP_OP(__vec16_d, double, double, __greater_equal, >=)

static FORCEINLINE __vec16_i1 __ordered_double(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 16; ++i)
        ret.v |= ((a.v[i] == a.v[i]) && (b.v[i] == b.v[i])) ? (1 << i) : 0;
    return ret;
}

static FORCEINLINE __vec16_i1 __unordered_double(__vec16_d a, __vec16_d b) {
    __vec16_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 16; ++i)
        ret.v |= ((a.v[i] != a.v[i]) || (b.v[i] != b.v[i])) ? (1 << i) : 0;
    return ret;
}

#if 0
      case Instruction::FRem: intrinsic = "__frem"; break;
#endif

SELECT(__vec16_d)
INSERT_EXTRACT(__vec16_d, double)
SMEAR(__vec16_d, double, double)
SETZERO(__vec16_d, double)
UNDEF(__vec16_d, double)
BROADCAST(__vec16_d, double, double)
ROTATE(__vec16_d, double, double)
SHIFT(__vec16_d, double, double)
SHUFFLES(__vec16_d, double, double)
LOAD_STORE(__vec16_d, double)

///////////////////////////////////////////////////////////////////////////
// casts


#define CAST(TO, STO, FROM, SFROM, FUNC)        \
static FORCEINLINE TO FUNC(TO, FROM val) {      \
    TO ret;                                     \
    for (int i = 0; i < 16; ++i)                \
        ret.v[i] = (STO)((SFROM)(val.v[i]));    \
    return ret;                                 \
}

// sign extension conversions
CAST(__vec16_i64, int64_t, __vec16_i32, int32_t, __cast_sext)
CAST(__vec16_i64, int64_t, __vec16_i16, int16_t, __cast_sext)
CAST(__vec16_i64, int64_t, __vec16_i8,  int8_t,  __cast_sext)
CAST(__vec16_i32, int32_t, __vec16_i16, int16_t, __cast_sext)
CAST(__vec16_i32, int32_t, __vec16_i8,  int8_t,  __cast_sext)
CAST(__vec16_i16, int16_t, __vec16_i8,  int8_t,  __cast_sext)

#define CAST_SEXT_I1(TYPE)                            \
static FORCEINLINE TYPE __cast_sext(TYPE, __vec16_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 16; ++i) {                    \
        ret.v[i] = 0;                                 \
        if (v.v & (1 << i))                           \
            ret.v[i] = ~ret.v[i];                     \
    }                                                 \
    return ret;                                       \
}

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

#define CAST_ZEXT_I1(TYPE)                            \
static FORCEINLINE TYPE __cast_zext(TYPE, __vec16_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 16; ++i)                      \
        ret.v[i] = (v.v & (1 << i)) ? 1 : 0;          \
    return ret;                                       \
}

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
CAST(__vec16_f, float, __vec16_i8,   int8_t,  __cast_sitofp)
CAST(__vec16_f, float, __vec16_i16,  int16_t, __cast_sitofp)
CAST(__vec16_f, float, __vec16_i32,  int32_t, __cast_sitofp)
CAST(__vec16_f, float, __vec16_i64,  int64_t, __cast_sitofp)
CAST(__vec16_d, double, __vec16_i8,  int8_t,  __cast_sitofp)
CAST(__vec16_d, double, __vec16_i16, int16_t, __cast_sitofp)
CAST(__vec16_d, double, __vec16_i32, int32_t, __cast_sitofp)
CAST(__vec16_d, double, __vec16_i64, int64_t, __cast_sitofp)

// unsigned int to float/double
CAST(__vec16_f, float, __vec16_i8,   uint8_t,  __cast_uitofp)
CAST(__vec16_f, float, __vec16_i16,  uint16_t, __cast_uitofp)
CAST(__vec16_f, float, __vec16_i32,  uint32_t, __cast_uitofp)
CAST(__vec16_f, float, __vec16_i64,  uint64_t, __cast_uitofp)
CAST(__vec16_d, double, __vec16_i8,  uint8_t,  __cast_uitofp)
CAST(__vec16_d, double, __vec16_i16, uint16_t, __cast_uitofp)
CAST(__vec16_d, double, __vec16_i32, uint32_t, __cast_uitofp)
CAST(__vec16_d, double, __vec16_i64, uint64_t, __cast_uitofp)

static FORCEINLINE __vec16_f __cast_uitofp(__vec16_f, __vec16_i1 v) {
    __vec16_f ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = (v.v & (1 << i)) ? 1. : 0.;
    return ret;
}

// float/double to signed int
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

typedef union {
    int32_t i32;
    float f;
    int64_t i64;
    double d;
} BitcastUnion;

#define CAST_BITS(TO, TO_ELT, FROM, FROM_ELT)       \
static FORCEINLINE TO __cast_bits(TO, FROM val) {   \
    TO r;                                           \
    for (int i = 0; i < 16; ++i) {                  \
        BitcastUnion u;                             \
        u.FROM_ELT = val.v[i];                      \
        r.v[i] = u.TO_ELT;                          \
    }                                               \
    return r;                                       \
}

CAST_BITS(__vec16_f,   f,   __vec16_i32, i32)
CAST_BITS(__vec16_i32, i32, __vec16_f,   f)
CAST_BITS(__vec16_d,   d,   __vec16_i64, i64)
CAST_BITS(__vec16_i64, i64, __vec16_d,   d)

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

#define CAST_BITS_TRIVIAL(TYPE)                  \
static FORCEINLINE TYPE __cast_bits(TYPE, TYPE v) { return v; }

CAST_BITS_TRIVIAL(float)
CAST_BITS_TRIVIAL(double)
CAST_BITS_TRIVIAL(int8_t)
CAST_BITS_TRIVIAL(uint8_t)
CAST_BITS_TRIVIAL(int16_t)
CAST_BITS_TRIVIAL(uint16_t)
CAST_BITS_TRIVIAL(int32_t)
CAST_BITS_TRIVIAL(uint32_t)
CAST_BITS_TRIVIAL(int64_t)
CAST_BITS_TRIVIAL(uint64_t)
CAST_BITS_TRIVIAL(__vec16_f)
CAST_BITS_TRIVIAL(__vec16_d)
CAST_BITS_TRIVIAL(__vec16_i8)
CAST_BITS_TRIVIAL(__vec16_i16)
CAST_BITS_TRIVIAL(__vec16_i32)
CAST_BITS_TRIVIAL(__vec16_i64)
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

UNARY_OP(__vec16_f, __round_varying_float, roundf)
UNARY_OP(__vec16_f, __floor_varying_float, floorf)
UNARY_OP(__vec16_f, __ceil_varying_float, ceilf)
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


BINARY_OP_FUNC(__vec16_f, __max_varying_float, __max_uniform_float)
BINARY_OP_FUNC(__vec16_f, __min_varying_float, __min_uniform_float)
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

static FORCEINLINE double __rsqrt_uniform_double(double v) {
    return 1.0 / sqrt(v);
}

static FORCEINLINE float __rcp_uniform_float(float v) {
    return 1.f / v;
}

static FORCEINLINE double __rcp_uniform_double(double v) {
    return 1.0 / v;
}

static FORCEINLINE float __sqrt_uniform_float(float v) {
    return sqrtf(v);
}

static FORCEINLINE double __sqrt_uniform_double(double v) {
    return sqrt(v);
}

UNARY_OP(__vec16_f, __rcp_varying_float, __rcp_uniform_float)
UNARY_OP(__vec16_d, __rcp_varying_double, __rcp_uniform_double)
UNARY_OP(__vec16_f, __rsqrt_varying_float, __rsqrt_uniform_float)
UNARY_OP(__vec16_d, __rsqrt_varying_double, __rsqrt_uniform_double)
UNARY_OP(__vec16_f, __sqrt_varying_float, __sqrt_uniform_float)
UNARY_OP(__vec16_d, __sqrt_varying_double, __sqrt_uniform_double)

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

REDUCE_ADD(float, __vec16_f, __reduce_add_float)
REDUCE_MINMAX(float, __vec16_f, __reduce_min_float, <)
REDUCE_MINMAX(float, __vec16_f, __reduce_max_float, >)

REDUCE_ADD(double, __vec16_d, __reduce_add_double)
REDUCE_MINMAX(double, __vec16_d, __reduce_min_double, <)
REDUCE_MINMAX(double, __vec16_d, __reduce_max_double, >)

REDUCE_ADD(int16_t, __vec16_i8, __reduce_add_int8)
REDUCE_ADD(int32_t, __vec16_i16, __reduce_add_int16)

REDUCE_ADD(int64_t, __vec16_i32, __reduce_add_int32)
REDUCE_MINMAX(int32_t, __vec16_i32, __reduce_min_int32, <)
REDUCE_MINMAX(int32_t, __vec16_i32, __reduce_max_int32, >)

REDUCE_MINMAX(uint32_t, __vec16_i32, __reduce_min_uint32, <)
REDUCE_MINMAX(uint32_t, __vec16_i32, __reduce_max_uint32, >)

REDUCE_ADD(int64_t, __vec16_i64, __reduce_add_int64)
REDUCE_MINMAX(int64_t, __vec16_i64, __reduce_min_int64, <)
REDUCE_MINMAX(int64_t, __vec16_i64, __reduce_max_int64, >)

REDUCE_MINMAX(uint64_t, __vec16_i64, __reduce_min_uint64, <)
REDUCE_MINMAX(uint64_t, __vec16_i64, __reduce_max_uint64, >)

///////////////////////////////////////////////////////////////////////////
// masked load/store

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

static FORCEINLINE __vec16_i32 __masked_load_i32(void *p,
                                                 __vec16_i1 mask) {
    __vec16_i32 ret;
    int32_t *ptr = (int32_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret.v[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec16_f __masked_load_float(void *p,
                                                 __vec16_i1 mask) {
    __vec16_f ret;
    float *ptr = (float *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret.v[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec16_i64 __masked_load_i64(void *p,
                                                 __vec16_i1 mask) {
    __vec16_i64 ret;
    int64_t *ptr = (int64_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret.v[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec16_d __masked_load_double(void *p,
                                                  __vec16_i1 mask) {
    __vec16_d ret;
    double *ptr = (double *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret.v[i] = ptr[i];
    return ret;
}

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

static FORCEINLINE void __masked_store_i32(void *p, __vec16_i32 val,
                                           __vec16_i1 mask) {
    int32_t *ptr = (int32_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
}

static FORCEINLINE void __masked_store_float(void *p, __vec16_f val,
                                             __vec16_i1 mask) {
    float *ptr = (float *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
}

static FORCEINLINE void __masked_store_i64(void *p, __vec16_i64 val,
                                          __vec16_i1 mask) {
    int64_t *ptr = (int64_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
}

static FORCEINLINE void __masked_store_double(void *p, __vec16_d val,
                                              __vec16_i1 mask) {
    double *ptr = (double *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
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

// offsets * offsetScale is in bytes (for all of these)

#define GATHER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)                  \
static FORCEINLINE VTYPE FUNC(unsigned char *b, uint32_t scale,         \
                              OTYPE offset, __vec16_i1 mask) {          \
    VTYPE ret;                                                          \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 16; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * offset.v[i]);         \
            ret.v[i] = *ptr;                                            \
        }                                                               \
    return ret;                                                         \
}
    

GATHER_BASE_OFFSETS(__vec16_i8,  int8_t,  __vec16_i32, __gather_base_offsets32_i8)
GATHER_BASE_OFFSETS(__vec16_i8,  int8_t,  __vec16_i64, __gather_base_offsets64_i8)
GATHER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i32, __gather_base_offsets32_i16)
GATHER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i64, __gather_base_offsets64_i16)
GATHER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i32, __gather_base_offsets32_i32)
GATHER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i64, __gather_base_offsets64_i32)
GATHER_BASE_OFFSETS(__vec16_f,   float,   __vec16_i32, __gather_base_offsets32_float)
GATHER_BASE_OFFSETS(__vec16_f,   float,   __vec16_i64, __gather_base_offsets64_float)
GATHER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i32, __gather_base_offsets32_i64)
GATHER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i64, __gather_base_offsets64_i64)
GATHER_BASE_OFFSETS(__vec16_d,   double,  __vec16_i32, __gather_base_offsets32_double)
GATHER_BASE_OFFSETS(__vec16_d,   double,  __vec16_i64, __gather_base_offsets64_double)

#define GATHER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)         \
static FORCEINLINE VTYPE FUNC(PTRTYPE ptrs, __vec16_i1 mask) {   \
    VTYPE ret;                                              \
    for (int i = 0; i < 16; ++i)                            \
        if ((mask.v & (1 << i)) != 0) {                     \
            STYPE *ptr = (STYPE *)ptrs.v[i];                \
            ret.v[i] = *ptr;                                \
        }                                                   \
    return ret;                                             \
}

GATHER_GENERAL(__vec16_i8,  int8_t,  __vec16_i32, __gather32_i8)
GATHER_GENERAL(__vec16_i8,  int8_t,  __vec16_i64, __gather64_i8)
GATHER_GENERAL(__vec16_i16, int16_t, __vec16_i32, __gather32_i16)
GATHER_GENERAL(__vec16_i16, int16_t, __vec16_i64, __gather64_i16)
GATHER_GENERAL(__vec16_i32, int32_t, __vec16_i32, __gather32_i32)
GATHER_GENERAL(__vec16_i32, int32_t, __vec16_i64, __gather64_i32)
GATHER_GENERAL(__vec16_f,   float,   __vec16_i32, __gather32_float)
GATHER_GENERAL(__vec16_f,   float,   __vec16_i64, __gather64_float)
GATHER_GENERAL(__vec16_i64, int64_t, __vec16_i32, __gather32_i64)
GATHER_GENERAL(__vec16_i64, int64_t, __vec16_i64, __gather64_i64)
GATHER_GENERAL(__vec16_d,   double,  __vec16_i32, __gather32_double)
GATHER_GENERAL(__vec16_d,   double,  __vec16_i64, __gather64_double)

// scatter

#define SCATTER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)                 \
static FORCEINLINE void FUNC(unsigned char *b, uint32_t scale,          \
                             OTYPE offset, VTYPE val,                   \
                             __vec16_i1 mask) {                         \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 16; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * offset.v[i]);         \
            *ptr = val.v[i];                                            \
        }                                                               \
}
    

SCATTER_BASE_OFFSETS(__vec16_i8,  int8_t,  __vec16_i32, __scatter_base_offsets32_i8)
SCATTER_BASE_OFFSETS(__vec16_i8,  int8_t,  __vec16_i64, __scatter_base_offsets64_i8)
SCATTER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i32, __scatter_base_offsets32_i16)
SCATTER_BASE_OFFSETS(__vec16_i16, int16_t, __vec16_i64, __scatter_base_offsets64_i16)
SCATTER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i32, __scatter_base_offsets32_i32)
SCATTER_BASE_OFFSETS(__vec16_i32, int32_t, __vec16_i64, __scatter_base_offsets64_i32)
SCATTER_BASE_OFFSETS(__vec16_f,   float,   __vec16_i32, __scatter_base_offsets32_float)
SCATTER_BASE_OFFSETS(__vec16_f,   float,   __vec16_i64, __scatter_base_offsets64_float)
SCATTER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i32, __scatter_base_offsets32_i64)
SCATTER_BASE_OFFSETS(__vec16_i64, int64_t, __vec16_i64, __scatter_base_offsets64_i64)
SCATTER_BASE_OFFSETS(__vec16_d,   double,  __vec16_i32, __scatter_base_offsets32_double)
SCATTER_BASE_OFFSETS(__vec16_d,   double,  __vec16_i64, __scatter_base_offsets64_double)

#define SCATTER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)                 \
static FORCEINLINE void FUNC(PTRTYPE ptrs, VTYPE val, __vec16_i1 mask) {  \
    VTYPE ret;                                                       \
    for (int i = 0; i < 16; ++i)                                     \
        if ((mask.v & (1 << i)) != 0) {                              \
            STYPE *ptr = (STYPE *)ptrs.v[i];                         \
            *ptr = val.v[i];                                         \
        }                                                            \
}

SCATTER_GENERAL(__vec16_i8,  int8_t,  __vec16_i32, __scatter32_i8)
SCATTER_GENERAL(__vec16_i8,  int8_t,  __vec16_i64, __scatter64_i8)
SCATTER_GENERAL(__vec16_i16, int16_t, __vec16_i32, __scatter32_i16)
SCATTER_GENERAL(__vec16_i16, int16_t, __vec16_i64, __scatter64_i16)
SCATTER_GENERAL(__vec16_i32, int32_t, __vec16_i32, __scatter32_i32)
SCATTER_GENERAL(__vec16_i32, int32_t, __vec16_i64, __scatter64_i32)
SCATTER_GENERAL(__vec16_f,   float,   __vec16_i32, __scatter32_float)
SCATTER_GENERAL(__vec16_f,   float,   __vec16_i64, __scatter64_float)
SCATTER_GENERAL(__vec16_i64, int64_t, __vec16_i32, __scatter32_i64)
SCATTER_GENERAL(__vec16_i64, int64_t, __vec16_i64, __scatter64_i64)
SCATTER_GENERAL(__vec16_d,   double,  __vec16_i32, __scatter32_double)
SCATTER_GENERAL(__vec16_d,   double,  __vec16_i64, __scatter64_double)

///////////////////////////////////////////////////////////////////////////
// packed load/store

static FORCEINLINE int32_t __packed_load_active(int32_t *ptr, __vec16_i32 *val,
                                                __vec16_i1 mask) {
    int count = 0; 
    for (int i = 0; i < 16; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            val->v[i] = *ptr++;
            ++count;
        }
    }
    return count;
}


static FORCEINLINE int32_t __packed_store_active(int32_t *ptr, __vec16_i32 val,
                                                 __vec16_i1 mask) {
    int count = 0; 
    for (int i = 0; i < 16; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            *ptr++ = val.v[i];
            ++count;
        }
    }
    return count;
}


static FORCEINLINE int32_t __packed_store_active2(int32_t *ptr, __vec16_i32 val,
                                                 __vec16_i1 mask) {
    int count = 0;
    int32_t *ptr_ = ptr;
    for (int i = 0; i < 16; ++i) {
        *ptr = val.v[i];
        ptr += mask.v & 1;
        mask.v = mask.v >> 1;
    }
    return ptr - ptr_;
}


static FORCEINLINE int32_t __packed_load_active(uint32_t *ptr,
                                                __vec16_i32 *val,
                                                __vec16_i1 mask) {
    return __packed_load_active((int32_t *)ptr, val, mask);
}


static FORCEINLINE int32_t __packed_store_active(uint32_t *ptr, 
                                                 __vec16_i32 val,
                                                 __vec16_i1 mask) {
    return __packed_store_active((int32_t *)ptr, val, mask);
}


static FORCEINLINE int32_t __packed_store_active2(uint32_t *ptr,
                                                 __vec16_i32 val,
                                                 __vec16_i1 mask) {
    return __packed_store_active2((int32_t *)ptr, val, mask);
}


///////////////////////////////////////////////////////////////////////////
// aos/soa

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

static FORCEINLINE void __prefetch_read_uniform_1(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_2(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_3(unsigned char *) {
}

static FORCEINLINE void __prefetch_read_uniform_nt(unsigned char *) {
}

#define PREFETCH_READ_VARYING(CACHE_NUM)                                                                    \
static FORCEINLINE void __prefetch_read_varying_##CACHE_NUM##_native(uint8_t *base, uint32_t scale,         \
                                                                   __vec16_i32 offsets, __vec16_i1 mask) {} \
static FORCEINLINE void __prefetch_read_varying_##CACHE_NUM(__vec16_i64 addr, __vec16_i1 mask) {}           \

PREFETCH_READ_VARYING(1)
PREFETCH_READ_VARYING(2)
PREFETCH_READ_VARYING(3)
PREFETCH_READ_VARYING(nt)
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


///////////////////////////////////////////////////////////////////////////
// Transcendentals
//
//
#define TRANSCENDENTALS(op) \
static FORCEINLINE float __##op##_uniform_float(float v) { \
    return op##f(v); \
} \
static FORCEINLINE __vec16_f __##op##_varying_float(__vec16_f v) { \
    __vec16_f ret; \
    for (int i = 0; i < 16; ++i) \
        ret.v[i] = op##f(v.v[i]); \
    return ret; \
} \
static FORCEINLINE double __##op##_uniform_double(double v) { \
    return op(v); \
} \
static FORCEINLINE __vec16_d __##op##_varying_double(__vec16_d v) { \
    __vec16_d ret; \
    for (int i = 0; i < 16; ++i) \
        ret.v[i] = op(v.v[i]); \
    return ret; \
}

  TRANSCENDENTALS(log)
TRANSCENDENTALS(exp)


static FORCEINLINE float __pow_uniform_float(float a, float b) {
    return powf(a, b);
}
static FORCEINLINE __vec16_f __pow_varying_float(__vec16_f a, __vec16_f b) {
    __vec16_f ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = powf(a.v[i], b.v[i]);
    return ret;
}
static FORCEINLINE double __pow_uniform_double(double a, double b) {
    return pow(a, b);
}
static FORCEINLINE __vec16_d __pow_varying_double(__vec16_d a, __vec16_d b) {
    __vec16_d ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = pow(a.v[i], b.v[i]);
    return ret;
}

///////////////////////////////////////////////////////////////////////////
// Trigonometry

TRANSCENDENTALS(sin)
TRANSCENDENTALS(asin)
TRANSCENDENTALS(cos)
TRANSCENDENTALS(acos)
TRANSCENDENTALS(tan)
TRANSCENDENTALS(atan)


static FORCEINLINE float __atan2_uniform_float(float a, float b) {
    return atan2f(a, b);
}
static FORCEINLINE __vec16_f __atan2_varying_float(__vec16_f a, __vec16_f b) {
    __vec16_f ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = atan2f(a.v[i], b.v[i]);
    return ret;
}
static FORCEINLINE double __atan2_uniform_double(double a, double b) {
    return atan2(a, b);
}
static FORCEINLINE __vec16_d __atan2_varying_double(__vec16_d a, __vec16_d b) {
    __vec16_d ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = atan2(a.v[i], b.v[i]);
    return ret;
}

static FORCEINLINE void __sincos_uniform_float(float x, float *a, float *b) {
    sincosf(x,a,b);
}
static FORCEINLINE void __sincos_varying_float(__vec16_f x, __vec16_f *a, __vec16_f *b) {
    __vec16_f ret;
    for (int i = 0; i < 16; ++i)
        sincosf(x.v[i], (float*)a + i, (float*)b+i);
}
static FORCEINLINE void __sincos_uniform_double(double x, double *a, double *b) {
    sincos(x,a,b);
}
static FORCEINLINE void __sincos_varying_double(__vec16_d x, __vec16_d *a, __vec16_d *b) {
    __vec16_d ret;
    for (int i = 0; i < 16; ++i)
        sincos(x.v[i], (double*)a + i, (double*)b+i);
}
