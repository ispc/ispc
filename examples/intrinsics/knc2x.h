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

#include <assert.h>
#include <math.h>
#include <stdint.h>

#include <immintrin.h>
#include <zmmintrin.h>

#include "knc.h" // TODO: this should be possible

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#define PRE_ALIGN(x) /*__declspec(align(x))*/
#define POST_ALIGN(x)
#define roundf(x) (floorf(x + .5f))
#define round(x) (floor(x + .5))
#else
#define FORCEINLINE __attribute__((always_inline))
#define PRE_ALIGN(x)
#define POST_ALIGN(x) __attribute__((aligned(x)))
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

struct __vec32_f;
struct __vec32_i32;

typedef struct PRE_ALIGN(4) __vec32_i1 {
    __vec32_i1() {} // FIXME? __mm512_undef_mask();
    __vec32_i1(const uint32_t &in) { m = in; }
    __vec32_i1(const __vec32_i32 &in);
    __vec32_i1(uint32_t v00, uint32_t v01, uint32_t v02, uint32_t v03, uint32_t v04, uint32_t v05, uint32_t v06,
               uint32_t v07, uint32_t v08, uint32_t v09, uint32_t v10, uint32_t v11, uint32_t v12, uint32_t v13,
               uint32_t v14, uint32_t v15, uint32_t v16, uint32_t v17, uint32_t v18, uint32_t v19, uint32_t v20,
               uint32_t v21, uint32_t v22, uint32_t v23, uint32_t v24, uint32_t v25, uint32_t v26, uint32_t v27,
               uint32_t v28, uint32_t v29, uint32_t v30, uint32_t v31) {
        m16.m1 = (v00) | ((v01) << 1) | ((v02) << 2) | ((v03) << 3) | ((v04) << 4) | ((v05) << 5) | ((v06) << 6) |
                 ((v07) << 7) | ((v08) << 8) | ((v09) << 9) | ((v10) << 10) | ((v11) << 11) | ((v12) << 12) |
                 ((v13) << 13) | ((v14) << 14) | ((v15) << 15);
        m16.m2 = (v16) | ((v17) << 1) | ((v18) << 2) | ((v19) << 3) | ((v20) << 4) | ((v21) << 5) | ((v22) << 6) |
                 ((v23) << 7) | ((v24) << 8) | ((v25) << 9) | ((v26) << 10) | ((v27) << 11) | ((v28) << 12) |
                 ((v29) << 13) | ((v30) << 14) | ((v31) << 15);
    }

    union {
        uint32_t m;
        struct {
            __mmask16 m1;
            __mmask16 m2;
        } m16;
    };
} POST_ALIGN(4) __vec32_i1;

typedef struct PRE_ALIGN(64) __vec32_f {
    __vec32_f() : v1(_mm512_undefined_ps()), v2(_mm512_undefined_ps()) {}
    __vec32_f(float v00, float v01, float v02, float v03, float v04, float v05, float v06, float v07, float v08,
              float v09, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17,
              float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26,
              float v27, float v28, float v29, float v30, float v31) {
        v2 = _mm512_set_16to16_ps(v15, v14, v13, v12, v11, v10, v09, v08, v07, v06, v05, v04, v03, v02, v01, v00);
        v1 = _mm512_set_16to16_ps(v31, v30, v29, v28, v27, v26, v25, v24, v23, v22, v21, v20, v19, v18, v17, v16);
    }
    __m512 v1;
    __m512 v2;
} POST_ALIGN(64) __vec32_f;

typedef struct PRE_ALIGN(64) __vec32_i32 {
    __vec32_i32() v1(_mm512_undefined_epi32()), v2(_mm512_undefined_epi32()) {}
    __vec32_i32(const __vec32_i1 &in) {
        __mmask16 m;
        v1 = _mm512_setzero_epi32(); // _mm512_xor_epi32(zero, zero);
        v1 = _mm512_sbb_epi32(v1, in.m16.m1, v1, &m);
        v2 = _mm512_setzero_epi32();
        v2 = _mm512_sbb_epi32(v2, in.m16.m2, v2, &m);
    }
    __vec32_i32(int32_t v00, int32_t v01, int32_t v02, int32_t v03, int32_t v04, int32_t v05, int32_t v06, int32_t v07,
                int32_t v08, int32_t v09, int32_t v10, int32_t v11, int32_t v12, int32_t v13, int32_t v14, int32_t v15,
                int32_t v16, int32_t v17, int32_t v18, int32_t v19, int32_t v20, int32_t v21, int32_t v22, int32_t v23,
                int32_t v24, int32_t v25, int32_t v26, int32_t v27, int32_t v28, int32_t v29, int32_t v30,
                int32_t v31) {
        v1 = _mm512_set_16to16_pi(v15, v14, v13, v12, v11, v10, v09, v08, v07, v06, v05, v04, v03, v02, v01, v00);
        v2 = _mm512_set_16to16_pi(v31, v30, v29, v28, v27, v26, v25, v24, v23, v22, v21, v20, v19, v18, v17, v16);
    }
    __m512i v1;
    __m512i v2;
} POST_ALIGN(64) __vec32_i32;

FORCEINLINE __vec32_i1::__vec32_i1(const __vec32_i32 &in) {
    m16.m1 = _mm512_test_epi32_mask(in.v1, in.v1);
    m16.m2 = _mm512_test_epi32_mask(in.v2, in.v2);
}

// This does not map directly to an intrinsic type
typedef struct PRE_ALIGN(64) __vec32_d {
    double v[32];
} POST_ALIGN(64) __vec32_d;

template <typename T> struct vec32 {
    vec32() {}
    vec32(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15, T v16,
          T v17, T v18, T v19, T v20, T v21, T v22, T v23, T v24, T v25, T v26, T v27, T v28, T v29, T v30, T v31) {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
        v[3] = v3;
        v[4] = v4;
        v[5] = v5;
        v[6] = v6;
        v[7] = v7;
        v[8] = v8;
        v[9] = v9;
        v[10] = v10;
        v[11] = v11;
        v[12] = v12;
        v[13] = v13;
        v[14] = v14;
        v[15] = v15;
        v[16] = v16;
        v[17] = v17;
        v[18] = v18;
        v[19] = v19;
        v[20] = v20;
        v[21] = v21;
        v[22] = v22;
        v[23] = v23;
        v[24] = v24;
        v[25] = v25;
        v[26] = v26;
        v[27] = v27;
        v[28] = v28;
        v[29] = v29;
        v[30] = v30;
        v[31] = v31;
    }
    T v[32];
};

/*
PRE_ALIGN(64) struct __vec32_f : public vec16<float> {
    __vec32_f() { }
    __vec32_f(float v0, float v1, float v2, float v3,
              float v4, float v5, float v6, float v7,
              float v8, float v9, float v10, float v11,
              float v12, float v13, float v14, float v15)
        : vec16<float>(v0, v1, v2, v3, v4, v5, v6, v7,
                       v8, v9, v10, v11, v12, v13, v14, v15) { }

} POST_ALIGN(64);

PRE_ALIGN(128) struct __vec32_d : public vec16<double> {
    __vec32_d() { }
    __vec32_d(double v0, double v1, double v2, double v3,
              double v4, double v5, double v6, double v7,
              double v8, double v9, double v10, double v11,
              double v12, double v13, double v14, double v15)
        : vec16<double>(v0, v1, v2, v3, v4, v5, v6, v7,
                        v8, v9, v10, v11, v12, v13, v14, v15) { }

} POST_ALIGN(128);
*/

PRE_ALIGN(32) struct __vec32_i8 : public vec32<int8_t> {
    __vec32_i8() {}
    __vec32_i8(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5, int8_t v6, int8_t v7, int8_t v8,
               int8_t v9, int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15, int8_t v16,
               int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23, int8_t v24,
               int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31)
        : vec32<int8_t>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v17, v18, v19, v20, v21,
                        v22, v23, v24, v25, v25, v26, v27, v28, v29, v30, v31) {}
} POST_ALIGN(32);

PRE_ALIGN(64) struct __vec32_i16 : public vec32<int16_t> {
    __vec32_i16() {}
    __vec32_i16(int16_t v0, int16_t v1, int16_t v2, int16_t v3, int16_t v4, int16_t v5, int16_t v6, int16_t v7,
                int16_t v8, int16_t v9, int16_t v10, int16_t v11, int16_t v12, int16_t v13, int16_t v14, int16_t v15,
                int16_t v16, int16_t v17, int16_t v18, int16_t v19, int16_t v20, int16_t v21, int16_t v22, int16_t v23,
                int16_t v24, int16_t v25, int16_t v26, int16_t v27, int16_t v28, int16_t v29, int16_t v30, int16_t v31)
        : vec32<int16_t>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                         v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31) {}
} POST_ALIGN(64);

/*
PRE_ALIGN(64) struct __vec32_i32  : public vec16<int32_t> {
    __vec32_i32() { }
    __vec32_i32(int32_t v0, int32_t v1, int32_t v2, int32_t v3,
                int32_t v4, int32_t v5, int32_t v6, int32_t v7,
                int32_t v8, int32_t v9, int32_t v10, int32_t v11,
                int32_t v12, int32_t v13, int32_t v14, int32_t v15)
      : v.m512i_i32[0](v0), v.m512i_i32[1](v1), v.m512i_i32[2](v2), v.m512i_i32[3](v3),
        v.m512i_i32[4](v4), v.m512i_i32[5](v5), v.m512i_i32[6](v6), v.m512i_i32[7](v7),
        v.m512i_i32[8](v8), v.m512i_i32[9](v9), v.m512i_i32[10](v10), v.m512i_i32[11](v11),
        v.m512i_i32[12](v12), v.m512i_i32[13](v13), v.m512i_i32[14](v14), v.m512i_i32[15](v15), { }
  _#512i v;
} POST_ALIGN(64);

static inline int32_t __extract_element(__vec32_i32, int);

PRE_ALIGN(128) struct __vec32_i64  : public vec16<int64_t> {
    __vec32_i64() { }
    __vec32_i64(int64_t v0, int64_t v1, int64_t v2, int64_t v3,
                int64_t v4, int64_t v5, int64_t v6, int64_t v7,
                int64_t v8, int64_t v9, int64_t v10, int64_t v11,
                int64_t v12, int64_t v13, int64_t v14, int64_t v15)
        : vec16<int64_t>(v0, v1, v2, v3, v4, v5, v6, v7,
                         v8, v9, v10, v11, v12, v13, v14, v15) { }
} POST_ALIGN(128);
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

static FORCEINLINE uint32_t __movmsk(__vec32_i1 mask) {
    uint32_t m1 = _mm512_kmov(mask.m16.m1);
    uint32_t m2 = _mm512_kmov(mask.m16.m2);
    return ((m1 << 16) | m2);
}

static FORCEINLINE bool __any(__vec32_i1 mask) { return !_mm512_kortestz(mask.m16.m1, mask.m16.m2); }

static FORCEINLINE bool __all(__vec32_i1 mask) { return (mask.m == 0xFFFFFFFF); }

static FORCEINLINE bool __none(__vec32_i1 mask) { return !__any(mask); }

static FORCEINLINE __vec32_i1 __equal(__vec32_i1 a, __vec32_i1 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_knot(_mm512_kandn(a.m16.m1, b.m16.m1));
    ret.m16.m2 = _mm512_knot(_mm512_kandn(a.m16.m2, b.m16.m2));
    return ret;
}

static FORCEINLINE __vec32_i1 __not(__vec32_i1 a) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_knot(a.m16.m1);
    ret.m16.m2 = _mm512_knot(a.m16.m2);
    return ret;
}

static FORCEINLINE __vec32_i1 __and(__vec32_i1 a, __vec32_i1 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_kand(a.m16.m1, b.m16.m1);
    ret.m16.m2 = _mm512_kand(a.m16.m2, b.m16.m2);
    return ret;
}

static FORCEINLINE __vec32_i1 __and_not1(__vec32_i1 a, __vec32_i1 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_kandn(a.m16.m1, b.m16.m1);
    ret.m16.m2 = _mm512_kandn(a.m16.m2, b.m16.m2);
    return ret;
}

static FORCEINLINE __vec32_i1 __and_not2(__vec32_i1 a, __vec32_i1 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_kandnr(a.m16.m1, b.m16.m1);
    ret.m16.m2 = _mm512_kandnr(a.m16.m2, b.m16.m2);
    return ret;
}

static FORCEINLINE __vec32_i1 __xor(__vec32_i1 a, __vec32_i1 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_kxor(a.m16.m1, b.m16.m1);
    ret.m16.m2 = _mm512_kxor(a.m16.m2, b.m16.m2);
    return ret;
}

static FORCEINLINE __vec32_i1 __xnor(__vec32_i1 a, __vec32_i1 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_kxnor(a.m16.m1, b.m16.m1);
    ret.m16.m2 = _mm512_kxnor(a.m16.m2, b.m16.m2);
    return ret;
}

static FORCEINLINE __vec32_i1 __or(__vec32_i1 a, __vec32_i1 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_kor(a.m16.m1, b.m16.m1);
    ret.m16.m2 = _mm512_kor(a.m16.m2, b.m16.m2);
    return ret;
}

static FORCEINLINE __vec32_i1 __select(__vec32_i1 mask, __vec32_i1 a, __vec32_i1 b) {
    return (a.m & mask.m) | (b.m & ~mask.m);
}

static FORCEINLINE __vec32_i1 __select(bool cond, __vec32_i1 a, __vec32_i1 b) { return cond ? a : b; }

/*
static FORCEINLINE bool __extract_element(__vec32_i1 vec, int index) {
    return (vec.v & (1 << index)) ? true : false;
}

static FORCEINLINE void __insert_element(__vec32_i1 *vec, int index,
                                         bool val) {
    if (val == false)
        vec->v &= ~(1 << index);
    else
        vec->v |= (1 << index);
}
*/

template <int ALIGN> static FORCEINLINE __vec32_i1 __load(const __vec32_i1 *p) {
    uint16_t *ptr = (uint16_t *)p;
    __vec32_i1 ret;
    ret.m16.m1 = *ptr;
    ptr++;
    ret.m16.m2 = *ptr;
    return ret;
}

template <int ALIGN> static FORCEINLINE void __store(__vec32_i1 *p, __vec32_i1 v) {
    uint16_t *ptr = (uint16_t *)p;
    *ptr = v.m16.m1;
    ptr++;
    *ptr = v.m16.m2;
}

template <> static FORCEINLINE __vec32_i1 __smear_i1<__vec32_i1>(int i) { return i ? 0xFFFF : 0x0; }

template <> static FORCEINLINE __vec32_i1 __setzero_i1<__vec32_i1>() { return 0; }

template <> static FORCEINLINE __vec32_i1 __undef_i1<__vec32_i1>() { return __vec32_i1(); }

///////////////////////////////////////////////////////////////////////////
// int8

BINARY_OP(__vec32_i8, __add, +)
BINARY_OP(__vec32_i8, __sub, -)
BINARY_OP(__vec32_i8, __mul, *)

BINARY_OP(__vec32_i8, __or, |)
BINARY_OP(__vec32_i8, __and, &)
BINARY_OP(__vec32_i8, __xor, ^)
BINARY_OP(__vec32_i8, __shl, <<)

BINARY_OP_CAST(__vec32_i8, uint8_t, __udiv, /)
BINARY_OP_CAST(__vec32_i8, int8_t, __sdiv, /)

BINARY_OP_CAST(__vec32_i8, uint8_t, __urem, %)
BINARY_OP_CAST(__vec32_i8, int8_t, __srem, %)
BINARY_OP_CAST(__vec32_i8, uint8_t, __lshr, >>)
BINARY_OP_CAST(__vec32_i8, int8_t, __ashr, >>)

SHIFT_UNIFORM(__vec32_i8, uint8_t, __lshr, >>)
SHIFT_UNIFORM(__vec32_i8, int8_t, __ashr, >>)
SHIFT_UNIFORM(__vec32_i8, int8_t, __shl, <<)

CMP_OP(__vec32_i8, i8, int8_t, __equal, ==)
CMP_OP(__vec32_i8, i8, int8_t, __not_equal, !=)
CMP_OP(__vec32_i8, i8, uint8_t, __unsigned_less_equal, <=)
CMP_OP(__vec32_i8, i8, int8_t, __signed_less_equal, <=)
CMP_OP(__vec32_i8, i8, uint8_t, __unsigned_greater_equal, >=)
CMP_OP(__vec32_i8, i8, int8_t, __signed_greater_equal, >=)
CMP_OP(__vec32_i8, i8, uint8_t, __unsigned_less_than, <)
CMP_OP(__vec32_i8, i8, int8_t, __signed_less_than, <)
CMP_OP(__vec32_i8, i8, uint8_t, __unsigned_greater_than, >)
CMP_OP(__vec32_i8, i8, int8_t, __signed_greater_than, >)

SELECT(__vec32_i8)
INSERT_EXTRACT(__vec32_i8, int8_t)
SMEAR(__vec32_i8, i8, int8_t)
BROADCAST(__vec32_i8, i8, int8_t)
ROTATE(__vec32_i8, i8, int8_t)
SHUFFLES(__vec32_i8, i8, int8_t)
LOAD_STORE(__vec32_i8, int8_t)

///////////////////////////////////////////////////////////////////////////
// int16

BINARY_OP(__vec32_i16, __add, +)
BINARY_OP(__vec32_i16, __sub, -)
BINARY_OP(__vec32_i16, __mul, *)

BINARY_OP(__vec32_i16, __or, |)
BINARY_OP(__vec32_i16, __and, &)
BINARY_OP(__vec32_i16, __xor, ^)
BINARY_OP(__vec32_i16, __shl, <<)

BINARY_OP_CAST(__vec32_i16, uint16_t, __udiv, /)
BINARY_OP_CAST(__vec32_i16, int16_t, __sdiv, /)

BINARY_OP_CAST(__vec32_i16, uint16_t, __urem, %)
BINARY_OP_CAST(__vec32_i16, int16_t, __srem, %)
BINARY_OP_CAST(__vec32_i16, uint16_t, __lshr, >>)
BINARY_OP_CAST(__vec32_i16, int16_t, __ashr, >>)

SHIFT_UNIFORM(__vec32_i16, uint16_t, __lshr, >>)
SHIFT_UNIFORM(__vec32_i16, int16_t, __ashr, >>)
SHIFT_UNIFORM(__vec32_i16, int16_t, __shl, <<)

CMP_OP(__vec32_i16, i16, int16_t, __equal, ==)
CMP_OP(__vec32_i16, i16, int16_t, __not_equal, !=)
CMP_OP(__vec32_i16, i16, uint16_t, __unsigned_less_equal, <=)
CMP_OP(__vec32_i16, i16, int16_t, __signed_less_equal, <=)
CMP_OP(__vec32_i16, i16, uint16_t, __unsigned_greater_equal, >=)
CMP_OP(__vec32_i16, i16, int16_t, __signed_greater_equal, >=)
CMP_OP(__vec32_i16, i16, uint16_t, __unsigned_less_than, <)
CMP_OP(__vec32_i16, i16, int16_t, __signed_less_than, <)
CMP_OP(__vec32_i16, i16, uint16_t, __unsigned_greater_than, >)
CMP_OP(__vec32_i16, i16, int16_t, __signed_greater_than, >)

SELECT(__vec32_i16)
INSERT_EXTRACT(__vec32_i16, int16_t)
SMEAR(__vec32_i16, i16, int16_t)
BROADCAST(__vec32_i16, i16, int16_t)
ROTATE(__vec32_i16, i16, int16_t)
SHUFFLES(__vec32_i16, i16, int16_t)
LOAD_STORE(__vec32_i16, int16_t)

///////////////////////////////////////////////////////////////////////////
// int32

static FORCEINLINE __vec32_i32 __add(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_add_epi32(a.v1, b.v1);
    ret.v2 = _mm512_add_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __sub(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_sub_epi32(a.v1, b.v1);
    ret.v2 = _mm512_sub_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __mul(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_mullo_epi32(a.v1, b.v1);
    ret.v2 = _mm512_mullo_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __udiv(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_div_epu32(a.v1, b.v1);
    ret.v2 = _mm512_div_epu32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __sdiv(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_div_epi32(a.v1, b.v1);
    ret.v2 = _mm512_div_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __urem(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_rem_epu32(a.v1, b.v1);
    ret.v2 = _mm512_rem_epu32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __srem(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_rem_epi32(a.v1, b.v1);
    ret.v2 = _mm512_rem_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __or(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_or_epi32(a.v1, b.v1);
    ret.v2 = _mm512_or_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __and(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_and_epi32(a.v1, b.v1);
    ret.v2 = _mm512_and_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __xor(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_xor_epi32(a.v1, b.v1);
    ret.v2 = _mm512_xor_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __shl(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_sllv_epi32(a.v1, b.v1);
    ret.v2 = _mm512_sllv_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __lshr(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_srlv_epi32(a.v1, b.v1);
    ret.v2 = _mm512_srlv_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __ashr(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_srav_epi32(a.v1, b.v1);
    ret.v2 = _mm512_srav_epi32(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __shl(__vec32_i32 a, int32_t n) {
    __vec32_i32 ret;
    ret.v1 = _mm512_slli_epi32(a.v1, n);
    ret.v2 = _mm512_slli_epi32(a.v2, n);
    return ret;
}

static FORCEINLINE __vec32_i32 __lshr(__vec32_i32 a, int32_t n) {
    __vec32_i32 ret;
    ret.v1 = _mm512_srli_epi32(a.v1, n);
    ret.v2 = _mm512_srli_epi32(a.v2, n);
    return ret;
}

static FORCEINLINE __vec32_i32 __ashr(__vec32_i32 a, int32_t n) {
    __vec32_i32 ret;
    ret.v1 = _mm512_srai_epi32(a.v1, n);
    ret.v2 = _mm512_srai_epi32(a.v2, n);
    return ret;
}

static FORCEINLINE __vec32_i1 __equal_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpeq_epi32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpeq_epi32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __equal_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpeq_epi32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpeq_epi32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __not_equal_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpneq_epi32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpneq_epi32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __not_equal_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpneq_epi32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpneq_epi32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __unsigned_less_equal_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmple_epu32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmple_epu32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __unsigned_less_equal_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmple_epu32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmple_epu32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __signed_less_equal_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmple_epi32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmple_epi32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __signed_less_equal_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmple_epi32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmple_epi32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __unsigned_greater_equal_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpge_epu32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpge_epu32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __unsigned_greater_equal_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpge_epu32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpge_epu32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __signed_greater_equal_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpge_epi32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpge_epi32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __signed_greater_equal_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpge_epi32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpge_epi32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __unsigned_less_than_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmplt_epu32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmplt_epu32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __unsigned_less_than_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmplt_epu32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmplt_epu32_mask(m.m16.m1, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __signed_less_than_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmplt_epi32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmplt_epi32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __signed_less_than_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmplt_epi32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmplt_epi32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __unsigned_greater_than_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpgt_epu32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpgt_epu32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __unsigned_greater_than_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpgt_epu32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpgt_epu32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __signed_greater_than_i32(__vec32_i32 a, __vec32_i32 b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpgt_epi32_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpgt_epi32_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __signed_greater_than_i32_and_mask(__vec32_i32 a, __vec32_i32 b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpgt_epi32_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpgt_epi32_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __select(__vec32_i1 mask, __vec32_i32 a, __vec32_i32 b) {
    __vec32_i32 ret;
    ret.v1 = _mm512_mask_mov_epi32(b.v1, mask.m16.m1, a.v1);
    ret.v2 = _mm512_mask_mov_epi32(b.v2, mask.m16.m2, a.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __select(bool cond, __vec32_i32 a, __vec32_i32 b) { return cond ? a : b; }

static FORCEINLINE int32_t __extract_element(__vec32_i32 v, int index) { return ((int32_t *)&v.v1)[index]; }

static FORCEINLINE void __insert_element(__vec32_i32 *v, int index, int32_t val) { ((int32_t *)v)[index] = val; }

template <> static FORCEINLINE __vec32_i32 __smear_i32<__vec32_i32>(int32_t i) {
    __vec32_i32 ret;
    ret.v1 = _mm512_set1_epi32(i);
    ret.v2 = _mm512_set1_epi32(i);
    return ret;
}

template <> static FORCEINLINE __vec32_i32 __setzero_i32<__vec32_i32>() {
    __vec32_i32 ret;
    ret.v1 = _mm512_setzero_epi32();
    ret.v2 = _mm512_setzero_epi32();
    return ret;
}

template <> static FORCEINLINE __vec32_i32 __undef_i32<__vec32_i32>() { return __vec32_i32(); }

static FORCEINLINE __vec32_i32 __broadcast_i32(__vec32_i32 v, int index) {
    __vec32_i32 ret;
    int32_t val = __extract_element(v, index & 0xf);
    ret.v1 = _mm512_set1_epi32(val);
    ret.v2 = _mm512_set1_epi32(val);
    return ret;
}

/*
static FORCEINLINE __vec32_i32 __rotate_i32(__vec32_i32 v, int index) {

    __vec32_i32 ret1 = v;
    __vec32_i32 ret2 = v;
    __vec32_i32 ret; // unaligned load from ((uint8_t*)&ret1)+index

    //for (int i = 0; i < 16; ++i) ret.v[i] = v.v[(i+index) & 0xf]; return ret;
}

static FORCEINLINE __vec32_i32 __shuffle_i32(__vec32_i32 v, __vec32_i32 index) {
    __vec32_i32 ret; for (int i = 0; i < 16; ++i) ret.v[i] = v.v[__extract_element(index, i) & 0xf]; return ret;
}

static FORCEINLINE __vec32_i32 __shuffle2_i32(__vec32_i32 v0, __vec32_i32 v1, __vec32_i32 index) {
    __vec32_i32 ret; for (int i = 0; i < 16; ++i) { int ii = __extract_element(index, i) & 0x1f; ret.v[i] = (ii < 16) ?
v0.v[ii] : v1.v[ii-16]; } return ret;
}
*/

template <int ALIGN> static FORCEINLINE __vec32_i32 __load(const __vec32_i32 *p) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    return __load<64>(p);
#else
    __vec32_i32 ret;
    ret.v1 = _mm512_extloadunpacklo_epi32(ret.v1, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    ret.v1 = _mm512_extloadunpackhi_epi32(ret.v1, (uint8_t *)p + 64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    ret.v2 = _mm512_extloadunpacklo_epi32(ret.v2, (uint8_t *)p + 64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    ret.v2 = _mm512_extloadunpackhi_epi32(ret.v2, (uint8_t *)p + 128, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    return ret;
#endif
}

template <> static FORCEINLINE __vec32_i32 __load<64>(const __vec32_i32 *p) {
    __vec32_i32 ret;
    ret.v1 = _mm512_load_epi32(p);
    ret.v2 = _mm512_load_epi32((uint8_t *)p + 64);
    return ret;
}

template <> static FORCEINLINE __vec32_i32 __load<128>(const __vec32_i32 *p) { return __load<64>(p); }

template <int ALIGN> static FORCEINLINE void __store(__vec32_i32 *p, __vec32_i32 v) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    __store<64>(p, v);
#else
    _mm512_extpackstorelo_epi32(p, v.v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_epi32((uint8_t *)p + 64, v.v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_epi32((uint8_t *)p + 64, v.v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_epi32((uint8_t *)p + 128, v.v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
#endif
}

template <> static FORCEINLINE void __store<64>(__vec32_i32 *p, __vec32_i32 v) {
    _mm512_store_epi32(p, v.v1);
    _mm512_store_epi32((uint8_t *)p + 64, v.v2);
}

template <> static FORCEINLINE void __store<128>(__vec32_i32 *p, __vec32_i32 v) { __store<64>(p, v); }

///////////////////////////////////////////////////////////////////////////
// int64

BINARY_OP(__vec32_i64, __add, +)
BINARY_OP(__vec32_i64, __sub, -)
BINARY_OP(__vec32_i64, __mul, *)

BINARY_OP(__vec32_i64, __or, |)
BINARY_OP(__vec32_i64, __and, &)
BINARY_OP(__vec32_i64, __xor, ^)
BINARY_OP(__vec32_i64, __shl, <<)

BINARY_OP_CAST(__vec32_i64, uint64_t, __udiv, /)
BINARY_OP_CAST(__vec32_i64, int64_t, __sdiv, /)

BINARY_OP_CAST(__vec32_i64, uint64_t, __urem, %)
BINARY_OP_CAST(__vec32_i64, int64_t, __srem, %)
BINARY_OP_CAST(__vec32_i64, uint64_t, __lshr, >>)
BINARY_OP_CAST(__vec32_i64, int64_t, __ashr, >>)

SHIFT_UNIFORM(__vec32_i64, uint64_t, __lshr, >>)
SHIFT_UNIFORM(__vec32_i64, int64_t, __ashr, >>)
SHIFT_UNIFORM(__vec32_i64, int64_t, __shl, <<)

CMP_OP(__vec32_i64, i64, int64_t, __equal, ==)
CMP_OP(__vec32_i64, i64, int64_t, __not_equal, !=)
CMP_OP(__vec32_i64, i64, uint64_t, __unsigned_less_equal, <=)
CMP_OP(__vec32_i64, i64, int64_t, __signed_less_equal, <=)
CMP_OP(__vec32_i64, i64, uint64_t, __unsigned_greater_equal, >=)
CMP_OP(__vec32_i64, i64, int64_t, __signed_greater_equal, >=)
CMP_OP(__vec32_i64, i64, uint64_t, __unsigned_less_than, <)
CMP_OP(__vec32_i64, i64, int64_t, __signed_less_than, <)
CMP_OP(__vec32_i64, i64, uint64_t, __unsigned_greater_than, >)
CMP_OP(__vec32_i64, i64, int64_t, __signed_greater_than, >)

SELECT(__vec32_i64)
INSERT_EXTRACT(__vec32_i64, int64_t)
SMEAR(__vec32_i64, i64, int64_t)
BROADCAST(__vec32_i64, i64, int64_t)
ROTATE(__vec32_i64, i64, int64_t)
SHUFFLES(__vec32_i64, i64, int64_t)
LOAD_STORE(__vec32_i64, int64_t)

///////////////////////////////////////////////////////////////////////////
// float

static FORCEINLINE __vec32_f __add(__vec32_f a, __vec32_f b) {
    __vec32_f ret;
    ret.v1 = _mm512_add_ps(a.v1, b.v1);
    ret.v2 = _mm512_add_ps(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_f __sub(__vec32_f a, __vec32_f b) {
    __vec32_f ret;
    ret.v1 = _mm512_sub_ps(a.v1, b.v1);
    ret.v2 = _mm512_sub_ps(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_f __mul(__vec32_f a, __vec32_f b) {
    __vec32_f ret;
    ret.v1 = _mm512_mul_ps(a.v1, b.v1);
    ret.v2 = _mm512_mul_ps(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_f __div(__vec32_f a, __vec32_f b) {
    __vec32_f ret;
    /*
        __vec32_f rcpb;
        rcpb.v1 = _mm512_rcp23_ps(b.v1);
        rcpb.v2 = _mm512_rcp23_ps(b.v2);
        ret.v1 = _mm512_mul_ps(a.v1, rcpb.v1);
        ret.v2 = _mm512_mul_ps(a.v2, rcpb.v2);
    */
    ret.v1 = _mm512_div_ps(a.v1, b.v1);
    ret.v2 = _mm512_div_ps(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __equal_float(__vec32_f a, __vec32_f b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpeq_ps_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpeq_ps_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __equal_float_and_mask(__vec32_f a, __vec32_f b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpeq_ps_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpeq_ps_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __not_equal_float(__vec32_f a, __vec32_f b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpneq_ps_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpneq_ps_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __not_equal_float_and_mask(__vec32_f a, __vec32_f b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpneq_ps_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpneq_ps_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __less_than_float(__vec32_f a, __vec32_f b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmplt_ps_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmplt_ps_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __less_than_float_and_mask(__vec32_f a, __vec32_f b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmplt_ps_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmplt_ps_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __less_equal_float(__vec32_f a, __vec32_f b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmple_ps_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmple_ps_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __less_equal_float_and_mask(__vec32_f a, __vec32_f b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmple_ps_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmple_ps_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __greater_than_float(__vec32_f a, __vec32_f b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpnle_ps_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpnle_ps_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __greater_than_float_and_mask(__vec32_f a, __vec32_f b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpnle_ps_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpnle_ps_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __greater_equal_float(__vec32_f a, __vec32_f b) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_cmpnlt_ps_mask(a.v1, b.v1);
    ret.m16.m2 = _mm512_cmpnlt_ps_mask(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_i1 __greater_equal_float_and_mask(__vec32_f a, __vec32_f b, __vec32_i1 m) {
    __vec32_i1 ret;
    ret.m16.m1 = _mm512_mask_cmpnlt_ps_mask(m.m16.m1, a.v1, b.v1);
    ret.m16.m2 = _mm512_mask_cmpnlt_ps_mask(m.m16.m2, a.v2, b.v2);
    return ret;
}

/*
static FORCEINLINE __vec32_i1 __ordered(__vec32_f a, __vec32_f b) {
    __vec32_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 16; ++i)
        ret.v |= ((a.v[i] == a.v[i]) && (b.v[i] == b.v[i])) ? (1 << i) : 0;
    return ret;
}
*/

static FORCEINLINE __vec32_f __select(__vec32_i1 mask, __vec32_f a, __vec32_f b) {
    __vec32_f ret;
    ret.v1 = _mm512_mask_mov_ps(b.v1, mask.m16.m1, a.v1);
    ret.v2 = _mm512_mask_mov_ps(b.v2, mask.m16.m2, a.v2);
    return ret;
}

static FORCEINLINE __vec32_f __select(bool cond, __vec32_f a, __vec32_f b) { return cond ? a : b; }

static FORCEINLINE float __extract_element(__vec32_f v, int index) { return ((float *)&v.v1)[index]; }

static FORCEINLINE void __insert_element(__vec32_f *v, int index, float val) { ((float *)v)[index] = val; }

template <> static FORCEINLINE __vec32_f __smear_float<__vec32_f>(float f) {
    __vec32_f ret;
    ret.v1 = _mm512_extload_ps(&f, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
    ret.v2 = ret.v1;
    return ret;
}

template <> static FORCEINLINE __vec32_f __setzero_float<__vec32_f>() {
    __vec32_f ret;
    ret.v1 = _mm512_setzero_ps();
    ret.v2 = ret.v1;
    return ret;
}

template <> static FORCEINLINE __vec32_f __undef_float<__vec32_f>() { return __vec32_f(); }

static FORCEINLINE __vec32_f __broadcast_float(__vec32_f v, int index) {
    __vec32_f ret;
    int32_t val = __extract_element(v, index & 0xf);
    ret.v1 = _mm512_extload_ps(&val, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
    ret.v2 = ret.v1;
    return ret;
}

/*
static FORCEINLINE __vec32_f __rotate_float(__vec32_f v, int index) {
    __vec32_f ret; for (int i = 0; i < 16; ++i) ret.v[i] = v.v[(i+index) & 0xf]; return ret;
}
*/

static FORCEINLINE __vec32_f __shuffle_float(__vec32_f v, __vec32_i32 index) {
    __vec32_f ret;
    ret.v1 = _mm512_castsi512_ps(
        _mm512_mask_permutevar_epi32(_mm512_castps_si512(v.v1), 0xffff, index.v1, _mm512_castps_si512(v.v1)));
    ret.v2 = _mm512_castsi512_ps(
        _mm512_mask_permutevar_epi32(_mm512_castps_si512(v.v2), 0xffff, index.v2, _mm512_castps_si512(v.v2)));
    return ret;
}

/*
static FORCEINLINE __vec32_f __shuffle2_float(__vec32_f v0, __vec32_f v1, __vec32_i32 index) {
    __vec32_f ret; for (int i = 0; i < 16; ++i) { int ii = __extract_element(index, i) & 0x1f; ret.v[i] = (ii < 16) ?
v0.v[ii] : v1.v[ii-16]; } return ret;
}
*/

template <int ALIGN> static FORCEINLINE __vec32_f __load(const __vec32_f *p) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    return __load<64>(p);
#else
    __vec32_f ret;
    ret.v1 = _mm512_extloadunpacklo_ps(ret.v1, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    ret.v1 = _mm512_extloadunpackhi_ps(ret.v1, (uint8_t *)p + 64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    ret.v2 = _mm512_extloadunpacklo_ps(ret.v2, (uint8_t *)p + 64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    ret.v2 = _mm512_extloadunpackhi_ps(ret.v2, (uint8_t *)p + 128, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    return ret;
#endif
}

template <> static FORCEINLINE __vec32_f __load<64>(const __vec32_f *p) {
    __vec32_f ret;
    ret.v1 = _mm512_load_ps(p);
    ret.v2 = _mm512_load_ps((uint8_t *)p + 64);
    return ret;
}

template <> static FORCEINLINE __vec32_f __load<128>(const __vec32_f *p) { return __load<64>(p); }

template <int ALIGN> static FORCEINLINE void __store(__vec32_f *p, __vec32_f v) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    __store<64>(p, v);
#else
    _mm512_extpackstorelo_ps(p, v.v1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_ps((uint8_t *)p + 64, v.v1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_ps((uint8_t *)p + 64, v.v2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_ps((uint8_t *)p + 128, v.v2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
}

template <> static FORCEINLINE void __store<64>(__vec32_f *p, __vec32_f v) {
    _mm512_store_ps((uint8_t *)p, v.v1);
    _mm512_store_ps((uint8_t *)p + 64, v.v2);
}

template <> static FORCEINLINE void __store<128>(__vec32_f *p, __vec32_f v) { __store<64>(p, v); }

///////////////////////////////////////////////////////////////////////////
// double

BINARY_OP(__vec32_d, __add, +)
BINARY_OP(__vec32_d, __sub, -)
BINARY_OP(__vec32_d, __mul, *)
BINARY_OP(__vec32_d, __div, /)

CMP_OP(__vec32_d, d, double, __equal, ==)
CMP_OP(__vec32_d, d, double, __not_equal, !=)
CMP_OP(__vec32_d, d, double, __less_than, <)
CMP_OP(__vec32_d, d, double, __less_equal, <=)
CMP_OP(__vec32_d, d, double, __greater_than, >)
CMP_OP(__vec32_d, d, double, __greater_equal, >=)

/*
static FORCEINLINE __vec32_i1 __ordered(__vec32_d a, __vec32_d b) {
    __vec32_i1 ret;
    ret.v = 0;
    for (int i = 0; i < 16; ++i)
        ret.v |= ((a.v[i] == a.v[i]) && (b.v[i] == b.v[i])) ? (1 << i) : 0;
    return ret;
}
*/

#if 0
      case Instruction::FRem: intrinsic = "__frem"; break;
#endif

SELECT(__vec32_d)
INSERT_EXTRACT(__vec32_d, double)
SMEAR(__vec32_d, double, double)
BROADCAST(__vec32_d, double, double)
ROTATE(__vec32_d, double, double)
SHUFFLES(__vec32_d, double, double)
LOAD_STORE(__vec32_d, double)

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
CAST(__vec32_i64, int64_t, __vec32_i32, int32_t, __cast_sext)
CAST(__vec32_i64, int64_t, __vec32_i16, int16_t, __cast_sext)
CAST(__vec32_i64, int64_t, __vec32_i8, int8_t, __cast_sext)
CAST(__vec32_i32, int32_t, __vec32_i16, int16_t, __cast_sext)
CAST(__vec32_i32, int32_t, __vec32_i8, int8_t, __cast_sext)
CAST(__vec32_i16, int16_t, __vec32_i8, int8_t, __cast_sext)

#define CAST_SEXT_I1(TYPE)
/*
static FORCEINLINE TYPE __cast_sext(TYPE, __vec32_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 16; ++i) {                    \
        ret.v[i] = 0;                                 \
        if (v.v & (1 << i))                           \
            ret.v[i] = ~ret.v[i];                     \
    }                                                 \
    return ret;                                       \
}
*/
CAST_SEXT_I1(__vec32_i8)
CAST_SEXT_I1(__vec32_i16)
CAST_SEXT_I1(__vec32_i32)
CAST_SEXT_I1(__vec32_i64)

// zero extension
CAST(__vec32_i64, uint64_t, __vec32_i32, uint32_t, __cast_zext)
CAST(__vec32_i64, uint64_t, __vec32_i16, uint16_t, __cast_zext)
CAST(__vec32_i64, uint64_t, __vec32_i8, uint8_t, __cast_zext)
CAST(__vec32_i32, uint32_t, __vec32_i16, uint16_t, __cast_zext)
CAST(__vec32_i32, uint32_t, __vec32_i8, uint8_t, __cast_zext)
CAST(__vec32_i16, uint16_t, __vec32_i8, uint8_t, __cast_zext)

#define CAST_ZEXT_I1(TYPE)
/*
static FORCEINLINE TYPE __cast_zext(TYPE, __vec32_i1 v) {  \
    TYPE ret;                                         \
    for (int i = 0; i < 16; ++i)                      \
        ret.v[i] = (v.v & (1 << i)) ? 1 : 0;          \
    return ret;                                       \
}
*/
CAST_ZEXT_I1(__vec32_i8)
CAST_ZEXT_I1(__vec32_i16)
CAST_ZEXT_I1(__vec32_i32)
CAST_ZEXT_I1(__vec32_i64)

// truncations
CAST(__vec32_i32, int32_t, __vec32_i64, int64_t, __cast_trunc)
CAST(__vec32_i16, int16_t, __vec32_i64, int64_t, __cast_trunc)
CAST(__vec32_i8, int8_t, __vec32_i64, int64_t, __cast_trunc)
CAST(__vec32_i16, int16_t, __vec32_i32, int32_t, __cast_trunc)
CAST(__vec32_i8, int8_t, __vec32_i32, int32_t, __cast_trunc)
CAST(__vec32_i8, int8_t, __vec32_i16, int16_t, __cast_trunc)

// signed int to float/double
CAST(__vec32_f, float, __vec32_i8, int8_t, __cast_sitofp)
CAST(__vec32_f, float, __vec32_i16, int16_t, __cast_sitofp)
CAST(__vec32_f, float, __vec32_i32, int32_t, __cast_sitofp)
CAST(__vec32_f, float, __vec32_i64, int64_t, __cast_sitofp)
CAST(__vec32_d, double, __vec32_i8, int8_t, __cast_sitofp)
CAST(__vec32_d, double, __vec32_i16, int16_t, __cast_sitofp)
CAST(__vec32_d, double, __vec32_i32, int32_t, __cast_sitofp)
CAST(__vec32_d, double, __vec32_i64, int64_t, __cast_sitofp)

static FORCEINLINE __vec32_f __cast_sitofp(__vec32_f, __vec32_i8 val) {
    __vec32_f ret;
    ret.v1 = _mm512_extload_ps(&val, _MM_UPCONV_PS_SINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
    ret.v2 = _mm512_extload_ps((uint8_t *)&val + 16, _MM_UPCONV_PS_SINT8, _MM_BROADCAST_16X16, _MM_HINT_NONE);
    return ret;
}

static FORCEINLINE __vec32_f __cast_sitofp(__vec32_f, __vec32_i16 val) {
    __vec32_f ret;
    ret.v1 = _mm512_extload_ps(&val, _MM_UPCONV_PS_SINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
    ret.v2 = _mm512_extload_ps((uint16_t *)&val + 32, _MM_UPCONV_PS_SINT16, _MM_BROADCAST_16X16, _MM_HINT_NONE);
    return ret;
}

static FORCEINLINE __vec32_f __cast_sitofp(__vec32_f, __vec32_i32 val) {
    __vec32_f ret;
    ret.v1 = _mm512_cvtfxpnt_round_adjustepi32_ps(val.v1, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);
    ret.v2 = _mm512_cvtfxpnt_round_adjustepi32_ps(val.v2, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);
    return ret;
}

/*
static FORCEINLINE __vec32_f __cast_sitofp(__vec32_f, __vec32_i64 val) {
    __vec32_f ret; for (int i = 0; i < 16; ++i) ret.v[i] = (float)((int64_t)(val.v[i])); return ret;
}

static FORCEINLINE __vec32_d __cast_sitofp(__vec32_d, __vec32_i8 val) {
    __vec32_d ret; for (int i = 0; i < 16; ++i) ret.v[i] = (double)((int8_t)(val.v[i])); return ret;
}

static FORCEINLINE __vec32_d __cast_sitofp(__vec32_d, __vec32_i16 val) {
    __vec32_d ret; for (int i = 0; i < 16; ++i) ret.v[i] = (double)((int16_t)(val.v[i])); return ret;
}

static FORCEINLINE __vec32_d __cast_sitofp(__vec32_d, __vec32_i32 val) {
    __vec32_d ret; for (int i = 0; i < 16; ++i) ret.v[i] = (double)((int32_t)(val.v[i])); return ret;
}

static FORCEINLINE __vec32_d __cast_sitofp(__vec32_d, __vec32_i64 val) {
    __vec32_d ret; for (int i = 0; i < 16; ++i) ret.v[i] = (double)((int64_t)(val.v[i])); return ret;
}
*/

// unsigned int to float/double
CAST(__vec32_f, float, __vec32_i8, uint8_t, __cast_uitofp)
CAST(__vec32_f, float, __vec32_i16, uint16_t, __cast_uitofp)
CAST(__vec32_f, float, __vec32_i32, uint32_t, __cast_uitofp)
CAST(__vec32_f, float, __vec32_i64, uint64_t, __cast_uitofp)
CAST(__vec32_d, double, __vec32_i8, uint8_t, __cast_uitofp)
CAST(__vec32_d, double, __vec32_i16, uint16_t, __cast_uitofp)
CAST(__vec32_d, double, __vec32_i32, uint32_t, __cast_uitofp)
CAST(__vec32_d, double, __vec32_i64, uint64_t, __cast_uitofp)
/*
static FORCEINLINE __vec32_f __cast_uitofp(__vec32_f, __vec32_i1 v) {
    __vec32_f ret;
    for (int i = 0; i < 16; ++i)
        ret.v[i] = (v.v & (1 << i)) ? 1. : 0.;
    return ret;
}
*/
// float/double to signed int

static FORCEINLINE __vec32_i32 __cast_fptosi(__vec32_i32, __vec32_f val) {
    __vec32_i32 ret;
    ret.v1 = _mm512_cvtfxpnt_round_adjustps_epi32(val.v1, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
    ret.v2 = _mm512_cvtfxpnt_round_adjustps_epi32(val.v2, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
    return ret;
}

CAST(__vec32_i8, int8_t, __vec32_f, float, __cast_fptosi)
CAST(__vec32_i16, int16_t, __vec32_f, float, __cast_fptosi)
CAST(__vec32_i32, int32_t, __vec32_f, float, __cast_fptosi)
CAST(__vec32_i64, int64_t, __vec32_f, float, __cast_fptosi)
CAST(__vec32_i8, int8_t, __vec32_d, double, __cast_fptosi)
CAST(__vec32_i16, int16_t, __vec32_d, double, __cast_fptosi)
CAST(__vec32_i32, int32_t, __vec32_d, double, __cast_fptosi)
CAST(__vec32_i64, int64_t, __vec32_d, double, __cast_fptosi)

// float/double to unsigned int
CAST(__vec32_i8, uint8_t, __vec32_f, float, __cast_fptoui)
CAST(__vec32_i16, uint16_t, __vec32_f, float, __cast_fptoui)
CAST(__vec32_i32, uint32_t, __vec32_f, float, __cast_fptoui)
CAST(__vec32_i64, uint64_t, __vec32_f, float, __cast_fptoui)
CAST(__vec32_i8, uint8_t, __vec32_d, double, __cast_fptoui)
CAST(__vec32_i16, uint16_t, __vec32_d, double, __cast_fptoui)
CAST(__vec32_i32, uint32_t, __vec32_d, double, __cast_fptoui)
CAST(__vec32_i64, uint64_t, __vec32_d, double, __cast_fptoui)

// float/double conversions
CAST(__vec32_f, float, __vec32_d, double, __cast_fptrunc)
CAST(__vec32_d, double, __vec32_f, float, __cast_fpext)

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

static FORCEINLINE __vec32_f __cast_bits(__vec32_f, __vec32_i32 val) {
    // TODO: This should be doable without the reads...
    __vec32_f ret;
    ret.v1 = _mm512_castsi512_ps(val.v1);
    ret.v2 = _mm512_castsi512_ps(val.v2);
    return ret;
}

static FORCEINLINE __vec32_i32 __cast_bits(__vec32_i32, __vec32_f val) {
    // TODO: This should be doable without the reads...
    __vec32_i32 ret;
    ret.v1 = _mm512_castps_si512(val.v1);
    ret.v2 = _mm512_castps_si512(val.v2);
    return ret;
}

CAST_BITS(__vec32_d, double, d, __vec32_i64, i64)
CAST_BITS(__vec32_i64, int64_t, i64, __vec32_d, d)

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

static FORCEINLINE __vec32_f __round_varying_float(__vec32_f v) {
    __vec32_f ret;
    ret.v1 = _mm512_round_ps(v.v1, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);
    ret.v2 = _mm512_round_ps(v.v2, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);
    return ret;
}

static FORCEINLINE __vec32_f __floor_varying_float(__vec32_f v) {
    __vec32_f ret;
    ret.v1 = _mm512_floor_ps(v.v1);
    ret.v2 = _mm512_floor_ps(v.v2);
    return ret;
}

static FORCEINLINE __vec32_f __ceil_varying_float(__vec32_f v) {
    __vec32_f ret;
    ret.v1 = _mm512_ceil_ps(v.v1);
    ret.v2 = _mm512_ceil_ps(v.v2);
    return ret;
}

UNARY_OP(__vec32_d, __round_varying_double, round)
UNARY_OP(__vec32_d, __floor_varying_double, floor)
UNARY_OP(__vec32_d, __ceil_varying_double, ceil)

// min/max

static FORCEINLINE __vec32_f __max_varying_float(__vec32_f a, __vec32_f b) {
    __vec32_f ret;
    ret.v1 = _mm512_max_ps(a.v1, b.v1);
    ret.v2 = _mm512_max_ps(a.v2, b.v2);
    return ret;
}

static FORCEINLINE __vec32_f __min_varying_float(__vec32_f a, __vec32_f b) {
    __vec32_f ret;
    ret.v1 = _mm512_min_ps(a.v1, a.v1);
    ret.v2 = _mm512_min_ps(a.v2, b.v2);
    return ret;
}

BINARY_OP_FUNC(__vec32_d, __max_varying_double, __max_uniform_double)
BINARY_OP_FUNC(__vec32_d, __min_varying_double, __min_uniform_double)

BINARY_OP_FUNC(__vec32_i32, __max_varying_int32, __max_uniform_int32)
BINARY_OP_FUNC(__vec32_i32, __min_varying_int32, __min_uniform_int32)
BINARY_OP_FUNC(__vec32_i32, __max_varying_uint32, __max_uniform_uint32)
BINARY_OP_FUNC(__vec32_i32, __min_varying_uint32, __min_uniform_uint32)

BINARY_OP_FUNC(__vec32_i64, __max_varying_int64, __max_uniform_int64)
BINARY_OP_FUNC(__vec32_i64, __min_varying_int64, __min_uniform_int64)
BINARY_OP_FUNC(__vec32_i64, __max_varying_uint64, __max_uniform_uint64)
BINARY_OP_FUNC(__vec32_i64, __min_varying_uint64, __min_uniform_uint64)

// sqrt/rsqrt/rcp

static FORCEINLINE __vec32_f __sqrt_varying_float(__vec32_f v) {
    __vec32_f ret;
    ret.v1 = _mm512_sqrt_ps(v.v1);
    ret.v2 = _mm512_sqrt_ps(v.v2);
    return ret;
}

static FORCEINLINE __vec32_f __rcp_varying_float(__vec32_f v) {
    __vec32_f ret;
    ret.v1 = _mm512_recip_ps(v.v1);
    ret.v2 = _mm512_recip_ps(v.v2);
    return ret;
    // return _mm512_rcp23_ps(v); // 23-bits of accuracy
}

static FORCEINLINE __vec32_f __rsqrt_varying_float(__vec32_f v) {
    __vec32_f ret;
    ret.v1 = _mm512_rsqrt23_ps(v.v1); // to 0.775ULP accuracy
    ret.v2 = _mm512_rsqrt23_ps(v.v2); // to 0.775ULP accuracy
    return ret;
}

static FORCEINLINE __vec32_f __exp_varying_float(__vec32_f v) {
    __vec32_f ret;
    ret.v1 = _mm512_exp_ps(v.v1);
    ret.v2 = _mm512_exp_ps(v.v2);
    return ret;
}

static FORCEINLINE __vec32_f __log_varying_float(__vec32_f v) {
    __vec32_f ret;
    ret.v1 = _mm512_log_ps(v.v1);
    ret.v2 = _mm512_log_ps(v.v2);
    return ret;
}

static FORCEINLINE __vec32_f __pow_varying_float(__vec32_f a, __vec32_f b) {
    __vec32_f ret;
    ret.v1 = _mm512_pow_ps(a.v1, b.v1);
    ret.v2 = _mm512_pow_ps(a.v2, b.v2);
    return ret;
}

UNARY_OP(__vec32_f, __rcp_varying_float, __rcp_uniform_float)
UNARY_OP(__vec32_d, __sqrt_varying_double, __sqrt_uniform_double)

///////////////////////////////////////////////////////////////////////////
// bit ops
/*
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
*/
///////////////////////////////////////////////////////////////////////////
// reductions

REDUCE_ADD(int16_t, __vec32_i8, __reduce_add_int8)
REDUCE_ADD(int32_t, __vec32_i16, __reduce_add_int16)

static FORCEINLINE float __reduce_add_float(__vec32_f v) {
    return _mm512_reduce_add_ps(v.v1) + _mm512_reduce_add_ps(v.v2);
}

REDUCE_MINMAX(float, __vec32_f, __reduce_min_float, <)
REDUCE_MINMAX(float, __vec32_f, __reduce_max_float, >)

REDUCE_ADD(double, __vec32_d, __reduce_add_double)
REDUCE_MINMAX(double, __vec32_d, __reduce_min_double, <)
REDUCE_MINMAX(double, __vec32_d, __reduce_max_double, >)

REDUCE_ADD(uint32_t, __vec32_i32, __reduce_add_int32)
REDUCE_MINMAX(int32_t, __vec32_i32, __reduce_min_int32, <)
REDUCE_MINMAX(int32_t, __vec32_i32, __reduce_max_int32, >)

REDUCE_ADD(uint32_t, __vec32_i32, __reduce_add_uint32)
REDUCE_MINMAX(uint32_t, __vec32_i32, __reduce_min_uint32, <)
REDUCE_MINMAX(uint32_t, __vec32_i32, __reduce_max_uint32, >)

REDUCE_ADD(uint64_t, __vec32_i64, __reduce_add_int64)
REDUCE_MINMAX(int64_t, __vec32_i64, __reduce_min_int64, <)
REDUCE_MINMAX(int64_t, __vec32_i64, __reduce_max_int64, >)

REDUCE_ADD(uint64_t, __vec32_i64, __reduce_add_uint64)
REDUCE_MINMAX(uint64_t, __vec32_i64, __reduce_min_uint64, <)
REDUCE_MINMAX(uint64_t, __vec32_i64, __reduce_max_uint64, >)

///////////////////////////////////////////////////////////////////////////
// masked load/store
/*
static FORCEINLINE __vec32_i8 __masked_load_i8(void *p,
                                              __vec32_i1 mask) {
    __vec32_i8 ret;
    int8_t *ptr = (int8_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret.v[i] = ptr[i];
    return ret;
}

static FORCEINLINE __vec32_i16 __masked_load_i16(void *p,
                                                __vec32_i1 mask) {
    __vec32_i16 ret;
    int16_t *ptr = (int16_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ret.v[i] = ptr[i];
    return ret;
}
*/

static FORCEINLINE __vec32_i32 __masked_load_i32(void *p, __vec32_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    __vec32_i32 ret;
    ret.v1 = _mm512_mask_load_epi32(ret.v1, mask.m16.m1, p);
    ret.v2 = _mm512_mask_load_epi32(ret.v2, mask.m16.m2, (uint8_t *)p + 64);
    return ret;
#else
    __vec32_i32 tmp;
    tmp.v1 = _mm512_mask_extloadunpacklo_epi32(tmp.v1, 0xFFFF, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp.v1 = _mm512_mask_extloadunpackhi_epi32(tmp.v1, 0xFFFF, (uint8_t *)p + 64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp.v2 = _mm512_mask_extloadunpacklo_epi32(tmp.v2, 0xFFFF, (uint8_t *)p + 64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp.v2 =
        _mm512_mask_extloadunpackhi_epi32(tmp.v2, 0xFFFF, (uint8_t *)p + 128, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    __vec32_i32 ret;
    ret.v1 = _mm512_mask_mov_epi32(ret.v1, mask.m16.m1, tmp.v1);
    ret.vs = _mm512_mask_mov_epi32(ret.v2, mask.m16.m2, tmp.v2);
    return ret;
#endif
}

/*
static FORCEINLINE __vec32_i64 __masked_load_i64(void *p,
                                                __vec32_i1 mask) {
    union {
        __vec32_i64 v64;
        __vec32_i32 v32[2];
    } ret;

    ret.v32[0] = _mm512_mask_loadq(ret, mask, p, _MM_FULLUPC64_NONE, _MM_BROADCAST_8X8, _MM_HINT_NONE);
    ret.v32[1] = _mm512_mask_loadq(ret, mask, p, _MM_FULLUPC64_NONE, _MM_BROADCAST_8X8, _MM_HINT_NONE);

    return ret.v64;
}
*/

static FORCEINLINE __vec32_f __masked_load_float(void *p, __vec32_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    __vec32_f ret;
    ret.v1 = _mm512_mask_load_ps(ret.v1, mask.m16.m1, p);
    ret.v2 = _mm512_mask_load_ps(ret.v2, mask.m16.m2, (uint8_t *)p + 64);
    return ret;
#else
    __vec32_f tmp;
    tmp.v1 = _mm512_mask_extloadunpacklo_ps(tmp.v1, 0xFFFF, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp.v1 = _mm512_mask_extloadunpackhi_ps(tmp.v1, 0xFFFF, (uint8_t *)p + 64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp.v2 = _mm512_mask_extloadunpacklo_ps(tmp.v2, 0xFFFF, (uint8_t *)p + 64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp.v2 = _mm512_mask_extloadunpackhi_ps(tmp.v2, 0xFFFF, (uint8_t *)p + 128, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    __vec32_f ret;
    ret.v1 = _mm512_mask_mov_ps(ret.v1, mask.m16.m1, tmp.v1);
    ret.v2 = _mm512_mask_mov_ps(ret.v2, mask.m16.m2, tmp.v2);
    return ret;
#endif
}

/*
static FORCEINLINE void __masked_store_i8(void *p, __vec32_i8 val,
                                         __vec32_i1 mask) {
    int8_t *ptr = (int8_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
}

static FORCEINLINE void __masked_store_i16(void *p, __vec32_i16 val,
                                          __vec32_i1 mask) {
    int16_t *ptr = (int16_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
}
*/

static FORCEINLINE void __masked_store_i32(void *p, __vec32_i32 val, __vec32_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    _mm512_mask_store_epi32((uint8_t *)p, mask.m16.m1, val.v1);
    _mm512_mask_store_epi32((uint8_t *)p + 64, mask.m16.m2, val.v2);
#else
    __vec32_i32 tmp;
    tmp.v1 = _mm512_extloadunpacklo_epi32(tmp.v1, p, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp.v1 = _mm512_extloadunpackhi_epi32(tmp.v1, (uint8_t *)p + 64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp.v2 = _mm512_extloadunpacklo_epi32(tmp.v2, (uint8_t *)p + 64, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp.v2 = _mm512_extloadunpackhi_epi32(tmp.v2, (uint8_t *)p + 128, _MM_UPCONV_EPI32_NONE, _MM_HINT_NONE);
    tmp.v1 = _mm512_mask_mov_epi32(tmp.v1, mask.m16.m1, val.v1);
    tmp.v2 = _mm512_mask_mov_epi32(tmp.v2, mask.m16.m2, val.v2);
    _mm512_extpackstorelo_epi32(p, tmp.v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_epi32((uint8_t *)p + 64, tmp.v1, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_epi32((uint8_t *)p + 64, tmp.v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_epi32((uint8_t *)p + 128, tmp.v2, _MM_DOWNCONV_EPI32_NONE, _MM_HINT_NONE);
#endif
}

/*
static FORCEINLINE void __masked_store_i64(void *p, __vec32_i64 val,
                                          __vec32_i1 mask) {
    int64_t *ptr = (int64_t *)p;
    for (int i = 0; i < 16; ++i)
        if ((mask.v & (1 << i)) != 0)
            ptr[i] = val.v[i];
}
*/

static FORCEINLINE void __masked_store_float(void *p, __vec32_f val, __vec32_i1 mask) {
#ifdef ISPC_FORCE_ALIGNED_MEMORY
    _mm512_mask_store_ps(p, mask.m16.m1, val.v1);
    _mm512_mask_store_ps((uint8_t *)p + 64, mask.m16.m2, val.v2);
#else
    __vec32_f tmp;
    tmp.v1 = _mm512_extloadunpacklo_ps(tmp.v1, p, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp.v1 = _mm512_extloadunpackhi_ps(tmp.v1, (uint8_t *)p + 64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp.v2 = _mm512_extloadunpacklo_ps(tmp.v2, (uint8_t *)p + 64, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp.v2 = _mm512_extloadunpackhi_ps(tmp.v2, (uint8_t *)p + 128, _MM_UPCONV_PS_NONE, _MM_HINT_NONE);
    tmp.v1 = _mm512_mask_mov_ps(tmp.v1, mask.m16.m1, val.v1);
    tmp.v2 = _mm512_mask_mov_ps(tmp.v2, mask.m16.m2, val.v2);
    _mm512_extpackstorelo_ps(p, tmp.v1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_ps((uint8_t *)p + 64, tmp.v1, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_extpackstorelo_ps((uint8_t *)p + 64, tmp.v2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
    _mm512_extpackstorehi_ps((uint8_t *)p + 128, tmp.v2, _MM_DOWNCONV_PS_NONE, _MM_HINT_NONE);
#endif
}

/*
static FORCEINLINE void __masked_store_blend_i8(void *p, __vec32_i8 val,
                                               __vec32_i1 mask) {
    __masked_store_i8(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_i16(void *p, __vec32_i16 val,
                                                __vec32_i1 mask) {
    __masked_store_i16(p, val, mask);
}
*/

static FORCEINLINE void __masked_store_blend_i32(void *p, __vec32_i32 val, __vec32_i1 mask) {
    __masked_store_i32(p, val, mask);
}

/*
static FORCEINLINE void __masked_store_blend_i64(void *p, __vec32_i64 val,
                                                __vec32_i1 mask) {
    __masked_store_i64(p, val, mask);
}
*/
///////////////////////////////////////////////////////////////////////////
// gather/scatter

// offsets * offsetScale is in bytes (for all of these)

#define GATHER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)
/*
static FORCEINLINE VTYPE FUNC(unsigned char *b, OTYPE varyingOffset,    \
                              uint32_t scale, OTYPE constOffset, \
                              __vec32_i1 mask) {                        \
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

GATHER_BASE_OFFSETS(__vec32_i8, int8_t, __vec32_i32, __gather_base_offsets32_i8)
GATHER_BASE_OFFSETS(__vec32_i8, int8_t, __vec32_i64, __gather_base_offsets64_i8)
GATHER_BASE_OFFSETS(__vec32_i16, int16_t, __vec32_i32, __gather_base_offsets32_i16)
GATHER_BASE_OFFSETS(__vec32_i16, int16_t, __vec32_i64, __gather_base_offsets64_i16)
// GATHER_BASE_OFFSETS(__vec32_i32, int32_t, __vec32_i32, __gather_base_offsets32_i32)
GATHER_BASE_OFFSETS(__vec32_i32, int32_t, __vec32_i64, __gather_base_offsets64_i32)
GATHER_BASE_OFFSETS(__vec32_i64, int64_t, __vec32_i32, __gather_base_offsets32_i64)
GATHER_BASE_OFFSETS(__vec32_i64, int64_t, __vec32_i64, __gather_base_offsets64_i64)

static FORCEINLINE __vec32_i32 __gather_base_offsets32_i32(uint8_t *b, __vec32_i32 varyingOffset, uint32_t scale,
                                                           __vec32_i32 constOffset, __vec32_i1 mask) {
    __vec32_i32 ret;
    __vec32_i32 offsets;
    __vec32_i32 vscale = __smear_i32<__vec32_i32>(scale);

    // Loop generated by the intrinsic
    offsets = __add(__mul(vscale, varyingOffset), constOffset);
    ret.v1 = _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask.m16.m1, offsets.v1, b, _MM_UPCONV_EPI32_NONE,
                                            1, _MM_HINT_NONE);
    ret.v2 = _mm512_mask_i32extgather_epi32(_mm512_undefined_epi32(), mask.m16.m2, offsets.v2, b + 64,
                                            _MM_UPCONV_EPI32_NONE, 1, _MM_HINT_NONE);
    return ret;
}

static FORCEINLINE __vec32_f __gather_base_offsets32_float(uint8_t *b, __vec32_i32 varyingOffset, uint32_t scale,
                                                           __vec32_i32 constOffset, __vec32_i1 mask) {
    __vec32_f ret;
    __vec32_i32 offsets;
    __vec32_i32 vscale = __smear_i32<__vec32_i32>(scale);

    // Loop generated by the intrinsic
    offsets = __add(__mul(vscale, varyingOffset), constOffset);
    ret.v1 = _mm512_mask_i32extgather_ps(_mm512_undefined_ps(), mask.m16.m1, offsets.v1, b, _MM_UPCONV_PS_NONE, 1,
                                         _MM_HINT_NONE);
    ret.v2 = _mm512_mask_i32extgather_ps(_mm512_undefined_ps(), mask.m16.m2, offsets.v2, b + 64, _MM_UPCONV_PS_NONE, 1,
                                         _MM_HINT_NONE);
    return ret;
}

#define GATHER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)
/*
static FORCEINLINE VTYPE FUNC(PTRTYPE ptrs, __vec32_i1 mask) {   \
    VTYPE ret;                                              \
    for (int i = 0; i < 16; ++i)                            \
        if ((mask.v & (1 << i)) != 0) {                     \
            STYPE *ptr = (STYPE *)ptrs.v[i];                \
            ret.v[i] = *ptr;                                \
        }                                                   \
    return ret;                                             \
}
*/

GATHER_GENERAL(__vec32_i8, int8_t, __vec32_i32, __gather32_i8)
GATHER_GENERAL(__vec32_i8, int8_t, __vec32_i64, __gather64_i8)
GATHER_GENERAL(__vec32_i16, int16_t, __vec32_i32, __gather32_i16)
GATHER_GENERAL(__vec32_i16, int16_t, __vec32_i64, __gather64_i16)
GATHER_GENERAL(__vec32_i32, int32_t, __vec32_i32, __gather32_i32)
GATHER_GENERAL(__vec32_i32, int32_t, __vec32_i64, __gather64_i32)
GATHER_GENERAL(__vec32_i64, int64_t, __vec32_i32, __gather32_i64)
GATHER_GENERAL(__vec32_i64, int64_t, __vec32_i64, __gather64_i64)

// scatter

#define SCATTER_BASE_OFFSETS(VTYPE, STYPE, OTYPE, FUNC)
/*
static FORCEINLINE void FUNC(unsigned char *b, OTYPE varyingOffset,     \
                             uint32_t scale, OTYPE constOffset,         \
                             VTYPE val, __vec32_i1 mask) {              \
    int8_t *base = (int8_t *)b;                                         \
    for (int i = 0; i < 16; ++i)                                        \
        if ((mask.v & (1 << i)) != 0) {                                 \
            STYPE *ptr = (STYPE *)(base + scale * varyingOffset.v[i] +  \
                                   constOffset.v[i]);                   \
            *ptr = val.v[i];                                            \
        }                                                               \
}
*/

SCATTER_BASE_OFFSETS(__vec32_i8, int8_t, __vec32_i32, __scatter_base_offsets32_i8)
SCATTER_BASE_OFFSETS(__vec32_i8, int8_t, __vec32_i64, __scatter_base_offsets64_i8)
SCATTER_BASE_OFFSETS(__vec32_i16, int16_t, __vec32_i32, __scatter_base_offsets32_i16)
SCATTER_BASE_OFFSETS(__vec32_i16, int16_t, __vec32_i64, __scatter_base_offsets64_i16)
// SCATTER_BASE_OFFSETS(__vec32_i32, int32_t, __vec32_i32, __scatter_base_offsets32_i32)
SCATTER_BASE_OFFSETS(__vec32_i32, int32_t, __vec32_i64, __scatter_base_offsets64_i32)
SCATTER_BASE_OFFSETS(__vec32_i64, int64_t, __vec32_i32, __scatter_base_offsets32_i64)
SCATTER_BASE_OFFSETS(__vec32_i64, int64_t, __vec32_i64, __scatter_base_offsets64_i64)

static FORCEINLINE void __scatter_base_offsets32_i32(uint8_t *b, __vec32_i32 varyingOffset, uint32_t scale,
                                                     __vec32_i32 constOffset, __vec32_i32 val, __vec32_i1 mask) {
    __vec32_i32 offsets;
    __vec32_i32 vscale = __smear_i32<__vec32_i32>(scale);

    // Loop generated by the intrinsic
    offsets = __add(__mul(vscale, varyingOffset), constOffset);
    _mm512_mask_i32extscatter_epi32(b, mask.m16.m1, offsets.v1, val.v1, _MM_DOWNCONV_EPI32_NONE, 1, _MM_HINT_NONE);
    _mm512_mask_i32extscatter_epi32(b, mask.m16.m2, offsets.v2, val.v2, _MM_DOWNCONV_EPI32_NONE, 1, _MM_HINT_NONE);
}

#define SCATTER_GENERAL(VTYPE, STYPE, PTRTYPE, FUNC)
/*
static FORCEINLINE void FUNC(PTRTYPE ptrs, VTYPE val, __vec32_i1 mask) {  \
    VTYPE ret;                                                       \
    for (int i = 0; i < 16; ++i)                                     \
        if ((mask.v & (1 << i)) != 0) {                              \
            STYPE *ptr = (STYPE *)ptrs.v[i];                         \
            *ptr = val.v[i];                                         \
        }                                                            \
}
*/
SCATTER_GENERAL(__vec32_i8, int8_t, __vec32_i32, __scatter32_i8)
SCATTER_GENERAL(__vec32_i8, int8_t, __vec32_i64, __scatter64_i8)
SCATTER_GENERAL(__vec32_i16, int16_t, __vec32_i32, __scatter32_i16)
SCATTER_GENERAL(__vec32_i16, int16_t, __vec32_i64, __scatter64_i16)
SCATTER_GENERAL(__vec32_i32, int32_t, __vec32_i32, __scatter32_i32)
SCATTER_GENERAL(__vec32_i32, int32_t, __vec32_i64, __scatter64_i32)
SCATTER_GENERAL(__vec32_i64, int64_t, __vec32_i32, __scatter32_i64)
SCATTER_GENERAL(__vec32_i64, int64_t, __vec32_i64, __scatter64_i64)

///////////////////////////////////////////////////////////////////////////
// packed load/store
/*
static FORCEINLINE int32_t __packed_load_active(int32_t *ptr, __vec32_i32 *val,
                                                __vec32_i1 mask) {
    int count = 0;
    for (int i = 0; i < 16; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            val->v[i] = *ptr++;
            ++count;
        }
    }
    return count;
}


static FORCEINLINE int32_t __packed_store_active(int32_t *ptr, __vec32_i32 val,
                                                 __vec32_i1 mask) {
    int count = 0;
    for (int i = 0; i < 16; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            *ptr++ = val.v[i];
            ++count;
        }
    }
    return count;
}

static FORCEINLINE int32_t __packed_load_active(uint32_t *ptr,
                                                __vec32_i32 *val,
                                                __vec32_i1 mask) {
    int count = 0;
    for (int i = 0; i < 16; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            val->v[i] = *ptr++;
            ++count;
        }
    }
    return count;
}


static FORCEINLINE int32_t __packed_store_active(uint32_t *ptr,
                                                 __vec32_i32 val,
                                                 __vec32_i1 mask) {
    int count = 0;
    for (int i = 0; i < 16; ++i) {
        if ((mask.v & (1 << i)) != 0) {
            *ptr++ = val.v[i];
            ++count;
        }
    }
    return count;
}
*/

///////////////////////////////////////////////////////////////////////////
// aos/soa

/*
static FORCEINLINE void __soa_to_aos3_float(__vec32_f v0, __vec32_f v1, __vec32_f v2,
                                            float *ptr) {
    for (int i = 0; i < 16; ++i) {
        *ptr++ = __extract_element(v0, i);
        *ptr++ = __extract_element(v1, i);
        *ptr++ = __extract_element(v2, i);
    }
}

static FORCEINLINE void __aos_to_soa3_float(float *ptr, __vec32_f *out0, __vec32_f *out1,
                                            __vec32_f *out2) {
    for (int i = 0; i < 16; ++i) {
        __insert_element(out0, i, *ptr++);
        __insert_element(out1, i, *ptr++);
        __insert_element(out2, i, *ptr++);
    }
}

static FORCEINLINE void __soa_to_aos4_float(__vec32_f v0, __vec32_f v1, __vec32_f v2,
                                            __vec32_f v3, float *ptr) {
    for (int i = 0; i < 16; ++i) {
        *ptr++ = __extract_element(v0, i);
        *ptr++ = __extract_element(v1, i);
        *ptr++ = __extract_element(v2, i);
        *ptr++ = __extract_element(v3, i);
    }
}

static FORCEINLINE void __aos_to_soa4_float(float *ptr, __vec32_f *out0, __vec32_f *out1,
                                            __vec32_f *out2, __vec32_f *out3) {
    for (int i = 0; i < 16; ++i) {
        __insert_element(out0, i, *ptr++);
        __insert_element(out1, i, *ptr++);
        __insert_element(out2, i, *ptr++);
        __insert_element(out3, i, *ptr++);
    }
}
*/

#ifdef WIN32
#include <windows.h>
#define __clock __rdtsc
#else // WIN32
static FORCEINLINE uint64_t __clock() {
    uint32_t low, high;
#ifdef __x86_64
    __asm__ __volatile__("xorl %%eax,%%eax \n    cpuid" ::: "%rax", "%rbx", "%rcx", "%rdx");
#else
    __asm__ __volatile__("xorl %%eax,%%eax \n    cpuid" ::: "%eax", "%ebx", "%ecx", "%edx");
#endif
    __asm__ __volatile__("rdtsc" : "=a"(low), "=d"(high));
    return (uint64_t)high << 32 | low;
}
#endif // !WIN32

#undef FORCEINLINE
#undef PRE_ALIGN
#undef POST_ALIGN
