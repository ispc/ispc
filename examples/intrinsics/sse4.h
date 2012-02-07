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
#ifdef _MSC_VER
#include <intrin.h>
#endif // _MSC_VER

#include <smmintrin.h>
#include <nmmintrin.h>

#if !defined(__SSE4_2__) && !defined(_MSC_VER)
#error "SSE 4.2 must be enabled in the C++ compiler to use this header."
#endif // !__SSE4_2__ && !msvc

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE __attribute__((always_inline)) inline
#endif

#undef FORCEINLINE
#define FORCEINLINE

typedef float __vec1_f;
typedef double __vec1_d;
typedef int8_t __vec1_i8;
typedef int16_t __vec1_i16;
typedef int32_t __vec1_i32;
typedef int64_t __vec1_i64;

struct __vec4_i1 {
    __vec4_i1() { }
    __vec4_i1(__m128 vv) : v(vv) {  }
    FORCEINLINE __vec4_i1(__m128i vv) : v(_mm_castsi128_ps(vv)) { }
    FORCEINLINE __vec4_i1(int a, int b, int c, int d) {
        v = _mm_castsi128_ps(_mm_set_epi32(d ? -1 : 0, c ? -1 : 0, 
                                           b ? -1 : 0, a ? -1 : 0));
    }

    __m128 v;
};

struct __vec4_f {
    __vec4_f() { }
    __vec4_f(__m128 vv) : v(vv) {  }
    FORCEINLINE __vec4_f(float a, float b, float c, float d) {
        v = _mm_set_ps(d, c, b, a);
    }
    FORCEINLINE __vec4_f(float *p) {
        v = _mm_loadu_ps(p);
    }

    FORCEINLINE operator __m128i() const { return _mm_castps_si128(v); }

    __m128 v;
};

struct __vec4_i64 {
    __vec4_i64() { }
    FORCEINLINE __vec4_i64(__m128i a, __m128i b) { v[0] = a; v[1] = b; }
    FORCEINLINE __vec4_i64(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        v[0] = _mm_set_epi32((b >> 32) & 0xffffffff, b & 0xffffffff, 
                             (a >> 32) & 0xffffffff, a & 0xffffffff);
        v[1] = _mm_set_epi32((d >> 32) & 0xffffffff, d & 0xffffffff, 
                             (c >> 32) & 0xffffffff, c & 0xffffffff);
    }
    FORCEINLINE __vec4_i64(uint64_t *p) {
        v[0] = _mm_loadu_si128((__m128i *)p);
        v[1] = _mm_loadu_si128((__m128i *)(p+2));
    }
    FORCEINLINE uint64_t &operator[](int i) { return ((uint64_t *)v)[i]; }

    __m128i v[2];
};

struct __vec4_i32 {
    __vec4_i32() { }
    FORCEINLINE __vec4_i32(__m128i vv) : v(vv) {  }
    FORCEINLINE __vec4_i32(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        v = _mm_set_epi32(d, c, b, a);
    }
    FORCEINLINE __vec4_i32(uint32_t *p) {
        v = _mm_loadu_si128((__m128i *)p);
    }

    FORCEINLINE operator __m128() const { return _mm_castsi128_ps(v); }

    __m128i v;
};

static inline int32_t __extract_element(__vec4_i32 v, int index);

struct __vec4_i16 {
    __vec4_i16() { }
    FORCEINLINE __vec4_i16(__m128i vv) : v(vv) {  }
    FORCEINLINE __vec4_i16(uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
        v = _mm_set_epi16(0, 0, 0, 0, d, c, b, a);
    }
    FORCEINLINE __vec4_i16(uint16_t *p) {
        v = _mm_set_epi16(0, 0, 0, 0, p[3], p[2], p[1], p[0]);
    }

    __m128i v;
};


struct __vec4_i8 {
    __vec4_i8() { }
    FORCEINLINE __vec4_i8(__m128i vv) : v(vv) {  }
    FORCEINLINE __vec4_i8(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
        v = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, d, c, b, a);
                         
    }
    FORCEINLINE __vec4_i8(uint8_t *p) {
        v = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, p[3], p[2], p[1], p[0]);
    }

    __m128i v;
};


struct __vec4_d {
    __vec4_d() { }
    FORCEINLINE __vec4_d(__m128d a, __m128d b) { v[0] = a; v[1] = b; }
    FORCEINLINE __vec4_d(double a, double b, double c, double d) {
        v[0] = _mm_set_pd(b, a);
        v[1] = _mm_set_pd(d, c);
    }

    __m128d v[2];
};


///////////////////////////////////////////////////////////////////////////
// SSE helpers / utility functions

static FORCEINLINE double _mm_extract_pd(__m128d v, int i)  {
    return ((double *)&v)[i];
}

static FORCEINLINE float bits_as_float(uint32_t v) {
    union {
        uint32_t ui;
        float f;
    } u;
    u.ui = v;
    return u.f;
}

template <typename T>
static FORCEINLINE T __select(bool test, T a, T b) {
    return test ? a : b;
}

#define INSERT_EXTRACT(VTYPE, STYPE)                                  \
    static FORCEINLINE STYPE __extract_element(VTYPE v, int index) { \
    return ((STYPE *)&v)[index];                                      \
}                                                    \
static FORCEINLINE void __insert_element(VTYPE *v, int index, STYPE val) { \
    ((STYPE *)v)[index] = val;                                        \
}

INSERT_EXTRACT(__vec1_i8, int8_t)
INSERT_EXTRACT(__vec1_i16, int16_t)
INSERT_EXTRACT(__vec1_i32, int32_t)
INSERT_EXTRACT(__vec1_i64, int64_t)
INSERT_EXTRACT(__vec1_f, float)
INSERT_EXTRACT(__vec1_d, double)

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
// mask ops

static FORCEINLINE uint32_t __movmsk(__vec4_i1 mask) {
    return _mm_movemask_ps(mask.v);
}

static FORCEINLINE __vec4_i1 __equal(__vec4_i1 a, __vec4_i1 b) {
    return _mm_cmpeq_epi32(_mm_castps_si128(a.v), _mm_castps_si128(b.v));
}

static FORCEINLINE __vec4_i1 __and(__vec4_i1 a, __vec4_i1 b) {
    return _mm_and_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __xor(__vec4_i1 a, __vec4_i1 b) {
    return _mm_xor_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __or(__vec4_i1 a, __vec4_i1 b) {
    return _mm_or_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __select(__vec4_i1 mask, __vec4_i1 a, __vec4_i1 b) {
    return _mm_blendv_ps(b.v, a.v, mask.v);
}

static FORCEINLINE bool __extract_element(__vec4_i1 v, int index) {
    return ((int32_t *)&v)[index] ? true : false;
}

static FORCEINLINE void __insert_element(__vec4_i1 *v, int index, bool val) {
    ((int32_t *)v)[index] = val ? -1 : 0;
}

static FORCEINLINE __vec4_i1 __load(__vec4_i1 *v, int align) {
    // FIXME: handle align of 16...
    return _mm_loadu_ps((float *)(&v->v));
}

static FORCEINLINE void __store(__vec4_i1 *p, __vec4_i1 value, int align) {
    // FIXME: handle align
    _mm_storeu_ps((float *)(&p->v), value.v);
}

///////////////////////////////////////////////////////////////////////////
// int8

static FORCEINLINE __vec4_i8 __add(__vec4_i8 a, __vec4_i8 b) {
    return _mm_add_epi8(a.v, b.v);
}

static FORCEINLINE __vec4_i8 __sub(__vec4_i8 a, __vec4_i8 b) {
    return _mm_sub_epi8(a.v, b.v);
}

static FORCEINLINE __vec4_i8 __mul(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8(_mm_extract_epi8(a.v, 0) * _mm_extract_epi8(b.v, 0),
                     _mm_extract_epi8(a.v, 1) * _mm_extract_epi8(b.v, 1),
                     _mm_extract_epi8(a.v, 2) * _mm_extract_epi8(b.v, 2),
                     _mm_extract_epi8(a.v, 3) * _mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i8 __or(__vec4_i8 a, __vec4_i8 b) {
    return _mm_or_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i8 __and(__vec4_i8 a, __vec4_i8 b) {
    return _mm_and_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i8 __xor(__vec4_i8 a, __vec4_i8 b) {
    return _mm_xor_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i8 __shl(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8(_mm_extract_epi8(a.v, 0) << _mm_extract_epi8(b.v, 0),
                     _mm_extract_epi8(a.v, 1) << _mm_extract_epi8(b.v, 1),
                     _mm_extract_epi8(a.v, 2) << _mm_extract_epi8(b.v, 2),
                     _mm_extract_epi8(a.v, 3) << _mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i8 __shl(__vec4_i8 a, int32_t b) {
    return __vec4_i8(_mm_extract_epi8(a.v, 0) << b,
                     _mm_extract_epi8(a.v, 1) << b,
                     _mm_extract_epi8(a.v, 2) << b,
                     _mm_extract_epi8(a.v, 3) << b);
}

static FORCEINLINE __vec4_i8 __udiv(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8((uint8_t)_mm_extract_epi8(a.v, 0) / 
                     (uint8_t)_mm_extract_epi8(b.v, 0),
                     (uint8_t)_mm_extract_epi8(a.v, 1) / 
                     (uint8_t)_mm_extract_epi8(b.v, 1),
                     (uint8_t)_mm_extract_epi8(a.v, 2) / 
                     (uint8_t)_mm_extract_epi8(b.v, 2),
                     (uint8_t)_mm_extract_epi8(a.v, 3) / 
                     (uint8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i8  __sdiv(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8((int8_t)_mm_extract_epi8(a.v, 0) / 
                     (int8_t)_mm_extract_epi8(b.v, 0),
                     (int8_t)_mm_extract_epi8(a.v, 1) / 
                     (int8_t)_mm_extract_epi8(b.v, 1),
                     (int8_t)_mm_extract_epi8(a.v, 2) / 
                     (int8_t)_mm_extract_epi8(b.v, 2),
                     (int8_t)_mm_extract_epi8(a.v, 3) / 
                     (int8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i8 __urem(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8((uint8_t)_mm_extract_epi8(a.v, 0) % 
                     (uint8_t)_mm_extract_epi8(b.v, 0),
                     (uint8_t)_mm_extract_epi8(a.v, 1) % 
                     (uint8_t)_mm_extract_epi8(b.v, 1),
                     (uint8_t)_mm_extract_epi8(a.v, 2) %
                     (uint8_t)_mm_extract_epi8(b.v, 2),
                     (uint8_t)_mm_extract_epi8(a.v, 3) %
                     (uint8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i8  __srem(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8((int8_t)_mm_extract_epi8(a.v, 0) % 
                     (int8_t)_mm_extract_epi8(b.v, 0),
                     (int8_t)_mm_extract_epi8(a.v, 1) % 
                     (int8_t)_mm_extract_epi8(b.v, 1),
                     (int8_t)_mm_extract_epi8(a.v, 2) %
                     (int8_t)_mm_extract_epi8(b.v, 2),
                     (int8_t)_mm_extract_epi8(a.v, 3) %
                     (int8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i8 __lshr(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8((uint8_t)_mm_extract_epi8(a.v, 0) >>
                     (uint8_t)_mm_extract_epi8(b.v, 0),
                     (uint8_t)_mm_extract_epi8(a.v, 1) >>
                     (uint8_t)_mm_extract_epi8(b.v, 1),
                     (uint8_t)_mm_extract_epi8(a.v, 2) >>
                     (uint8_t)_mm_extract_epi8(b.v, 2),
                     (uint8_t)_mm_extract_epi8(a.v, 3) >>
                     (uint8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i8 __lshr(__vec4_i8 a, int32_t b) {
    return __vec4_i8((uint8_t)_mm_extract_epi8(a.v, 0) >> b,
                     (uint8_t)_mm_extract_epi8(a.v, 1) >> b,
                     (uint8_t)_mm_extract_epi8(a.v, 2) >> b,
                     (uint8_t)_mm_extract_epi8(a.v, 3) >> b);
}

static FORCEINLINE __vec4_i8 __ashr(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8((int8_t)_mm_extract_epi8(a.v, 0) >>
                     (int8_t)_mm_extract_epi8(b.v, 0),
                     (int8_t)_mm_extract_epi8(a.v, 1) >>
                     (int8_t)_mm_extract_epi8(b.v, 1),
                     (int8_t)_mm_extract_epi8(a.v, 2) >>
                     (int8_t)_mm_extract_epi8(b.v, 2),
                     (int8_t)_mm_extract_epi8(a.v, 3) >>
                     (int8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i8 __ashr(__vec4_i8 a, int32_t b) {
    return __vec4_i8((int8_t)_mm_extract_epi8(a.v, 0) >> b,
                     (int8_t)_mm_extract_epi8(a.v, 1) >> b,
                     (int8_t)_mm_extract_epi8(a.v, 2) >> b,
                     (int8_t)_mm_extract_epi8(a.v, 3) >> b);
}

static FORCEINLINE __vec4_i1 __equal(__vec4_i8 a, __vec4_i8 b) {
    __m128i cmp = _mm_cmpeq_epi8(a.v, b.v);
    return __vec4_i1(_mm_extract_epi8(cmp, 0),
                     _mm_extract_epi8(cmp, 1),
                     _mm_extract_epi8(cmp, 2),
                     _mm_extract_epi8(cmp, 3));
}

static FORCEINLINE __vec4_i1 __not_equal(__vec4_i8 a, __vec4_i8 b) {
    return __xor(__equal(a, b), __vec4_i1(1, 1, 1, 1));
}

static FORCEINLINE __vec4_i1 __unsigned_less_equal(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i1((uint8_t)_mm_extract_epi8(a.v, 0) <=
                     (uint8_t)_mm_extract_epi8(b.v, 0),
                     (uint8_t)_mm_extract_epi8(a.v, 1) <=
                     (uint8_t)_mm_extract_epi8(b.v, 1),
                     (uint8_t)_mm_extract_epi8(a.v, 2) <=
                     (uint8_t)_mm_extract_epi8(b.v, 2),
                     (uint8_t)_mm_extract_epi8(a.v, 3) <= 
                     (uint8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i1 __unsigned_greater_equal(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i1((uint8_t)_mm_extract_epi8(a.v, 0) >=
                     (uint8_t)_mm_extract_epi8(b.v, 0),
                     (uint8_t)_mm_extract_epi8(a.v, 1) >=
                     (uint8_t)_mm_extract_epi8(b.v, 1),
                     (uint8_t)_mm_extract_epi8(a.v, 2) >=
                     (uint8_t)_mm_extract_epi8(b.v, 2),
                     (uint8_t)_mm_extract_epi8(a.v, 3) >=
                     (uint8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i1 __unsigned_less_than(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i1((uint8_t)_mm_extract_epi8(a.v, 0) <
                     (uint8_t)_mm_extract_epi8(b.v, 0),
                     (uint8_t)_mm_extract_epi8(a.v, 1) <
                     (uint8_t)_mm_extract_epi8(b.v, 1),
                     (uint8_t)_mm_extract_epi8(a.v, 2) <
                     (uint8_t)_mm_extract_epi8(b.v, 2),
                     (uint8_t)_mm_extract_epi8(a.v, 3) <
                     (uint8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i1 __unsigned_greater_than(__vec4_i8 a, __vec4_i8 b) {
    return __vec4_i1((uint8_t)_mm_extract_epi8(a.v, 0) >
                     (uint8_t)_mm_extract_epi8(b.v, 0),
                     (uint8_t)_mm_extract_epi8(a.v, 1) >
                     (uint8_t)_mm_extract_epi8(b.v, 1),
                     (uint8_t)_mm_extract_epi8(a.v, 2) >
                     (uint8_t)_mm_extract_epi8(b.v, 2),
                     (uint8_t)_mm_extract_epi8(a.v, 3) >
                     (uint8_t)_mm_extract_epi8(b.v, 3));
}

static FORCEINLINE __vec4_i1  __signed_less_than(__vec4_i8 a, __vec4_i8 b) {
    __m128i cmp = _mm_cmplt_epi8(a.v, b.v);
    return __vec4_i1(_mm_extract_epi8(cmp, 0),
                     _mm_extract_epi8(cmp, 1),
                     _mm_extract_epi8(cmp, 2),
                     _mm_extract_epi8(cmp, 3));
}

static FORCEINLINE __vec4_i1  __signed_less_equal(__vec4_i8 a, __vec4_i8 b) {
    return __or(__signed_less_than(a, b), __equal(a, b));
}

static FORCEINLINE __vec4_i1  __signed_greater_than(__vec4_i8 a, __vec4_i8 b) {
    __m128i cmp = _mm_cmpgt_epi8(a.v, b.v);
    return __vec4_i1(_mm_extract_epi8(cmp, 0),
                     _mm_extract_epi8(cmp, 1),
                     _mm_extract_epi8(cmp, 2),
                     _mm_extract_epi8(cmp, 3));
}

static FORCEINLINE __vec4_i1  __signed_greater_equal(__vec4_i8 a, __vec4_i8 b) {
    return __or(__signed_greater_than(a, b), __equal(a, b));
}

static FORCEINLINE __vec4_i8 __select(__vec4_i1 mask, __vec4_i8 a, __vec4_i8 b) {
    return __vec4_i8((_mm_extract_ps(mask.v, 0) != 0) ? _mm_extract_epi8(a.v, 0) : 
                                                        _mm_extract_epi8(b.v, 0),
                     (_mm_extract_ps(mask.v, 1) != 0) ? _mm_extract_epi8(a.v, 1) : 
                                                        _mm_extract_epi8(b.v, 1),
                     (_mm_extract_ps(mask.v, 2) != 0) ? _mm_extract_epi8(a.v, 2) : 
                                                        _mm_extract_epi8(b.v, 2),
                     (_mm_extract_ps(mask.v, 3) != 0) ? _mm_extract_epi8(a.v, 3) : 
                                                        _mm_extract_epi8(b.v, 3));
}

static FORCEINLINE int8_t __extract_element(__vec4_i8 v, int index) {
    return ((int8_t *)&v)[index];
}

static FORCEINLINE void __insert_element(__vec4_i8 *v, int index, int8_t val) {
    ((int8_t *)v)[index] = val;
}

static FORCEINLINE __vec4_i8 __smear_i8(int8_t v) {
    return _mm_set1_epi8(v);
}

static FORCEINLINE __vec4_i8 __broadcast_i8(__vec4_i8 v, int index) {
    return _mm_set1_epi8(__extract_element(v, index));
}

static FORCEINLINE __vec4_i8 __rotate_i8(__vec4_i8 v, int delta) {
    return __vec4_i8(__extract_element(v, delta     & 0x3),
                     __extract_element(v, (delta+1) & 0x3),
                     __extract_element(v, (delta+2) & 0x3),
                     __extract_element(v, (delta+3) & 0x3));
}

static FORCEINLINE __vec4_i8 __shuffle_i8(__vec4_i8 v, __vec4_i32 index) {
    return __vec4_i8(__extract_element(v, __extract_element(index, 0) & 0x3),
                     __extract_element(v, __extract_element(index, 1) & 0x3),
                     __extract_element(v, __extract_element(index, 2) & 0x3),
                     __extract_element(v, __extract_element(index, 3) & 0x3));
}

static FORCEINLINE __vec4_i8 __shuffle2_i8(__vec4_i8 v0, __vec4_i8 v1, 
                                           __vec4_i32 index) {
    uint8_t r[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t elt = __extract_element(index, i) & 0x7;
        r[i] = (elt < 4) ? __extract_element(v0, elt) : __extract_element(v1, elt & 0x3);
    }
    return __vec4_i8(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE __vec4_i8 __load(__vec4_i8 *v, int align) {
    uint8_t *ptr = (uint8_t *)(&v->v);
    return __vec4_i8(ptr[0], ptr[1], ptr[2], ptr[3]);
}

static FORCEINLINE void __store(__vec4_i8 *p, __vec4_i8 value, int align) {
    uint8_t *ptr = (uint8_t *)(&p->v);
    ptr[0] = _mm_extract_epi8(value.v, 0);
    ptr[1] = _mm_extract_epi8(value.v, 1);
    ptr[2] = _mm_extract_epi8(value.v, 2);
    ptr[3] = _mm_extract_epi8(value.v, 3);
}

///////////////////////////////////////////////////////////////////////////
// int16

static FORCEINLINE __vec4_i16 __add(__vec4_i16 a, __vec4_i16 b) {
    return _mm_add_epi16(a.v, b.v);
}

static FORCEINLINE __vec4_i16 __sub(__vec4_i16 a, __vec4_i16 b) {
    return _mm_sub_epi16(a.v, b.v);
}

static FORCEINLINE __vec4_i16 __mul(__vec4_i16 a, __vec4_i16 b) {
    return _mm_mullo_epi16(a.v, b.v);
}

static FORCEINLINE __vec4_i16 __or(__vec4_i16 a, __vec4_i16 b) {
    return _mm_or_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i16 __and(__vec4_i16 a, __vec4_i16 b) {
    return _mm_and_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i16 __xor(__vec4_i16 a, __vec4_i16 b) {
    return _mm_xor_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i16 __shl(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i16(_mm_extract_epi16(a.v, 0) << _mm_extract_epi16(b.v, 0),
                      _mm_extract_epi16(a.v, 1) << _mm_extract_epi16(b.v, 1),
                      _mm_extract_epi16(a.v, 2) << _mm_extract_epi16(b.v, 2),
                      _mm_extract_epi16(a.v, 3) << _mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i16 __shl(__vec4_i16 a, int32_t b) {
    return _mm_sll_epi16(a.v, _mm_set_epi32(0, 0, 0, b));
}

static FORCEINLINE __vec4_i16 __udiv(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i16((uint16_t)_mm_extract_epi16(a.v, 0) /
                      (uint16_t)_mm_extract_epi16(b.v, 0),
                      (uint16_t)_mm_extract_epi16(a.v, 1) /
                      (uint16_t)_mm_extract_epi16(b.v, 1),
                      (uint16_t)_mm_extract_epi16(a.v, 2) /
                      (uint16_t)_mm_extract_epi16(b.v, 2),
                      (uint16_t)_mm_extract_epi16(a.v, 3) /
                      (uint16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i16  __sdiv(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i16((int16_t)_mm_extract_epi16(a.v, 0) /
                      (int16_t)_mm_extract_epi16(b.v, 0),
                      (int16_t)_mm_extract_epi16(a.v, 1) /
                      (int16_t)_mm_extract_epi16(b.v, 1),
                      (int16_t)_mm_extract_epi16(a.v, 2) /
                      (int16_t)_mm_extract_epi16(b.v, 2),
                      (int16_t)_mm_extract_epi16(a.v, 3) /
                      (int16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i16 __urem(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i16((uint16_t)_mm_extract_epi16(a.v, 0) %
                      (uint16_t)_mm_extract_epi16(b.v, 0),
                      (uint16_t)_mm_extract_epi16(a.v, 1) %
                      (uint16_t)_mm_extract_epi16(b.v, 1),
                      (uint16_t)_mm_extract_epi16(a.v, 2) %
                      (uint16_t)_mm_extract_epi16(b.v, 2),
                      (uint16_t)_mm_extract_epi16(a.v, 3) %
                      (uint16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i16 __srem(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i16((int16_t)_mm_extract_epi16(a.v, 0) %
                      (int16_t)_mm_extract_epi16(b.v, 0),
                      (int16_t)_mm_extract_epi16(a.v, 1) %
                      (int16_t)_mm_extract_epi16(b.v, 1),
                      (int16_t)_mm_extract_epi16(a.v, 2) %
                      (int16_t)_mm_extract_epi16(b.v, 2),
                      (int16_t)_mm_extract_epi16(a.v, 3) %
                      (int16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i16 __lshr(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i16((uint16_t)_mm_extract_epi16(a.v, 0) >>
                      (uint16_t)_mm_extract_epi16(b.v, 0),
                      (uint16_t)_mm_extract_epi16(a.v, 1) >>
                      (uint16_t)_mm_extract_epi16(b.v, 1),
                      (uint16_t)_mm_extract_epi16(a.v, 2) >>
                      (uint16_t)_mm_extract_epi16(b.v, 2),
                      (uint16_t)_mm_extract_epi16(a.v, 3) >>
                      (uint16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i16 __lshr(__vec4_i16 a, int32_t b) {
    return _mm_srl_epi16(a.v, _mm_set_epi32(0, 0, 0, b));
}

static FORCEINLINE __vec4_i16 __ashr(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i16((int16_t)_mm_extract_epi16(a.v, 0) >>
                      (int16_t)_mm_extract_epi16(b.v, 0),
                      (int16_t)_mm_extract_epi16(a.v, 1) >>
                      (int16_t)_mm_extract_epi16(b.v, 1),
                      (int16_t)_mm_extract_epi16(a.v, 2) >>
                      (int16_t)_mm_extract_epi16(b.v, 2),
                      (int16_t)_mm_extract_epi16(a.v, 3) >>
                      (int16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i16 __ashr(__vec4_i16 a, int32_t b) {
    return _mm_sra_epi16(a.v, _mm_set_epi32(0, 0, 0, b));
}

static FORCEINLINE __vec4_i1 __equal(__vec4_i16 a, __vec4_i16 b) {
    __m128i cmp = _mm_cmpeq_epi16(a.v, b.v);
    return __vec4_i1(_mm_extract_epi16(cmp, 0),
                     _mm_extract_epi16(cmp, 1),
                     _mm_extract_epi16(cmp, 2),
                     _mm_extract_epi16(cmp, 3));
}

static FORCEINLINE __vec4_i1  __not_equal(__vec4_i16 a, __vec4_i16 b) {
    return __xor(__equal(a, b), __vec4_i1(1, 1, 1, 1));
}

static FORCEINLINE __vec4_i1 __unsigned_less_equal(__vec4_i16 a, __vec4_i16 b) {
    // FIXME: could use the trick that int32 does for the unsigned
    // comparisons so that we don't need to scalarie them.  (This also
    // applies to i8s...)
    return __vec4_i1((uint16_t)_mm_extract_epi16(a.v, 0) <=
                     (uint16_t)_mm_extract_epi16(b.v, 0),
                     (uint16_t)_mm_extract_epi16(a.v, 1) <=
                     (uint16_t)_mm_extract_epi16(b.v, 1),
                     (uint16_t)_mm_extract_epi16(a.v, 2) <=
                     (uint16_t)_mm_extract_epi16(b.v, 2),
                     (uint16_t)_mm_extract_epi16(a.v, 3) <= 
                     (uint16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i1 __unsigned_greater_equal(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i1((uint16_t)_mm_extract_epi16(a.v, 0) >=
                     (uint16_t)_mm_extract_epi16(b.v, 0),
                     (uint16_t)_mm_extract_epi16(a.v, 1) >=
                     (uint16_t)_mm_extract_epi16(b.v, 1),
                     (uint16_t)_mm_extract_epi16(a.v, 2) >=
                     (uint16_t)_mm_extract_epi16(b.v, 2),
                     (uint16_t)_mm_extract_epi16(a.v, 3) >=
                     (uint16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i1 __unsigned_less_than(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i1((uint16_t)_mm_extract_epi16(a.v, 0) <
                     (uint16_t)_mm_extract_epi16(b.v, 0),
                     (uint16_t)_mm_extract_epi16(a.v, 1) <
                     (uint16_t)_mm_extract_epi16(b.v, 1),
                     (uint16_t)_mm_extract_epi16(a.v, 2) <
                     (uint16_t)_mm_extract_epi16(b.v, 2),
                     (uint16_t)_mm_extract_epi16(a.v, 3) <
                     (uint16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i1 __unsigned_greater_than(__vec4_i16 a, __vec4_i16 b) {
    return __vec4_i1((uint16_t)_mm_extract_epi16(a.v, 0) >
                     (uint16_t)_mm_extract_epi16(b.v, 0),
                     (uint16_t)_mm_extract_epi16(a.v, 1) >
                     (uint16_t)_mm_extract_epi16(b.v, 1),
                     (uint16_t)_mm_extract_epi16(a.v, 2) >
                     (uint16_t)_mm_extract_epi16(b.v, 2),
                     (uint16_t)_mm_extract_epi16(a.v, 3) >
                     (uint16_t)_mm_extract_epi16(b.v, 3));
}

static FORCEINLINE __vec4_i1  __signed_less_than(__vec4_i16 a, __vec4_i16 b) {
    __m128i cmp = _mm_cmplt_epi16(a.v, b.v);
    return __vec4_i1(_mm_extract_epi16(cmp, 0),
                     _mm_extract_epi16(cmp, 1),
                     _mm_extract_epi16(cmp, 2),
                     _mm_extract_epi16(cmp, 3));
}

static FORCEINLINE __vec4_i1  __signed_less_equal(__vec4_i16 a, __vec4_i16 b) {
    return __or(__signed_less_than(a, b), __equal(a, b));
}

static FORCEINLINE __vec4_i1  __signed_greater_than(__vec4_i16 a, __vec4_i16 b) {
    __m128i cmp =  _mm_cmpgt_epi16(a.v, b.v);
    return __vec4_i1(_mm_extract_epi16(cmp, 0),
                     _mm_extract_epi16(cmp, 1),
                     _mm_extract_epi16(cmp, 2),
                     _mm_extract_epi16(cmp, 3));
}

static FORCEINLINE __vec4_i1  __signed_greater_equal(__vec4_i16 a, __vec4_i16 b) {
    return __or(__signed_greater_than(a, b), __equal(a, b));
}

static FORCEINLINE __vec4_i16 __select(__vec4_i1 mask, __vec4_i16 a, __vec4_i16 b) {
    return __vec4_i16((_mm_extract_ps(mask.v, 0) != 0) ? _mm_extract_epi16(a.v, 0) : 
                                                         _mm_extract_epi16(b.v, 0),
                      (_mm_extract_ps(mask.v, 1) != 0) ? _mm_extract_epi16(a.v, 1) : 
                                                         _mm_extract_epi16(b.v, 1),
                      (_mm_extract_ps(mask.v, 2) != 0) ? _mm_extract_epi16(a.v, 2) : 
                                                         _mm_extract_epi16(b.v, 2),
                      (_mm_extract_ps(mask.v, 3) != 0) ? _mm_extract_epi16(a.v, 3) : 
                                                         _mm_extract_epi16(b.v, 3));
}

static FORCEINLINE int16_t __extract_element(__vec4_i16 v, int index) {
    return ((int16_t *)&v)[index];
}

static FORCEINLINE void __insert_element(__vec4_i16 *v, int index, int16_t val) {
    ((int16_t *)v)[index] = val;
}

static FORCEINLINE __vec4_i16 __smear_i16(int16_t v) {
    return _mm_set1_epi16(v);
}

static FORCEINLINE __vec4_i16 __broadcast_i16(__vec4_i16 v, int index) {
    return _mm_set1_epi16(__extract_element(v, index));
}

static FORCEINLINE __vec4_i16 __rotate_i16(__vec4_i16 v, int delta) {
    return __vec4_i16(__extract_element(v, delta     & 0x3),
                      __extract_element(v, (delta+1) & 0x3),
                      __extract_element(v, (delta+2) & 0x3),
                      __extract_element(v, (delta+3) & 0x3));
}

static FORCEINLINE __vec4_i16 __shuffle_i16(__vec4_i16 v, __vec4_i32 index) {
    return __vec4_i16(__extract_element(v, __extract_element(index, 0) & 0x3),
                      __extract_element(v, __extract_element(index, 1) & 0x3),
                      __extract_element(v, __extract_element(index, 2) & 0x3),
                      __extract_element(v, __extract_element(index, 3) & 0x3));
}

static FORCEINLINE __vec4_i16 __shuffle2_i16(__vec4_i16 v0, __vec4_i16 v1, 
                                           __vec4_i32 index) {
    uint16_t r[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t elt = __extract_element(index, i) & 0x7;
        r[i] = (elt < 4) ? __extract_element(v0, elt) : __extract_element(v1, elt & 0x3);
    }
    return __vec4_i16(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE __vec4_i16 __load(__vec4_i16 *v, int align) {
    uint16_t *ptr = (uint16_t *)(&v->v);
    return __vec4_i16(ptr[0], ptr[1], ptr[2], ptr[3]);
}

static FORCEINLINE void __store(__vec4_i16 *p, __vec4_i16 value, int align) {
    uint16_t *ptr = (uint16_t *)(&p->v);
    ptr[0] = _mm_extract_epi16(value.v, 0);
    ptr[1] = _mm_extract_epi16(value.v, 1);
    ptr[2] = _mm_extract_epi16(value.v, 2);
    ptr[3] = _mm_extract_epi16(value.v, 3);
}


///////////////////////////////////////////////////////////////////////////
// int32

static FORCEINLINE __vec4_i32 __add(__vec4_i32 a, __vec4_i32 b) {
    return _mm_add_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __sub(__vec4_i32 a, __vec4_i32 b) {
    return _mm_sub_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __mul(__vec4_i32 a, __vec4_i32 b) {
    return _mm_mullo_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __or(__vec4_i32 a, __vec4_i32 b) {
    return _mm_or_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __and(__vec4_i32 a, __vec4_i32 b) {
    return _mm_and_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __xor(__vec4_i32 a, __vec4_i32 b) {
    return _mm_xor_si128(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __shl(__vec4_i32 a, __vec4_i32 b) {
    /* fixme: llvm generates thie code for shift left, which is presumably
       more efficient than doing each component individually as below.

LCPI0_0:
        .long   1065353216              ## 0x3f800000
        .long   1065353216              ## 0x3f800000
        .long   1065353216              ## 0x3f800000
        .long   1065353216              ## 0x3f800000
        .section        __TEXT,__text,regular,pure_instructions
        .globl  _f___ii
        .align  4, 0x90
_f___ii:                                ## @f___ii
## BB#0:                                ## %allocas
        pslld   $23, %xmm1
        paddd   LCPI0_0(%rip), %xmm1
        cvttps2dq       %xmm1, %xmm1
        pmulld  %xmm0, %xmm1
        movdqa  %xmm1, %xmm0
        ret

     */
    return __vec4_i32((uint32_t)_mm_extract_epi32(a.v, 0) << 
                      _mm_extract_epi32(b.v, 0),
                      (uint32_t)_mm_extract_epi32(a.v, 1) << 
                      _mm_extract_epi32(b.v, 1),
                      (uint32_t)_mm_extract_epi32(a.v, 2) << 
                      _mm_extract_epi32(b.v, 2),
                      (uint32_t)_mm_extract_epi32(a.v, 3) << 
                      _mm_extract_epi32(b.v, 3));
}

static FORCEINLINE __vec4_i32 __shl(__vec4_i32 a, int32_t b) {
    return _mm_sll_epi32(a.v, _mm_set_epi32(0, 0, 0, b));
}

static FORCEINLINE __vec4_i32 __udiv(__vec4_i32 a, __vec4_i32 b) {
    return __vec4_i32((uint32_t)_mm_extract_epi32(a.v, 0) / 
                      (uint32_t)_mm_extract_epi32(b.v, 0),
                      (uint32_t)_mm_extract_epi32(a.v, 1) / 
                      (uint32_t)_mm_extract_epi32(b.v, 1),
                      (uint32_t)_mm_extract_epi32(a.v, 2) / 
                      (uint32_t)_mm_extract_epi32(b.v, 2),
                      (uint32_t)_mm_extract_epi32(a.v, 3) / 
                      (uint32_t)_mm_extract_epi32(b.v, 3));
}

static FORCEINLINE __vec4_i32 __sdiv(__vec4_i32 a, __vec4_i32 b) {
    return __vec4_i32((int32_t)_mm_extract_epi32(a.v, 0) / 
                      (int32_t)_mm_extract_epi32(b.v, 0),
                      (int32_t)_mm_extract_epi32(a.v, 1) / 
                      (int32_t)_mm_extract_epi32(b.v, 1),
                      (int32_t)_mm_extract_epi32(a.v, 2) / 
                      (int32_t)_mm_extract_epi32(b.v, 2),
                      (int32_t)_mm_extract_epi32(a.v, 3) / 
                      (int32_t)_mm_extract_epi32(b.v, 3));
}

static FORCEINLINE __vec4_i32 __urem(__vec4_i32 a, __vec4_i32 b) {
    return __vec4_i32((uint32_t)_mm_extract_epi32(a.v, 0) %
                      (uint32_t)_mm_extract_epi32(b.v, 0),
                      (uint32_t)_mm_extract_epi32(a.v, 1) %
                      (uint32_t)_mm_extract_epi32(b.v, 1),
                      (uint32_t)_mm_extract_epi32(a.v, 2) %
                      (uint32_t)_mm_extract_epi32(b.v, 2),
                      (uint32_t)_mm_extract_epi32(a.v, 3) %
                      (uint32_t)_mm_extract_epi32(b.v, 3));
}

static FORCEINLINE __vec4_i32 __srem(__vec4_i32 a, __vec4_i32 b) {
    return __vec4_i32((int32_t)_mm_extract_epi32(a.v, 0) %
                      (int32_t)_mm_extract_epi32(b.v, 0),
                      (int32_t)_mm_extract_epi32(a.v, 1) %
                      (int32_t)_mm_extract_epi32(b.v, 1),
                      (int32_t)_mm_extract_epi32(a.v, 2) %
                      (int32_t)_mm_extract_epi32(b.v, 2),
                      (int32_t)_mm_extract_epi32(a.v, 3) %
                      (int32_t)_mm_extract_epi32(b.v, 3));
}

static FORCEINLINE __vec4_i32 __lshr(__vec4_i32 a, __vec4_i32 b) {
    return __vec4_i32((uint32_t)_mm_extract_epi32(a.v, 0) >>
                      _mm_extract_epi32(b.v, 0),
                      (uint32_t)_mm_extract_epi32(a.v, 1) >>
                      _mm_extract_epi32(b.v, 1),
                      (uint32_t)_mm_extract_epi32(a.v, 2) >>
                      _mm_extract_epi32(b.v, 2),
                      (uint32_t)_mm_extract_epi32(a.v, 3) >>
                      _mm_extract_epi32(b.v, 3));
}

static FORCEINLINE __vec4_i32 __lshr(__vec4_i32 a, int32_t b) {
    return _mm_srl_epi32(a.v, _mm_set_epi32(0, 0, 0, b));
}

static FORCEINLINE __vec4_i32 __ashr(__vec4_i32 a, __vec4_i32 b) {
    return __vec4_i32((int32_t)_mm_extract_epi32(a.v, 0) >>
                      _mm_extract_epi32(b.v, 0),
                      (int32_t)_mm_extract_epi32(a.v, 1) >>
                      _mm_extract_epi32(b.v, 1),
                      (int32_t)_mm_extract_epi32(a.v, 2) >>
                      _mm_extract_epi32(b.v, 2),
                      (int32_t)_mm_extract_epi32(a.v, 3) >>
                      _mm_extract_epi32(b.v, 3));
}

static FORCEINLINE __vec4_i32 __ashr(__vec4_i32 a, int32_t b) {
    return _mm_sra_epi32(a.v, _mm_set_epi32(0, 0, 0, b));
}

static FORCEINLINE __vec4_i1 __equal(__vec4_i32 a, __vec4_i32 b) {
    return _mm_cmpeq_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __not_equal(__vec4_i32 a, __vec4_i32 b) {
    return _mm_xor_si128(_mm_cmpeq_epi32(a.v, b.v),
                         _mm_cmpeq_epi32(a.v, a.v));
}

static FORCEINLINE __vec4_i1 __unsigned_less_equal(__vec4_i32 a, __vec4_i32 b) {
    // a<=b == (min(a,b) == a)
    return _mm_cmpeq_epi32(_mm_min_epu32(a.v, b.v), a.v);
}

static FORCEINLINE __vec4_i1 __signed_less_equal(__vec4_i32 a, __vec4_i32 b) {
    return _mm_or_si128(_mm_cmplt_epi32(a.v, b.v),
                        _mm_cmpeq_epi32(a.v, b.v));
}

static FORCEINLINE __vec4_i1 __unsigned_greater_equal(__vec4_i32 a, __vec4_i32 b) {
    // a>=b == (max(a,b) == a)
    return _mm_cmpeq_epi32(_mm_max_epu32(a.v, b.v), a.v);
}

static FORCEINLINE __vec4_i1 __signed_greater_equal(__vec4_i32 a, __vec4_i32 b) {
    return _mm_or_si128(_mm_cmpgt_epi32(a.v, b.v),
                        _mm_cmpeq_epi32(a.v, b.v));
}

static FORCEINLINE __vec4_i1 __unsigned_less_than(__vec4_i32 a, __vec4_i32 b) {
    a.v = _mm_xor_si128(a.v, _mm_set1_epi32(0x80000000));
    b.v = _mm_xor_si128(b.v, _mm_set1_epi32(0x80000000));
    return _mm_cmplt_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __signed_less_than(__vec4_i32 a, __vec4_i32 b) {
    return _mm_cmplt_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __unsigned_greater_than(__vec4_i32 a, __vec4_i32 b) {
    a.v = _mm_xor_si128(a.v, _mm_set1_epi32(0x80000000));
    b.v = _mm_xor_si128(b.v, _mm_set1_epi32(0x80000000));
    return _mm_cmpgt_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __signed_greater_than(__vec4_i32 a, __vec4_i32 b) {
    return _mm_cmpgt_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __select(__vec4_i1 mask, __vec4_i32 a, __vec4_i32 b) {
    return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(b.v), 
                                          _mm_castsi128_ps(a.v), mask.v));
}

static FORCEINLINE __vec4_i32 __smear_i32(int32_t v) {
    return _mm_set1_epi32(v);
}

static FORCEINLINE int32_t __extract_element(__vec4_i32 v, int index) {
    return ((int32_t *)&v)[index];
}

static FORCEINLINE void __insert_element(__vec4_i32 *v, int index, int32_t val) {
    ((int32_t *)v)[index] = val;
}

static FORCEINLINE __vec4_i32 __broadcast_i32(__vec4_i32 v, int index) {
    return _mm_set1_epi32(__extract_element(v, index));
}

static FORCEINLINE __vec4_i32 __rotate_i32(__vec4_i32 v, int delta) {
    return __vec4_i32(__extract_element(v, delta     & 0x3),
                      __extract_element(v, (delta+1) & 0x3),
                      __extract_element(v, (delta+2) & 0x3),
                      __extract_element(v, (delta+3) & 0x3));
}

static FORCEINLINE __vec4_i32 __shuffle_i32(__vec4_i32 v, __vec4_i32 index) {
    return __vec4_i32(__extract_element(v, __extract_element(index, 0) & 0x3),
                      __extract_element(v, __extract_element(index, 1) & 0x3),
                      __extract_element(v, __extract_element(index, 2) & 0x3),
                      __extract_element(v, __extract_element(index, 3) & 0x3));
}

static FORCEINLINE __vec4_i32 __shuffle2_i32(__vec4_i32 v0, __vec4_i32 v1, 
                                           __vec4_i32 index) {
    uint32_t r[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t elt = __extract_element(index, i) & 0x7;
        r[i] = (elt < 4) ? __extract_element(v0, elt) : __extract_element(v1, elt & 0x3);
    }
    return __vec4_i32(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE __vec4_i32 __load(__vec4_i32 *v, int align) {
    // FIXME: handle align of 16...
    return _mm_loadu_si128((__m128i *)(&v->v));
}

static void __store(__vec4_i32 *p, __vec4_i32 value, int align) {
    // FIXME: handle align
    _mm_storeu_si128((__m128i *)(&p->v), value.v);
}

///////////////////////////////////////////////////////////////////////////
// int64

static FORCEINLINE __vec4_i64 __add(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64(_mm_add_epi64(a.v[0], b.v[0]),
                      _mm_add_epi64(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_i64 __sub(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64(_mm_sub_epi64(a.v[0], b.v[0]),
                      _mm_sub_epi64(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_i64 __mul(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64(_mm_extract_epi64(a.v[0], 0) * _mm_extract_epi64(b.v[0], 0),
                      _mm_extract_epi64(a.v[0], 1) * _mm_extract_epi64(b.v[0], 1),
                      _mm_extract_epi64(a.v[1], 0) * _mm_extract_epi64(b.v[1], 0),
                      _mm_extract_epi64(a.v[1], 1) * _mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i64 __or(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64(_mm_or_si128(a.v[0], b.v[0]),
                      _mm_or_si128(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_i64 __and(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64(_mm_and_si128(a.v[0], b.v[0]),
                      _mm_and_si128(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_i64 __xor(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64(_mm_xor_si128(a.v[0], b.v[0]),
                      _mm_xor_si128(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_i64 __shl(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64(_mm_extract_epi64(a.v[0], 0) << _mm_extract_epi64(b.v[0], 0),
                      _mm_extract_epi64(a.v[0], 1) << _mm_extract_epi64(b.v[0], 1),
                      _mm_extract_epi64(a.v[1], 0) << _mm_extract_epi64(b.v[1], 0),
                      _mm_extract_epi64(a.v[1], 1) << _mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i64 __shl(__vec4_i64 a, int32_t b) {
    __m128i amt = _mm_set_epi32(0, 0, 0, b);
    return __vec4_i64(_mm_sll_epi64(a.v[0], amt),
                      _mm_sll_epi64(a.v[1], amt));
}

static FORCEINLINE __vec4_i64 __udiv(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((uint64_t)_mm_extract_epi64(a.v[0], 0) /
                      (uint64_t)_mm_extract_epi64(b.v[0], 0),
                      (uint64_t)_mm_extract_epi64(a.v[0], 1) /
                      (uint64_t)_mm_extract_epi64(b.v[0], 1),
                      (uint64_t)_mm_extract_epi64(a.v[1], 0) /
                      (uint64_t)_mm_extract_epi64(b.v[1], 0),
                      (uint64_t)_mm_extract_epi64(a.v[1], 1) /
                      (uint64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i64 __sdiv(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((int64_t)_mm_extract_epi64(a.v[0], 0) /
                      (int64_t)_mm_extract_epi64(b.v[0], 0),
                      (int64_t)_mm_extract_epi64(a.v[0], 1) /
                      (int64_t)_mm_extract_epi64(b.v[0], 1),
                      (int64_t)_mm_extract_epi64(a.v[1], 0) /
                      (int64_t)_mm_extract_epi64(b.v[1], 0),
                      (int64_t)_mm_extract_epi64(a.v[1], 1) /
                      (int64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i64 __urem(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((uint64_t)_mm_extract_epi64(a.v[0], 0) %
                      (uint64_t)_mm_extract_epi64(b.v[0], 0),
                      (uint64_t)_mm_extract_epi64(a.v[0], 1) %
                      (uint64_t)_mm_extract_epi64(b.v[0], 1),
                      (uint64_t)_mm_extract_epi64(a.v[1], 0) %
                      (uint64_t)_mm_extract_epi64(b.v[1], 0),
                      (uint64_t)_mm_extract_epi64(a.v[1], 1) %
                      (uint64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i64 __srem(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((int64_t)_mm_extract_epi64(a.v[0], 0) %
                      (int64_t)_mm_extract_epi64(b.v[0], 0),
                      (int64_t)_mm_extract_epi64(a.v[0], 1) %
                      (int64_t)_mm_extract_epi64(b.v[0], 1),
                      (int64_t)_mm_extract_epi64(a.v[1], 0) %
                      (int64_t)_mm_extract_epi64(b.v[1], 0),
                      (int64_t)_mm_extract_epi64(a.v[1], 1) %
                      (int64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i64 __lshr(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((uint64_t)_mm_extract_epi64(a.v[0], 0) >>
                      (uint64_t)_mm_extract_epi64(b.v[0], 0),
                      (uint64_t)_mm_extract_epi64(a.v[0], 1) >>
                      (uint64_t)_mm_extract_epi64(b.v[0], 1),
                      (uint64_t)_mm_extract_epi64(a.v[1], 0) >>
                      (uint64_t)_mm_extract_epi64(b.v[1], 0),
                      (uint64_t)_mm_extract_epi64(a.v[1], 1) >>
                      (uint64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i64 __lshr(__vec4_i64 a, int32_t b) {
    __m128i amt = _mm_set_epi32(0, 0, 0, b);
    return __vec4_i64(_mm_srl_epi64(a.v[0], amt),
                      _mm_srl_epi64(a.v[1], amt));
}

static FORCEINLINE __vec4_i64 __ashr(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((int64_t)_mm_extract_epi64(a.v[0], 0) >>
                      (int64_t)_mm_extract_epi64(b.v[0], 0),
                      (int64_t)_mm_extract_epi64(a.v[0], 1) >>
                      (int64_t)_mm_extract_epi64(b.v[0], 1),
                      (int64_t)_mm_extract_epi64(a.v[1], 0) >>
                      (int64_t)_mm_extract_epi64(b.v[1], 0),
                      (int64_t)_mm_extract_epi64(a.v[1], 1) >>
                      (int64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i64 __ashr(__vec4_i64 a, int32_t b) {
    return __vec4_i64((int64_t)_mm_extract_epi64(a.v[0], 0) >> b,
                      (int64_t)_mm_extract_epi64(a.v[0], 1) >> b,
                      (int64_t)_mm_extract_epi64(a.v[1], 0) >> b,
                      (int64_t)_mm_extract_epi64(a.v[1], 1) >> b);
}

static FORCEINLINE __vec4_i1 __equal(__vec4_i64 a, __vec4_i64 b) {
    __m128i cmp0 = _mm_cmpeq_epi64(a.v[0], b.v[0]);
    __m128i cmp1 = _mm_cmpeq_epi64(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castsi128_ps(cmp0), _mm_castsi128_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 2, 0));
}

static FORCEINLINE __vec4_i1 __not_equal(__vec4_i64 a, __vec4_i64 b) {
    return __xor(__equal(a, b), __vec4_i1(1, 1, 1, 1));
}

static FORCEINLINE __vec4_i1 __unsigned_less_equal(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i1((uint64_t)_mm_extract_epi64(a.v[0], 0) <=
                     (uint64_t)_mm_extract_epi64(b.v[0], 0),
                     (uint64_t)_mm_extract_epi64(a.v[0], 1) <=
                     (uint64_t)_mm_extract_epi64(b.v[0], 1),
                     (uint64_t)_mm_extract_epi64(a.v[1], 0) <=
                     (uint64_t)_mm_extract_epi64(b.v[1], 0),
                     (uint64_t)_mm_extract_epi64(a.v[1], 1) <=
                     (uint64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i1 __unsigned_greater_equal(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i1((uint64_t)_mm_extract_epi64(a.v[0], 0) >=
                     (uint64_t)_mm_extract_epi64(b.v[0], 0),
                     (uint64_t)_mm_extract_epi64(a.v[0], 1) >=
                     (uint64_t)_mm_extract_epi64(b.v[0], 1),
                     (uint64_t)_mm_extract_epi64(a.v[1], 0) >=
                     (uint64_t)_mm_extract_epi64(b.v[1], 0),
                     (uint64_t)_mm_extract_epi64(a.v[1], 1) >=
                     (uint64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i1 __unsigned_less_than(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i1((uint64_t)_mm_extract_epi64(a.v[0], 0) <
                     (uint64_t)_mm_extract_epi64(b.v[0], 0),
                     (uint64_t)_mm_extract_epi64(a.v[0], 1) <
                     (uint64_t)_mm_extract_epi64(b.v[0], 1),
                     (uint64_t)_mm_extract_epi64(a.v[1], 0) <
                     (uint64_t)_mm_extract_epi64(b.v[1], 0),
                     (uint64_t)_mm_extract_epi64(a.v[1], 1) <
                     (uint64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i1 __unsigned_greater_than(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i1((uint64_t)_mm_extract_epi64(a.v[0], 0) >
                     (uint64_t)_mm_extract_epi64(b.v[0], 0),
                     (uint64_t)_mm_extract_epi64(a.v[0], 1) >
                     (uint64_t)_mm_extract_epi64(b.v[0], 1),
                     (uint64_t)_mm_extract_epi64(a.v[1], 0) >
                     (uint64_t)_mm_extract_epi64(b.v[1], 0),
                     (uint64_t)_mm_extract_epi64(a.v[1], 1) >
                     (uint64_t)_mm_extract_epi64(b.v[1], 1));
}

static FORCEINLINE __vec4_i1 __signed_greater_than(__vec4_i64 a, __vec4_i64 b) {
    __m128i cmp0 = _mm_cmpgt_epi64(a.v[0], b.v[0]);
    __m128i cmp1 = _mm_cmpgt_epi64(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castsi128_ps(cmp0), _mm_castsi128_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 2, 0));
}

static FORCEINLINE __vec4_i1 __signed_greater_equal(__vec4_i64 a, __vec4_i64 b) {
    return __or(__signed_greater_than(a, b), __equal(a, b));
}

static FORCEINLINE __vec4_i1 __signed_less_than(__vec4_i64 a, __vec4_i64 b) {
    return __xor(__signed_greater_equal(a, b), __vec4_i1(1, 1, 1, 1));
}

static FORCEINLINE __vec4_i1 __signed_less_equal(__vec4_i64 a, __vec4_i64 b) {
    return __xor(__signed_greater_than(a, b), __vec4_i1(1, 1, 1, 1));
}

static FORCEINLINE __vec4_i64 __select(__vec4_i1 mask, __vec4_i64 a, __vec4_i64 b) {
    __m128 m0 = _mm_shuffle_ps(mask.v, mask.v, _MM_SHUFFLE(1, 1, 0, 0));
    __m128 m1 = _mm_shuffle_ps(mask.v, mask.v, _MM_SHUFFLE(3, 3, 2, 2));
    __m128d m0d = _mm_castps_pd(m0);
    __m128d m1d = _mm_castps_pd(m1);
    __m128d r0 = _mm_blendv_pd(_mm_castsi128_pd(b.v[0]), _mm_castsi128_pd(a.v[0]), m0d);
    __m128d r1 = _mm_blendv_pd(_mm_castsi128_pd(b.v[1]), _mm_castsi128_pd(a.v[1]), m1d);
    return __vec4_i64(_mm_castpd_si128(r0), _mm_castpd_si128(r1));
}

static FORCEINLINE __vec4_i64 __smear_i64(int64_t v) {
    return __vec4_i64(v, v, v, v);
}

static FORCEINLINE int64_t __extract_element(__vec4_i64 v, int index) {
    return ((int64_t *)&v)[index];
}

static FORCEINLINE void __insert_element(__vec4_i64 *v, int index, int64_t val) {
    ((int64_t *)v)[index] = val;
}

static FORCEINLINE __vec4_i64 __broadcast_i64(__vec4_i64 v, int index) {
    uint64_t val = __extract_element(v, index);
    return __vec4_i64(val, val, val, val);
}

static FORCEINLINE __vec4_i64 __rotate_i64(__vec4_i64 v, int delta) {
    return __vec4_i64(__extract_element(v, delta     & 0x3),
                      __extract_element(v, (delta+1) & 0x3),
                      __extract_element(v, (delta+2) & 0x3),
                      __extract_element(v, (delta+3) & 0x3));
}

static FORCEINLINE __vec4_i64 __shuffle_i64(__vec4_i64 v, __vec4_i32 index) {
    return __vec4_i64(__extract_element(v, __extract_element(index, 0) & 0x3),
                      __extract_element(v, __extract_element(index, 1) & 0x3),
                      __extract_element(v, __extract_element(index, 2) & 0x3),
                      __extract_element(v, __extract_element(index, 3) & 0x3));
}

static FORCEINLINE __vec4_i64 __shuffle2_i64(__vec4_i64 v0, __vec4_i64 v1, 
                                           __vec4_i32 index) {
    uint64_t r[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t elt = __extract_element(index, i) & 0x7;
        r[i] = (elt < 4) ? __extract_element(v0, elt) : __extract_element(v1, elt & 0x3);
    }
    return __vec4_i64(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE __vec4_i64 __load(__vec4_i64 *v, int align) {
    // FIXME: handle align of 16...
    return __vec4_i64(_mm_loadu_si128((__m128i *)(&v->v[0])),
                      _mm_loadu_si128((__m128i *)(&v->v[1])));
}

static FORCEINLINE void __store(__vec4_i64 *p, __vec4_i64 value, int align) {
    // FIXME: handle align
    _mm_storeu_si128((__m128i *)(&p->v[0]), value.v[0]);
    _mm_storeu_si128((__m128i *)(&p->v[1]), value.v[1]);
}

///////////////////////////////////////////////////////////////////////////
// float

static FORCEINLINE __vec4_f __add(__vec4_f a, __vec4_f b) {
    return _mm_add_ps(a.v, b.v);
}

static FORCEINLINE __vec4_f __sub(__vec4_f a, __vec4_f b) {
    return _mm_sub_ps(a.v, b.v);
}

static FORCEINLINE __vec4_f __mul(__vec4_f a, __vec4_f b) {
    return _mm_mul_ps(a.v, b.v);
}

static FORCEINLINE __vec4_f __div(__vec4_f a, __vec4_f b) {
    return _mm_div_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __equal(__vec4_f a, __vec4_f b) {
    return _mm_cmpeq_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __not_equal(__vec4_f a, __vec4_f b) {
    return _mm_cmpneq_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __less_than(__vec4_f a, __vec4_f b) {
    return _mm_cmplt_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __less_equal(__vec4_f a, __vec4_f b) {
    return _mm_cmple_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __greater_than(__vec4_f a, __vec4_f b) {
    return _mm_cmpgt_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __greater_equal(__vec4_f a, __vec4_f b) {
    return _mm_cmpge_ps(a.v, b.v);
}

static FORCEINLINE __vec4_i1 __ordered(__vec4_f a, __vec4_f b) {
    return _mm_cmpord_ps(a.v, b.v);
}

static FORCEINLINE __vec4_f __select(__vec4_i1 mask, __vec4_f a, __vec4_f b) {
    return _mm_blendv_ps(b.v, a.v, mask.v);
}

static FORCEINLINE __vec4_f __smear_float(float v) {
    return _mm_set1_ps(v);
}

static FORCEINLINE float __extract_element(__vec4_f v, int index) {
    return ((float *)&v)[index];
}

static FORCEINLINE void __insert_element(__vec4_f *v, int index, float val) {
    ((float *)v)[index] = val;
}

static FORCEINLINE __vec4_f __broadcast_float(__vec4_f v, int index) {
    return _mm_set1_ps(__extract_element(v, index));
}

static FORCEINLINE __vec4_f __rotate_float(__vec4_f v, int delta) {
    return __vec4_f(__extract_element(v, delta     & 0x3),
                    __extract_element(v, (delta+1) & 0x3),
                    __extract_element(v, (delta+2) & 0x3),
                    __extract_element(v, (delta+3) & 0x3));
}

static FORCEINLINE __vec4_f __shuffle_float(__vec4_f v, __vec4_i32 index) {
    return __vec4_f(__extract_element(v, __extract_element(index, 0) & 0x3),
                    __extract_element(v, __extract_element(index, 1) & 0x3),
                    __extract_element(v, __extract_element(index, 2) & 0x3),
                    __extract_element(v, __extract_element(index, 3) & 0x3));
}

static FORCEINLINE __vec4_f __shuffle2_float(__vec4_f v0, __vec4_f v1, 
                                             __vec4_i32 index) {
    float r[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t elt = __extract_element(index, i) & 0x7;
        r[i] = (elt < 4) ? __extract_element(v0, elt) : __extract_element(v1, elt & 0x3);
    }
    return __vec4_f(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE __vec4_f __load(__vec4_f *v, int align) {
    // FIXME: handle align of 16...
    return _mm_loadu_ps((float *)(&v->v));
}

static FORCEINLINE void __store(__vec4_f *p, __vec4_f value, int align) {
    // FIXME: handle align
    _mm_storeu_ps((float *)(&p->v), value.v);
}

///////////////////////////////////////////////////////////////////////////
// double

static FORCEINLINE __vec4_d __add(__vec4_d a, __vec4_d b) {
    return __vec4_d(_mm_add_pd(a.v[0], b.v[0]),
                    _mm_add_pd(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_d __sub(__vec4_d a, __vec4_d b) {
    return __vec4_d(_mm_sub_pd(a.v[0], b.v[0]),
                    _mm_sub_pd(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_d __mul(__vec4_d a, __vec4_d b) {
    return __vec4_d(_mm_mul_pd(a.v[0], b.v[0]),
                    _mm_mul_pd(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_d __div(__vec4_d a, __vec4_d b) {
    return __vec4_d(_mm_div_pd(a.v[0], b.v[0]),
                    _mm_div_pd(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_i1 __equal(__vec4_d a, __vec4_d b) {
    __m128d cmp0 = _mm_cmpeq_pd(a.v[0], b.v[0]);
    __m128d cmp1 = _mm_cmpeq_pd(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castpd_ps(cmp0), _mm_castpd_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 2, 0));
}

static FORCEINLINE __vec4_i1 __not_equal(__vec4_d a, __vec4_d b) {
    __m128d cmp0 = _mm_cmpneq_pd(a.v[0], b.v[0]);
    __m128d cmp1 = _mm_cmpneq_pd(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castpd_ps(cmp0), _mm_castpd_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 2, 0));
}

static FORCEINLINE __vec4_i1 __less_than(__vec4_d a, __vec4_d b) {
    __m128d cmp0 = _mm_cmplt_pd(a.v[0], b.v[0]);
    __m128d cmp1 = _mm_cmplt_pd(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castpd_ps(cmp0), _mm_castpd_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 2, 0));
}

static FORCEINLINE __vec4_i1 __less_equal(__vec4_d a, __vec4_d b) {
    __m128d cmp0 = _mm_cmple_pd(a.v[0], b.v[0]);
    __m128d cmp1 = _mm_cmple_pd(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castpd_ps(cmp0), _mm_castpd_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 2, 0));
}

static FORCEINLINE __vec4_i1 __greater_than(__vec4_d a, __vec4_d b) {
    __m128d cmp0 = _mm_cmpgt_pd(a.v[0], b.v[0]);
    __m128d cmp1 = _mm_cmpgt_pd(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castpd_ps(cmp0), _mm_castpd_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 0 ,2));
}

static FORCEINLINE __vec4_i1 __greater_equal(__vec4_d a, __vec4_d b) {
    __m128d cmp0 = _mm_cmpge_pd(a.v[0], b.v[0]);
    __m128d cmp1 = _mm_cmpge_pd(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castpd_ps(cmp0), _mm_castpd_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 2, 0));
}

static FORCEINLINE __vec4_i1 __ordered(__vec4_d a, __vec4_d b) {
    __m128d cmp0 = _mm_cmpord_pd(a.v[0], b.v[0]);
    __m128d cmp1 = _mm_cmpord_pd(a.v[1], b.v[1]);
    return _mm_shuffle_ps(_mm_castpd_ps(cmp0), _mm_castpd_ps(cmp1),
                          _MM_SHUFFLE(2, 0, 2, 0));
}

static FORCEINLINE __vec4_d __select(__vec4_i1 mask, __vec4_d a, __vec4_d b) {
    __m128 m0 = _mm_shuffle_ps(mask.v, mask.v, _MM_SHUFFLE(1, 1, 0, 0));
    __m128 m1 = _mm_shuffle_ps(mask.v, mask.v, _MM_SHUFFLE(3, 3, 2, 2));
    __m128d m0d = _mm_castps_pd(m0);
    __m128d m1d = _mm_castps_pd(m1);
    __m128d r0 = _mm_blendv_pd(b.v[0], a.v[0], m0d);
    __m128d r1 = _mm_blendv_pd(b.v[1], a.v[1], m1d);
    return __vec4_d(r0, r1);
}

static FORCEINLINE __vec4_d __smear_double(double v) {
    return __vec4_d(_mm_set1_pd(v), _mm_set1_pd(v));
}

static FORCEINLINE double __extract_element(__vec4_d v, int index) {
    return ((double *)&v)[index];
}

static FORCEINLINE void __insert_element(__vec4_d *v, int index, double val) {
    ((double *)v)[index] = val;
}

static FORCEINLINE __vec4_d __broadcast_double(__vec4_d v, int index) {
    return __vec4_d(_mm_set1_pd(__extract_element(v, index)),
                    _mm_set1_pd(__extract_element(v, index)));
}

static FORCEINLINE __vec4_d __rotate_double(__vec4_d v, int delta) {
    return __vec4_d(__extract_element(v, delta     & 0x3),
                    __extract_element(v, (delta+1) & 0x3),
                    __extract_element(v, (delta+2) & 0x3),
                    __extract_element(v, (delta+3) & 0x3));
}

static FORCEINLINE __vec4_d __shuffle_double(__vec4_d v, __vec4_i32 index) {
    return __vec4_d(__extract_element(v, __extract_element(index, 0) & 0x3),
                    __extract_element(v, __extract_element(index, 1) & 0x3),
                    __extract_element(v, __extract_element(index, 2) & 0x3),
                    __extract_element(v, __extract_element(index, 3) & 0x3));
}

static FORCEINLINE __vec4_d __shuffle2_double(__vec4_d v0, __vec4_d v1, 
                                              __vec4_i32 index) {
    double r[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t elt = __extract_element(index, i) & 0x7;
        r[i] = (elt < 4) ? __extract_element(v0, elt) : __extract_element(v1, elt & 0x3);
    }
    return __vec4_d(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE __vec4_d __load(__vec4_d *v, int align) {
    // FIXME: handle align of 16...
    return __vec4_d(_mm_loadu_pd((double *)(&v->v[0])),
                    _mm_loadu_pd((double *)(&v->v[1])));
}

static FORCEINLINE void __store(__vec4_d *p, __vec4_d value, int align) {
    // FIXME: handle align
    _mm_storeu_pd((double *)(&p->v[0]), value.v[0]);
    _mm_storeu_pd((double *)(&p->v[1]), value.v[1]);
}

///////////////////////////////////////////////////////////////////////////
// casts
// sign extension conversions

static FORCEINLINE __vec4_i64 __cast_sext(__vec4_i64, __vec4_i32 val) {
    return __vec4_i64((int64_t)((int32_t)_mm_extract_epi32(val.v, 0)),
                      (int64_t)((int32_t)_mm_extract_epi32(val.v, 1)),
                      (int64_t)((int32_t)_mm_extract_epi32(val.v, 2)),
                      (int64_t)((int32_t)_mm_extract_epi32(val.v, 3)));
}

static FORCEINLINE __vec4_i64 __cast_sext(__vec4_i64, __vec4_i16 val) {
    return __vec4_i64((int64_t)((int16_t)_mm_extract_epi16(val.v, 0)),
                      (int64_t)((int16_t)_mm_extract_epi16(val.v, 1)),
                      (int64_t)((int16_t)_mm_extract_epi16(val.v, 2)),
                      (int64_t)((int16_t)_mm_extract_epi16(val.v, 3)));
}

static FORCEINLINE __vec4_i64 __cast_sext(__vec4_i64, __vec4_i8 val) {
    return __vec4_i64((int64_t)((int8_t)_mm_extract_epi8(val.v, 0)),
                      (int64_t)((int8_t)_mm_extract_epi8(val.v, 1)),
                      (int64_t)((int8_t)_mm_extract_epi8(val.v, 2)),
                      (int64_t)((int8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_i32 __cast_sext(__vec4_i32, __vec4_i16 val) {
    return __vec4_i32((int32_t)((int16_t)_mm_extract_epi16(val.v, 0)),
                      (int32_t)((int16_t)_mm_extract_epi16(val.v, 1)),
                      (int32_t)((int16_t)_mm_extract_epi16(val.v, 2)),
                      (int32_t)((int16_t)_mm_extract_epi16(val.v, 3)));
}

static FORCEINLINE __vec4_i32 __cast_sext(__vec4_i32, __vec4_i8 val) {
    return __vec4_i32((int32_t)((int8_t)_mm_extract_epi8(val.v, 0)),
                      (int32_t)((int8_t)_mm_extract_epi8(val.v, 1)),
                      (int32_t)((int8_t)_mm_extract_epi8(val.v, 2)),
                      (int32_t)((int8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_i16 __cast_sext(__vec4_i16, __vec4_i8 val) {
    return __vec4_i16((int16_t)((int8_t)_mm_extract_epi8(val.v, 0)),
                      (int16_t)((int8_t)_mm_extract_epi8(val.v, 1)),
                      (int16_t)((int8_t)_mm_extract_epi8(val.v, 2)),
                      (int16_t)((int8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_i8 __cast_sext(__vec4_i8, __vec4_i1 v) {
    return __select(v, __smear_i8(0xff), __smear_i8(0));
}

static FORCEINLINE __vec4_i16 __cast_sext(__vec4_i16, __vec4_i1 v) {
    return __select(v, __smear_i16(0xffff), __smear_i16(0));
}

static FORCEINLINE __vec4_i32 __cast_sext(__vec4_i32, __vec4_i1 v) {
    return _mm_castps_si128(v.v);
}

static FORCEINLINE __vec4_i64 __cast_sext(__vec4_i64, __vec4_i1 v) {
    // For once it's nice that _mm_extract_ps() returns an int
    // representation of the float bits.
    return __vec4_i64((int64_t)((int32_t)_mm_extract_ps(v.v, 0)),
                      (int64_t)((int32_t)_mm_extract_ps(v.v, 1)),
                      (int64_t)((int32_t)_mm_extract_ps(v.v, 2)),
                      (int64_t)((int32_t)_mm_extract_ps(v.v, 3)));
}

// zero extension
static FORCEINLINE __vec4_i64 __cast_zext(__vec4_i64, __vec4_i32 val) {
    return __vec4_i64((uint64_t)((uint32_t)_mm_extract_epi32(val.v, 0)),
                      (uint64_t)((uint32_t)_mm_extract_epi32(val.v, 1)),
                      (uint64_t)((uint32_t)_mm_extract_epi32(val.v, 2)),
                      (uint64_t)((uint32_t)_mm_extract_epi32(val.v, 3)));
}

static FORCEINLINE __vec4_i64 __cast_zext(__vec4_i64, __vec4_i16 val) {
    return __vec4_i64((uint64_t)((uint16_t)_mm_extract_epi16(val.v, 0)),
                      (uint64_t)((uint16_t)_mm_extract_epi16(val.v, 1)),
                      (uint64_t)((uint16_t)_mm_extract_epi16(val.v, 2)),
                      (uint64_t)((uint16_t)_mm_extract_epi16(val.v, 3)));
}

static FORCEINLINE __vec4_i64 __cast_zext(__vec4_i64, __vec4_i8 val) {
    return __vec4_i64((uint64_t)((uint8_t)_mm_extract_epi8(val.v, 0)),
                      (uint64_t)((uint8_t)_mm_extract_epi8(val.v, 1)),
                      (uint64_t)((uint8_t)_mm_extract_epi8(val.v, 2)),
                      (uint64_t)((uint8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_i32 __cast_zext(__vec4_i32, __vec4_i16 val) {
    return __vec4_i32((uint32_t)((uint16_t)_mm_extract_epi16(val.v, 0)),
                      (uint32_t)((uint16_t)_mm_extract_epi16(val.v, 1)),
                      (uint32_t)((uint16_t)_mm_extract_epi16(val.v, 2)),
                      (uint32_t)((uint16_t)_mm_extract_epi16(val.v, 3)));
}

static FORCEINLINE __vec4_i32 __cast_zext(__vec4_i32, __vec4_i8 val) {
    return __vec4_i32((uint32_t)((uint8_t)_mm_extract_epi8(val.v, 0)),
                      (uint32_t)((uint8_t)_mm_extract_epi8(val.v, 1)),
                      (uint32_t)((uint8_t)_mm_extract_epi8(val.v, 2)),
                      (uint32_t)((uint8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_i16 __cast_zext(__vec4_i16, __vec4_i8 val) {
    return __vec4_i16((uint16_t)((uint8_t)_mm_extract_epi8(val.v, 0)),
                      (uint16_t)((uint8_t)_mm_extract_epi8(val.v, 1)),
                      (uint16_t)((uint8_t)_mm_extract_epi8(val.v, 2)),
                      (uint16_t)((uint8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_i8 __cast_zext(__vec4_i8, __vec4_i1 v) {
    return __select(v, __smear_i8(1), __smear_i8(0));
}

static FORCEINLINE __vec4_i16 __cast_zext(__vec4_i16, __vec4_i1 v) {
    return __select(v, __smear_i16(1), __smear_i16(0));
}

static FORCEINLINE __vec4_i32 __cast_zext(__vec4_i32, __vec4_i1 v) {
    return _mm_and_si128(_mm_castps_si128(v.v), _mm_set1_epi32(1));
}

static FORCEINLINE __vec4_i64 __cast_zext(__vec4_i64, __vec4_i1 v) {
    return __select(v, __smear_i64(1), __smear_i64(0));
}

// truncations
static FORCEINLINE __vec4_i32 __cast_trunc(__vec4_i32, __vec4_i64 val) {
    return __vec4_i32((int32_t)((int64_t)_mm_extract_epi64(val.v[0], 0)),
                      (int32_t)((int64_t)_mm_extract_epi64(val.v[0], 1)),
                      (int32_t)((int64_t)_mm_extract_epi64(val.v[1], 0)),
                      (int32_t)((int64_t)_mm_extract_epi64(val.v[1], 1)));
}

static FORCEINLINE __vec4_i16 __cast_trunc(__vec4_i16, __vec4_i64 val) {
    return __vec4_i16((int16_t)((int64_t)_mm_extract_epi64(val.v[0], 0)),
                      (int16_t)((int64_t)_mm_extract_epi64(val.v[0], 1)),
                      (int16_t)((int64_t)_mm_extract_epi64(val.v[1], 0)),
                      (int16_t)((int64_t)_mm_extract_epi64(val.v[1], 1)));
}

static FORCEINLINE __vec4_i8 __cast_trunc(__vec4_i8, __vec4_i64 val) {
    return __vec4_i8((int8_t)((int64_t)_mm_extract_epi64(val.v[0], 0)),
                     (int8_t)((int64_t)_mm_extract_epi64(val.v[0], 1)),
                     (int8_t)((int64_t)_mm_extract_epi64(val.v[1], 0)),
                     (int8_t)((int64_t)_mm_extract_epi64(val.v[1], 1)));
}

static FORCEINLINE __vec4_i16 __cast_trunc(__vec4_i16, __vec4_i32 val) {
    return __vec4_i16((int16_t)((int32_t)_mm_extract_epi32(val.v, 0)),
                      (int16_t)((int32_t)_mm_extract_epi32(val.v, 1)),
                      (int16_t)((int32_t)_mm_extract_epi32(val.v, 2)),
                      (int16_t)((int32_t)_mm_extract_epi32(val.v, 3)));
}

static FORCEINLINE __vec4_i8 __cast_trunc(__vec4_i8, __vec4_i32 val) {
    return __vec4_i8((int8_t)((int32_t)_mm_extract_epi32(val.v, 0)),
                     (int8_t)((int32_t)_mm_extract_epi32(val.v, 1)),
                     (int8_t)((int32_t)_mm_extract_epi32(val.v, 2)),
                     (int8_t)((int32_t)_mm_extract_epi32(val.v, 3)));
}

static FORCEINLINE __vec4_i8 __cast_trunc(__vec4_i8, __vec4_i16 val) {
    return __vec4_i8((int8_t)((int16_t)_mm_extract_epi16(val.v, 0)),
                     (int8_t)((int16_t)_mm_extract_epi16(val.v, 1)),
                     (int8_t)((int16_t)_mm_extract_epi16(val.v, 2)),
                     (int8_t)((int16_t)_mm_extract_epi16(val.v, 3)));
}

// signed int to float/double
static FORCEINLINE __vec4_f __cast_sitofp(__vec4_f, __vec4_i8 val) {
    return __vec4_f((float)((int8_t)_mm_extract_epi8(val.v, 0)),
                    (float)((int8_t)_mm_extract_epi8(val.v, 1)),
                    (float)((int8_t)_mm_extract_epi8(val.v, 2)),
                    (float)((int8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_f __cast_sitofp(__vec4_f, __vec4_i16 val) {
    return __vec4_f((float)((int16_t)_mm_extract_epi16(val.v, 0)),
                    (float)((int16_t)_mm_extract_epi16(val.v, 1)),
                    (float)((int16_t)_mm_extract_epi16(val.v, 2)),
                    (float)((int16_t)_mm_extract_epi16(val.v, 3)));
}

static FORCEINLINE __vec4_f __cast_sitofp(__vec4_f, __vec4_i32 val) {
    return _mm_cvtepi32_ps(val.v);
}

static FORCEINLINE __vec4_f __cast_sitofp(__vec4_f, __vec4_i64 val) {
    return __vec4_f((float)((int64_t)_mm_extract_epi64(val.v[0], 0)),
                    (float)((int64_t)_mm_extract_epi64(val.v[0], 1)),
                    (float)((int64_t)_mm_extract_epi64(val.v[1], 0)),
                    (float)((int64_t)_mm_extract_epi64(val.v[1], 1)));
}

static FORCEINLINE __vec4_d __cast_sitofp(__vec4_d, __vec4_i8 val) {
    return __vec4_d((double)((int8_t)_mm_extract_epi8(val.v, 0)),
                    (double)((int8_t)_mm_extract_epi8(val.v, 1)),
                    (double)((int8_t)_mm_extract_epi8(val.v, 2)),
                    (double)((int8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_d __cast_sitofp(__vec4_d, __vec4_i16 val) {
    return __vec4_d((double)((int16_t)_mm_extract_epi16(val.v, 0)),
                    (double)((int16_t)_mm_extract_epi16(val.v, 1)),
                    (double)((int16_t)_mm_extract_epi16(val.v, 2)),
                    (double)((int16_t)_mm_extract_epi16(val.v, 3)));
}

static FORCEINLINE __vec4_d __cast_sitofp(__vec4_d, __vec4_i32 val) {
    __m128d r0 = _mm_cvtepi32_pd(val.v);
    __m128 shuf = _mm_shuffle_ps(_mm_castsi128_ps(val.v),
                                 _mm_castsi128_ps(val.v),
                                 _MM_SHUFFLE(3, 2, 3, 2));
    __m128d r1 = _mm_cvtepi32_pd(_mm_castps_si128(shuf));
    return __vec4_d(r0, r1);
}

static FORCEINLINE __vec4_d __cast_sitofp(__vec4_d, __vec4_i64 val) {
    return __vec4_d((double)((int64_t)_mm_extract_epi64(val.v[0], 0)),
                    (double)((int64_t)_mm_extract_epi64(val.v[0], 1)),
                    (double)((int64_t)_mm_extract_epi64(val.v[1], 0)),
                    (double)((int64_t)_mm_extract_epi64(val.v[1], 1)));
}

// unsigned int to float/double
static FORCEINLINE __vec4_f __cast_uitofp(__vec4_f, __vec4_i8 val) {
    return __vec4_f((float)((uint8_t)_mm_extract_epi8(val.v, 0)),
                    (float)((uint8_t)_mm_extract_epi8(val.v, 1)),
                    (float)((uint8_t)_mm_extract_epi8(val.v, 2)),
                    (float)((uint8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_f __cast_uitofp(__vec4_f, __vec4_i16 val) {
    return __vec4_f((float)((uint16_t)_mm_extract_epi16(val.v, 0)),
                    (float)((uint16_t)_mm_extract_epi16(val.v, 1)),
                    (float)((uint16_t)_mm_extract_epi16(val.v, 2)),
                    (float)((uint16_t)_mm_extract_epi16(val.v, 3)));
}

static FORCEINLINE __vec4_f __cast_uitofp(__vec4_f, __vec4_i32 val) {
    return __vec4_f((float)((uint32_t)_mm_extract_epi32(val.v, 0)),
                    (float)((uint32_t)_mm_extract_epi32(val.v, 1)),
                    (float)((uint32_t)_mm_extract_epi32(val.v, 2)),
                    (float)((uint32_t)_mm_extract_epi32(val.v, 3)));
}

static FORCEINLINE __vec4_f __cast_uitofp(__vec4_f, __vec4_i64 val) {
    return __vec4_f((float)((uint64_t)_mm_extract_epi64(val.v[0], 0)),
                    (float)((uint64_t)_mm_extract_epi64(val.v[0], 1)),
                    (float)((uint64_t)_mm_extract_epi64(val.v[1], 0)),
                    (float)((uint64_t)_mm_extract_epi64(val.v[1], 1)));
}

static FORCEINLINE __vec4_d __cast_uitofp(__vec4_d, __vec4_i8 val) {
    return __vec4_d((double)((uint8_t)_mm_extract_epi8(val.v, 0)),
                    (double)((uint8_t)_mm_extract_epi8(val.v, 1)),
                    (double)((uint8_t)_mm_extract_epi8(val.v, 2)),
                    (double)((uint8_t)_mm_extract_epi8(val.v, 3)));
}

static FORCEINLINE __vec4_d __cast_uitofp(__vec4_d, __vec4_i16 val) {
    return __vec4_d((double)((uint16_t)_mm_extract_epi16(val.v, 0)),
                    (double)((uint16_t)_mm_extract_epi16(val.v, 1)),
                    (double)((uint16_t)_mm_extract_epi16(val.v, 2)),
                    (double)((uint16_t)_mm_extract_epi16(val.v, 3)));
}

static FORCEINLINE __vec4_d __cast_uitofp(__vec4_d, __vec4_i32 val) {
    return __vec4_d((double)((uint32_t)_mm_extract_epi32(val.v, 0)),
                    (double)((uint32_t)_mm_extract_epi32(val.v, 1)),
                    (double)((uint32_t)_mm_extract_epi32(val.v, 2)),
                    (double)((uint32_t)_mm_extract_epi32(val.v, 3)));
}

static FORCEINLINE __vec4_d __cast_uitofp(__vec4_d, __vec4_i64 val) {
    return __vec4_d((double)((uint64_t)_mm_extract_epi64(val.v[0], 0)),
                    (double)((uint64_t)_mm_extract_epi64(val.v[0], 1)),
                    (double)((uint64_t)_mm_extract_epi64(val.v[1], 0)),
                    (double)((uint64_t)_mm_extract_epi64(val.v[1], 1)));
}

static FORCEINLINE __vec4_f __cast_uitofp(__vec4_f, __vec4_i1 v) {
    return __select(v, __smear_float(1.), __smear_float(0.));
}

static FORCEINLINE __vec4_d __cast_uitofp(__vec4_d, __vec4_i1 v) {
    return __select(v, __smear_double(1.), __smear_double(0.));
}

// float/double to signed int
static FORCEINLINE __vec4_i8 __cast_fptosi(__vec4_i8, __vec4_f val) {
    return __vec4_i8((int8_t)bits_as_float(_mm_extract_ps(val.v, 0)),
                     (int8_t)bits_as_float(_mm_extract_ps(val.v, 1)),
                     (int8_t)bits_as_float(_mm_extract_ps(val.v, 2)),
                     (int8_t)bits_as_float(_mm_extract_ps(val.v, 3)));
}

static FORCEINLINE __vec4_i16 __cast_fptosi(__vec4_i16, __vec4_f val) {
    return __vec4_i16((int16_t)bits_as_float(_mm_extract_ps(val.v, 0)),
                      (int16_t)bits_as_float(_mm_extract_ps(val.v, 1)),
                      (int16_t)bits_as_float(_mm_extract_ps(val.v, 2)),
                      (int16_t)bits_as_float(_mm_extract_ps(val.v, 3)));
}

static FORCEINLINE __vec4_i32 __cast_fptosi(__vec4_i32, __vec4_f val) {
    return _mm_cvttps_epi32(val.v);
}

static FORCEINLINE __vec4_i64 __cast_fptosi(__vec4_i64, __vec4_f val) {
    return __vec4_i64((int64_t)bits_as_float(_mm_extract_ps(val.v, 0)),
                      (int64_t)bits_as_float(_mm_extract_ps(val.v, 1)),
                      (int64_t)bits_as_float(_mm_extract_ps(val.v, 2)),
                      (int64_t)bits_as_float(_mm_extract_ps(val.v, 3)));
}

static FORCEINLINE __vec4_i8 __cast_fptosi(__vec4_i8, __vec4_d val) {
    return __vec4_i8((int8_t)_mm_extract_pd(val.v[0], 0),
                     (int8_t)_mm_extract_pd(val.v[0], 1),
                     (int8_t)_mm_extract_pd(val.v[1], 0),
                     (int8_t)_mm_extract_pd(val.v[1], 1));
}

static FORCEINLINE __vec4_i16 __cast_fptosi(__vec4_i16, __vec4_d val) {
    return __vec4_i16((int16_t)_mm_extract_pd(val.v[0], 0),
                      (int16_t)_mm_extract_pd(val.v[0], 1),
                      (int16_t)_mm_extract_pd(val.v[1], 0),
                      (int16_t)_mm_extract_pd(val.v[1], 1));
}

static FORCEINLINE __vec4_i32 __cast_fptosi(__vec4_i32, __vec4_d val) {
    __m128i r0 = _mm_cvtpd_epi32(val.v[0]);
    __m128i r1 = _mm_cvtpd_epi32(val.v[1]);
    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(r0), _mm_castsi128_ps(r1),
                                           _MM_SHUFFLE(1, 0, 1, 0)));
}

static FORCEINLINE __vec4_i64 __cast_fptosi(__vec4_i64, __vec4_d val) {
    return __vec4_i64((int64_t)_mm_extract_pd(val.v[0], 0),
                      (int64_t)_mm_extract_pd(val.v[0], 1),
                      (int64_t)_mm_extract_pd(val.v[1], 0),
                      (int64_t)_mm_extract_pd(val.v[1], 1));
}

// float/double to unsigned int
static FORCEINLINE __vec4_i8 __cast_fptoui(__vec4_i8, __vec4_f val) {
    return __vec4_i8((uint8_t)bits_as_float(_mm_extract_ps(val.v, 0)),
                     (uint8_t)bits_as_float(_mm_extract_ps(val.v, 1)),
                     (uint8_t)bits_as_float(_mm_extract_ps(val.v, 2)),
                     (uint8_t)bits_as_float(_mm_extract_ps(val.v, 3)));
}

static FORCEINLINE __vec4_i16 __cast_fptoui(__vec4_i16, __vec4_f val) {
    return __vec4_i16((uint16_t)bits_as_float(_mm_extract_ps(val.v, 0)),
                      (uint16_t)bits_as_float(_mm_extract_ps(val.v, 1)),
                      (uint16_t)bits_as_float(_mm_extract_ps(val.v, 2)),
                      (uint16_t)bits_as_float(_mm_extract_ps(val.v, 3)));
}

static FORCEINLINE __vec4_i32 __cast_fptoui(__vec4_i32, __vec4_f val) {
    return __vec4_i32((uint32_t)bits_as_float(_mm_extract_ps(val.v, 0)),
                      (uint32_t)bits_as_float(_mm_extract_ps(val.v, 1)),
                      (uint32_t)bits_as_float(_mm_extract_ps(val.v, 2)),
                      (uint32_t)bits_as_float(_mm_extract_ps(val.v, 3)));
}

static FORCEINLINE __vec4_i64 __cast_fptoui(__vec4_i64, __vec4_f val) {
    return __vec4_i64((uint64_t)bits_as_float(_mm_extract_ps(val.v, 0)),
                      (uint64_t)bits_as_float(_mm_extract_ps(val.v, 1)),
                      (uint64_t)bits_as_float(_mm_extract_ps(val.v, 2)),
                      (uint64_t)bits_as_float(_mm_extract_ps(val.v, 3)));
}

static FORCEINLINE __vec4_i8 __cast_fptoui(__vec4_i8, __vec4_d val) {
    return __vec4_i8((uint8_t)_mm_extract_pd(val.v[0], 0),
                     (uint8_t)_mm_extract_pd(val.v[0], 1),
                     (uint8_t)_mm_extract_pd(val.v[1], 0),
                     (uint8_t)_mm_extract_pd(val.v[1], 1));
}

static FORCEINLINE __vec4_i16 __cast_fptoui(__vec4_i16, __vec4_d val) {
    return __vec4_i16((uint16_t)_mm_extract_pd(val.v[0], 0),
                      (uint16_t)_mm_extract_pd(val.v[0], 1),
                      (uint16_t)_mm_extract_pd(val.v[1], 0),
                      (uint16_t)_mm_extract_pd(val.v[1], 1));
}

static FORCEINLINE __vec4_i32 __cast_fptoui(__vec4_i32, __vec4_d val) {
    return __vec4_i32((uint32_t)_mm_extract_pd(val.v[0], 0),
                      (uint32_t)_mm_extract_pd(val.v[0], 1),
                      (uint32_t)_mm_extract_pd(val.v[1], 0),
                      (uint32_t)_mm_extract_pd(val.v[1], 1));
}

static FORCEINLINE __vec4_i64 __cast_fptoui(__vec4_i64, __vec4_d val) {
    return __vec4_i64((int64_t)_mm_extract_pd(val.v[0], 0),
                      (int64_t)_mm_extract_pd(val.v[0], 1),
                      (int64_t)_mm_extract_pd(val.v[1], 0),
                      (int64_t)_mm_extract_pd(val.v[1], 1));
}

// float/double conversions
static FORCEINLINE __vec4_f __cast_fptrunc(__vec4_f, __vec4_d val) {
    __m128 r0 = _mm_cvtpd_ps(val.v[0]);
    __m128 r1 = _mm_cvtpd_ps(val.v[1]);
    return _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(1, 0, 1, 0));
}

static FORCEINLINE __vec4_d __cast_fpext(__vec4_d, __vec4_f val) {
    return __vec4_d(_mm_cvtps_pd(val.v),
                    _mm_cvtps_pd(_mm_shuffle_ps(val.v, val.v, 
                                                _MM_SHUFFLE(3, 2, 3, 2))));
}

static FORCEINLINE __vec4_f __cast_bits(__vec4_f, __vec4_i32 val) {
    return _mm_castsi128_ps(val.v);
}

static FORCEINLINE __vec4_i32 __cast_bits(__vec4_i32, __vec4_f val) {
    return _mm_castps_si128(val.v);
}

static FORCEINLINE __vec4_d __cast_bits(__vec4_d, __vec4_i64 val) {
    return __vec4_d(_mm_castsi128_pd(val.v[0]),
                    _mm_castsi128_pd(val.v[1]));
}

static FORCEINLINE __vec4_i64 __cast_bits(__vec4_i64, __vec4_d val) {
    return __vec4_i64(_mm_castpd_si128(val.v[0]),
                      _mm_castpd_si128(val.v[1]));
}

///////////////////////////////////////////////////////////////////////////
// various math functions

static FORCEINLINE void __fastmath() {
}

static FORCEINLINE float __round_uniform_float(float v) {
    __m128 r = _mm_set_ss(v);
    r = _mm_round_ss(r, r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return bits_as_float(_mm_extract_ps(r, 0));
}

static FORCEINLINE float __floor_uniform_float(float v) {
    __m128 r = _mm_set_ss(v);
    r = _mm_round_ss(r, r, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    return bits_as_float(_mm_extract_ps(r, 0));
}

static FORCEINLINE float __ceil_uniform_float(float v) {
    __m128 r = _mm_set_ss(v);
    r = _mm_round_ss(r, r, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    return bits_as_float(_mm_extract_ps(r, 0));
}

static FORCEINLINE double __round_uniform_double(double v) {
    __m128d r = _mm_set_sd(v);
    r = _mm_round_sd(r, r, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_extract_pd(r, 0);
}

static FORCEINLINE double __floor_uniform_double(double v) {
    __m128d r = _mm_set_sd(v);
    r = _mm_round_sd(r, r, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    return _mm_extract_pd(r, 0);
}

static FORCEINLINE double __ceil_uniform_double(double v) {
    __m128d r = _mm_set_sd(v);
    r = _mm_round_sd(r, r, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    return _mm_extract_pd(r, 0);
}

static FORCEINLINE __vec4_f __round_varying_float(__vec4_f v) {
    return _mm_round_ps(v.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static FORCEINLINE __vec4_f __floor_varying_float(__vec4_f v) {
    return _mm_round_ps(v.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

static FORCEINLINE __vec4_f __ceil_varying_float(__vec4_f v) {
    return _mm_round_ps(v.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

static FORCEINLINE __vec4_d __round_varying_double(__vec4_d v) {
    return __vec4_d(_mm_round_pd(v.v[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
                    _mm_round_pd(v.v[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

static FORCEINLINE __vec4_d __floor_varying_double(__vec4_d v) {
    return __vec4_d(_mm_round_pd(v.v[0], _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC),
                    _mm_round_pd(v.v[1], _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}

static FORCEINLINE __vec4_d __ceil_varying_double(__vec4_d v) {
    return __vec4_d(_mm_round_pd(v.v[0], _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC),
                    _mm_round_pd(v.v[1], _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
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

static FORCEINLINE  __vec4_f  __max_varying_float(__vec4_f a, __vec4_f b) {
    return _mm_max_ps(a.v, b.v);
}

static FORCEINLINE  __vec4_f  __min_varying_float(__vec4_f a, __vec4_f b) {
    return _mm_min_ps(a.v, b.v);
}

static FORCEINLINE __vec4_d __max_varying_double(__vec4_d a, __vec4_d b) {
    return __vec4_d(_mm_max_pd(a.v[0], b.v[0]),
                    _mm_max_pd(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_d __min_varying_double(__vec4_d a, __vec4_d b) {
    return __vec4_d(_mm_min_pd(a.v[0], b.v[0]),
                    _mm_min_pd(a.v[1], b.v[1]));
}

static FORCEINLINE __vec4_i32 __max_varying_int32(__vec4_i32 a, __vec4_i32 b) {
    return _mm_max_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __min_varying_int32(__vec4_i32 a, __vec4_i32 b) {
    return _mm_min_epi32(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __max_varying_uint32(__vec4_i32 a, __vec4_i32 b) {
    return _mm_max_epu32(a.v, b.v);
}

static FORCEINLINE __vec4_i32 __min_varying_uint32(__vec4_i32 a, __vec4_i32 b) {
    return _mm_min_epu32(a.v, b.v);
}

static FORCEINLINE __vec4_i64 __max_varying_int64(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((int64_t)a[0] > (int64_t)b[0] ? a[0] : b[0],
                      (int64_t)a[1] > (int64_t)b[1] ? a[1] : b[1],
                      (int64_t)a[2] > (int64_t)b[2] ? a[2] : b[2],
                      (int64_t)a[3] > (int64_t)b[3] ? a[3] : b[3]);
}

static FORCEINLINE __vec4_i64 __min_varying_int64(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((int64_t)a[0] < (int64_t)b[0] ? a[0] : b[0],
                      (int64_t)a[1] < (int64_t)b[1] ? a[1] : b[1],
                      (int64_t)a[2] < (int64_t)b[2] ? a[2] : b[2],
                      (int64_t)a[3] < (int64_t)b[3] ? a[3] : b[3]);
}

static FORCEINLINE __vec4_i64 __max_varying_uint64(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((uint64_t)a[0] > (uint64_t)b[0] ? a[0] : b[0],
                      (uint64_t)a[1] > (uint64_t)b[1] ? a[1] : b[1],
                      (uint64_t)a[2] > (uint64_t)b[2] ? a[2] : b[2],
                      (uint64_t)a[3] > (uint64_t)b[3] ? a[3] : b[3]);
}

static FORCEINLINE __vec4_i64 __min_varying_uint64(__vec4_i64 a, __vec4_i64 b) {
    return __vec4_i64((uint64_t)a[0] < (uint64_t)b[0] ? a[0] : b[0],
                      (uint64_t)a[1] < (uint64_t)b[1] ? a[1] : b[1],
                      (uint64_t)a[2] < (uint64_t)b[2] ? a[2] : b[2],
                      (uint64_t)a[3] < (uint64_t)b[3] ? a[3] : b[3]);
}

// sqrt/rsqrt/rcp

static FORCEINLINE float __rsqrt_uniform_float(float v) {
    __m128 vv = _mm_set_ss(v);
    __m128 rsqrt = _mm_rsqrt_ss(vv);
    // Newton-Raphson iteration to improve precision
    // return 0.5 * rsqrt * (3. - (v * rsqrt) * rsqrt);
    __m128 v_rsqrt = _mm_mul_ss(rsqrt, vv);
    __m128 v_r_r = _mm_mul_ss(v_rsqrt, rsqrt);
    __m128 three_sub = _mm_sub_ss(_mm_set_ss(3.f), v_r_r);
    __m128 rs_mul = _mm_mul_ss(rsqrt, three_sub);
    __m128 half_scale = _mm_mul_ss(_mm_set_ss(0.5), rs_mul);
    return bits_as_float(_mm_extract_ps(half_scale, 0));
}

static FORCEINLINE float __rcp_uniform_float(float v) {
    __m128 rcp = _mm_rcp_ss(_mm_set_ss(v));
    // N-R iteration:
    __m128 m = _mm_mul_ss(_mm_set_ss(v), rcp);
    __m128 twominus = _mm_sub_ss(_mm_set_ss(2.f), m);
    __m128 r = _mm_mul_ss(rcp, twominus);
    return bits_as_float(_mm_extract_ps(r, 0));
}

static FORCEINLINE float __sqrt_uniform_float(float v) {
    __m128 r = _mm_set_ss(v);
    r = _mm_sqrt_ss(r);
    return bits_as_float(_mm_extract_ps(r, 0));
}

static FORCEINLINE double __sqrt_uniform_double(double v) {
    __m128d r = _mm_set_sd(v);
    r = _mm_sqrt_sd(r, r);
    return _mm_extract_pd(r, 0);
}

static FORCEINLINE __vec4_f __rcp_varying_float(__vec4_f v) {
    __m128 rcp = _mm_rcp_ps(v.v);
    // N-R iteration:
    __m128 m = _mm_mul_ps(v.v, rcp);
    __m128 twominus = _mm_sub_ps(_mm_set1_ps(2.f), m);
    __m128 r = _mm_mul_ps(rcp, twominus);
    return r;
}

static FORCEINLINE __vec4_f __rsqrt_varying_float(__vec4_f v) {
    __m128 rsqrt = _mm_rsqrt_ps(v.v);
    // Newton-Raphson iteration to improve precision
    // return 0.5 * rsqrt * (3. - (v * rsqrt) * rsqrt);
    __m128 v_rsqrt = _mm_mul_ps(rsqrt, v.v);
    __m128 v_r_r = _mm_mul_ps(v_rsqrt, rsqrt);
    __m128 three_sub = _mm_sub_ps(_mm_set1_ps(3.f), v_r_r);
    __m128 rs_mul = _mm_mul_ps(rsqrt, three_sub);
    __m128 half_scale = _mm_mul_ps(_mm_set1_ps(0.5), rs_mul);
    return half_scale;
}

static FORCEINLINE __vec4_f __sqrt_varying_float(__vec4_f v) {
    return _mm_sqrt_ps(v.v);
}

static FORCEINLINE __vec4_d __sqrt_varying_double(__vec4_d v) {
    return __vec4_d(_mm_sqrt_pd(v.v[0]), _mm_sqrt_pd(v.v[1]));
}

///////////////////////////////////////////////////////////////////////////
// bit ops

static FORCEINLINE int32_t __popcnt_int32(uint32_t v) {
    return _mm_popcnt_u32(v);
}

static FORCEINLINE int32_t __popcnt_int64(uint64_t v) {
    return _mm_popcnt_u64(v);
}

static FORCEINLINE int32_t __count_trailing_zeros_i32(uint32_t v) {
#if 0
    // http://aggregate.org/MAGIC/#Trailing Zero Count
    return __popcnt_int32((v & -v) - 1);
#else
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward(&i, v);
    return i;
#else
    return __builtin_ctz(v);
#endif
#endif
}

static FORCEINLINE int64_t __count_trailing_zeros_i64(uint64_t v) {
#if 0
    // http://aggregate.org/MAGIC/#Trailing Zero Count
    return __popcnt_int64((v & -v) - 1);
#else
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, v);
    return i;
#else
    return __builtin_ctzl(v);
#endif
#endif
}

static FORCEINLINE int32_t __count_leading_zeros_i32(uint32_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanReverse(&i, v);
    return i;
#else
    return __builtin_clz(v);
#endif
}

static FORCEINLINE int64_t __count_leading_zeros_i64(uint64_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanReverse64(&i, v);
    return i;
#else
    return __builtin_clzl(v);
#endif
}


///////////////////////////////////////////////////////////////////////////
// reductions

static FORCEINLINE float __reduce_add_float(__vec4_f v) {
    float r = bits_as_float(_mm_extract_ps(v.v, 0));
    r += bits_as_float(_mm_extract_ps(v.v, 1));
    r += bits_as_float(_mm_extract_ps(v.v, 2));
    r += bits_as_float(_mm_extract_ps(v.v, 3));
    return r;
}

static FORCEINLINE float __reduce_min_float(__vec4_f v) {
    float r = bits_as_float(_mm_extract_ps(v.v, 0));
    float val = bits_as_float(_mm_extract_ps(v.v, 1));
    r = (r < val) ? r : val;
    val = bits_as_float(_mm_extract_ps(v.v, 2));
    r = (r < val) ? r : val;
    val = bits_as_float(_mm_extract_ps(v.v, 3));
    r = (r < val) ? r : val;
    return r;
}

static FORCEINLINE float __reduce_max_float(__vec4_f v) {
    float r = bits_as_float(_mm_extract_ps(v.v, 0));
    float val = bits_as_float(_mm_extract_ps(v.v, 1));
    r = (r > val) ? r : val;
    val = bits_as_float(_mm_extract_ps(v.v, 2));
    r = (r > val) ? r : val;
    val = bits_as_float(_mm_extract_ps(v.v, 3));
    r = (r > val) ? r : val;
    return r;
}

static FORCEINLINE double __reduce_add_double(__vec4_d v) {
    double r = _mm_extract_pd(v.v[0], 0);
    r += _mm_extract_pd(v.v[0], 1);
    r += _mm_extract_pd(v.v[1], 0);
    r += _mm_extract_pd(v.v[1], 1);
    return r;
}

static FORCEINLINE double __reduce_min_double(__vec4_d v) {
    double r = _mm_extract_pd(v.v[0], 0);
    r = (r < _mm_extract_pd(v.v[0], 1)) ? r : _mm_extract_pd(v.v[0], 1);
    r = (r < _mm_extract_pd(v.v[1], 0)) ? r : _mm_extract_pd(v.v[1], 0);
    r = (r < _mm_extract_pd(v.v[1], 1)) ? r : _mm_extract_pd(v.v[1], 1);
    return r;
}

static FORCEINLINE double __reduce_max_double(__vec4_d v) {
    double r = _mm_extract_pd(v.v[0], 0);
    r = (r > _mm_extract_pd(v.v[0], 1)) ? r : _mm_extract_pd(v.v[0], 1);
    r = (r > _mm_extract_pd(v.v[1], 0)) ? r : _mm_extract_pd(v.v[1], 0);
    r = (r > _mm_extract_pd(v.v[1], 1)) ? r : _mm_extract_pd(v.v[1], 1);
    return r;
}

static FORCEINLINE uint32_t __reduce_add_int32(__vec4_i32 v) {
    int32_t r = _mm_extract_epi32(v.v, 0);
    r += _mm_extract_epi32(v.v, 1);
    r += _mm_extract_epi32(v.v, 2);
    r += _mm_extract_epi32(v.v, 3);
    return r;
}

static FORCEINLINE int32_t __reduce_min_int32(__vec4_i32 v) {
    int32_t r = _mm_extract_epi32(v.v, 0);
    int32_t val = _mm_extract_epi32(v.v, 1);
    r = (r < val) ? r : val;
    val = _mm_extract_epi32(v.v, 2);
    r = (r < val) ? r : val;
    val = _mm_extract_epi32(v.v, 3);
    r = (r < val) ? r : val;
    return r;
}

static FORCEINLINE int32_t __reduce_max_int32(__vec4_i32 v) {
    int32_t r = _mm_extract_epi32(v.v, 0);
    int32_t val = _mm_extract_epi32(v.v, 1);
    r = (r > val) ? r : val;
    val = _mm_extract_epi32(v.v, 2);
    r = (r > val) ? r : val;
    val = _mm_extract_epi32(v.v, 3);
    r = (r > val) ? r : val;

    return r;
}

static FORCEINLINE uint32_t __reduce_add_uint32(__vec4_i32 v) {
    uint32_t r = _mm_extract_epi32(v.v, 0);
    r += _mm_extract_epi32(v.v, 1);
    r += _mm_extract_epi32(v.v, 2);
    r += _mm_extract_epi32(v.v, 3);
    return r;
}

static FORCEINLINE uint32_t __reduce_min_uint32(__vec4_i32 v) {
    uint32_t r = _mm_extract_epi32(v.v, 0);
    uint32_t val = _mm_extract_epi32(v.v, 1);
    r = (r < val) ? r : val;
    val = _mm_extract_epi32(v.v, 2);
    r = (r < val) ? r : val;
    val = _mm_extract_epi32(v.v, 3);
    r = (r < val) ? r : val;
    return r;
}

static FORCEINLINE uint32_t __reduce_max_uint32(__vec4_i32 v) {
    uint32_t r = _mm_extract_epi32(v.v, 0);
    uint32_t val = _mm_extract_epi32(v.v, 1);
    r = (r > val) ? r : val;
    val = _mm_extract_epi32(v.v, 2);
    r = (r > val) ? r : val;
    val = _mm_extract_epi32(v.v, 3);
    r = (r > val) ? r : val;
    return r;
}

static FORCEINLINE uint64_t __reduce_add_int64(__vec4_i64 v) {
    int64_t r = _mm_extract_epi64(v.v[0], 0);
    r += _mm_extract_epi64(v.v[0], 1);
    r += _mm_extract_epi64(v.v[1], 0);
    r += _mm_extract_epi64(v.v[1], 1);
    return r;
}

static FORCEINLINE int64_t __reduce_min_int64(__vec4_i64 v) {
    int64_t r = _mm_extract_epi64(v.v[0], 0);
    r = ((int64_t)_mm_extract_epi64(v.v[0], 1) < r) ? _mm_extract_epi64(v.v[0], 1) : r;
    r = ((int64_t)_mm_extract_epi64(v.v[1], 0) < r) ? _mm_extract_epi64(v.v[1], 0) : r;
    r = ((int64_t)_mm_extract_epi64(v.v[1], 1) < r) ? _mm_extract_epi64(v.v[1], 1) : r;
    return r;
}

static FORCEINLINE int64_t __reduce_max_int64(__vec4_i64 v) {
    int64_t r = _mm_extract_epi64(v.v[0], 0);
    r = ((int64_t)_mm_extract_epi64(v.v[0], 1) > r) ? _mm_extract_epi64(v.v[0], 1) : r;
    r = ((int64_t)_mm_extract_epi64(v.v[1], 0) > r) ? _mm_extract_epi64(v.v[1], 0) : r;
    r = ((int64_t)_mm_extract_epi64(v.v[1], 1) > r) ? _mm_extract_epi64(v.v[1], 1) : r;
    return r;
}

static FORCEINLINE uint64_t __reduce_add_uint64(__vec4_i64 v) {
    uint64_t r = _mm_extract_epi64(v.v[0], 0);
    r += _mm_extract_epi64(v.v[0], 1);
    r += _mm_extract_epi64(v.v[1], 0);
    r += _mm_extract_epi64(v.v[1], 1);
    return r;
}

static FORCEINLINE uint64_t __reduce_min_uint64(__vec4_i64 v) {
    uint64_t r = _mm_extract_epi64(v.v[0], 0);
    r = ((uint64_t)_mm_extract_epi64(v.v[0], 1) < r) ? _mm_extract_epi64(v.v[0], 1) : r;
    r = ((uint64_t)_mm_extract_epi64(v.v[1], 0) < r) ? _mm_extract_epi64(v.v[1], 0) : r;
    r = ((uint64_t)_mm_extract_epi64(v.v[1], 1) < r) ? _mm_extract_epi64(v.v[1], 1) : r;
    return r;
}

static FORCEINLINE uint64_t __reduce_max_uint64(__vec4_i64 v) {
    uint64_t r = _mm_extract_epi64(v.v[0], 0);
    r = ((uint64_t)_mm_extract_epi64(v.v[0], 1) > r) ? _mm_extract_epi64(v.v[0], 1) : r;
    r = ((uint64_t)_mm_extract_epi64(v.v[1], 0) > r) ? _mm_extract_epi64(v.v[1], 0) : r;
    r = ((uint64_t)_mm_extract_epi64(v.v[1], 1) > r) ? _mm_extract_epi64(v.v[1], 1) : r;
    return r;
}

///////////////////////////////////////////////////////////////////////////
// masked load/store

static FORCEINLINE __vec4_i8 __masked_load_8(void *p, 
                                             __vec4_i1 mask) {
    int8_t r[4];
    int8_t *ptr = (int8_t *)p;
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0)
        r[0] = ptr[0];
    m = _mm_extract_ps(mask.v, 1);
    if (m != 0)
        r[1] = ptr[1];
    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        r[2] = ptr[2];
    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        r[3] = ptr[3];

    return __vec4_i8(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE __vec4_i16 __masked_load_16(void *p, 
                                               __vec4_i1 mask) {
    int16_t r[4];
    int16_t *ptr = (int16_t *)p;

    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0)
        r[0] = ptr[0];

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0)
        r[1] = ptr[1];

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        r[2] = ptr[2];

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        r[3] = ptr[3];

    return __vec4_i16(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE __vec4_i32 __masked_load_32(void *p, 
                                               __vec4_i1 mask) {
    __m128i r = _mm_set_epi32(0, 0, 0, 0);
    int32_t *ptr = (int32_t *)p;
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0)
        r = _mm_insert_epi32(r, ptr[0], 0);

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0)
        r = _mm_insert_epi32(r, ptr[1], 1);

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        r = _mm_insert_epi32(r, ptr[2], 2);

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        r = _mm_insert_epi32(r, ptr[3], 3);

    return r;
}

static FORCEINLINE __vec4_i64 __masked_load_64(void *p, 
                                               __vec4_i1 mask) {
    uint64_t r[4];
    uint64_t *ptr = (uint64_t *)p;
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0)
        r[0] = ptr[0];

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0)
        r[1] = ptr[1];

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        r[2] = ptr[2];

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        r[3] = ptr[3];

    return __vec4_i64(r[0], r[1], r[2], r[3]);
}

static FORCEINLINE void __masked_store_8(void *p, __vec4_i8 val, 
                                         __vec4_i1 mask) {
    int8_t *ptr = (int8_t *)p;

    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0)
        ptr[0] = _mm_extract_epi8(val.v, 0);

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0)
        ptr[1] = _mm_extract_epi8(val.v, 1);

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        ptr[2] = _mm_extract_epi8(val.v, 2);

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        ptr[3] = _mm_extract_epi8(val.v, 3);
}

static FORCEINLINE void __masked_store_16(void *p, __vec4_i16 val,
                                          __vec4_i1 mask) {
    int16_t *ptr = (int16_t *)p;

    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0)
        ptr[0] = _mm_extract_epi16(val.v, 0);

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0)
        ptr[1] = _mm_extract_epi16(val.v, 1);

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        ptr[2] = _mm_extract_epi16(val.v, 2);

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        ptr[3] = _mm_extract_epi16(val.v, 3);
}

static FORCEINLINE void __masked_store_32(void *p, __vec4_i32 val, 
                                          __vec4_i1 mask) {
    int32_t *ptr = (int32_t *)p;
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0)
        ptr[0] = _mm_extract_epi32(val.v, 0);

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0)
        ptr[1] = _mm_extract_epi32(val.v, 1);

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        ptr[2] = _mm_extract_epi32(val.v, 2);

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        ptr[3] = _mm_extract_epi32(val.v, 3);
}

static FORCEINLINE void __masked_store_64(void *p, __vec4_i64 val, 
                                          __vec4_i1 mask) {
    int64_t *ptr = (int64_t *)p;
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) 
        ptr[0] = _mm_extract_epi64(val.v[0], 0);

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) 
        ptr[1] = _mm_extract_epi64(val.v[0], 1);

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) 
        ptr[2] = _mm_extract_epi64(val.v[1], 0);

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) 
        ptr[3] = _mm_extract_epi64(val.v[1], 1);
}

static FORCEINLINE void __masked_store_blend_8(void *p, __vec4_i8 val, 
                                               __vec4_i1 mask) {
    __masked_store_8(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_16(void *p, __vec4_i16 val, 
                                                __vec4_i1 mask) {
    __masked_store_16(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_32(void *p, __vec4_i32 val, 
                                                __vec4_i1 mask) {
    // FIXME: do a load, blendvps, store here...
    __masked_store_32(p, val, mask);
}

static FORCEINLINE void __masked_store_blend_64(void *p, __vec4_i64 val, 
                                                __vec4_i1 mask) {
    // FIXME: do a 2x (load, blendvps, store) here...
    __masked_store_64(p, val, mask);
}

///////////////////////////////////////////////////////////////////////////
// gather/scatter
// offsets * offsetScale is in bytes (for all of these)

template<typename RetVec, typename RetScalar>
static FORCEINLINE RetVec
lGatherBaseOffsets32(RetVec, RetScalar, unsigned char *p, __vec4_i32 offsets, 
                     uint32_t scale, __vec4_i32 constOffset, __vec4_i1 mask) {
    RetScalar r[4];
#if 1
    // "Fast gather" trick...
    offsets = __select(mask, offsets, __smear_i32(0));
    constOffset = __select(mask, constOffset, __smear_i32(0));

    int offset = scale * _mm_extract_epi32(offsets.v, 0) + _mm_extract_epi32(constOffset.v, 0);
    RetScalar *ptr = (RetScalar *)(p + offset);
    r[0] = *ptr;

    offset = scale * _mm_extract_epi32(offsets.v, 1) + _mm_extract_epi32(constOffset.v, 1);
    ptr = (RetScalar *)(p + offset);
    r[1] = *ptr;

    offset = scale * _mm_extract_epi32(offsets.v, 2) + _mm_extract_epi32(constOffset.v, 2);
    ptr = (RetScalar *)(p + offset);
    r[2] = *ptr;

    offset = scale * _mm_extract_epi32(offsets.v, 3) + _mm_extract_epi32(constOffset.v, 3);
    ptr = (RetScalar *)(p + offset);
    r[3] = *ptr;
#else
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        int offset = scale * _mm_extract_epi32(offsets.v, 0) + _mm_extract_epi32(constOffset.v, 0);
        RetScalar *ptr = (RetScalar *)(p + offset);
        r[0] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        int offset = scale * _mm_extract_epi32(offsets.v, 1) + _mm_extract_epi32(constOffset.v, 1);
        RetScalar *ptr = (RetScalar *)(p + offset);
        r[1] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        int offset = scale * _mm_extract_epi32(offsets.v, 2) + _mm_extract_epi32(constOffset.v, 2);
        RetScalar *ptr = (RetScalar *)(p + offset);
        r[2] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        int offset = scale * _mm_extract_epi32(offsets.v, 3) + _mm_extract_epi32(constOffset.v, 3);
        RetScalar *ptr = (RetScalar *)(p + offset);
        r[3] = *ptr;
    }
#endif
    return RetVec(r[0], r[1], r[2], r[3]);
}


template<typename RetVec, typename RetScalar>
static FORCEINLINE RetVec
lGatherBaseOffsets64(RetVec, RetScalar, unsigned char *p, __vec4_i64 offsets,
                     uint32_t scale, __vec4_i64 constOffset, __vec4_i1 mask) {
    RetScalar r[4];
#if 1
    // "Fast gather" trick...
    offsets = __select(mask, offsets, __smear_i64(0));
    constOffset = __select(mask, constOffset, __smear_i64(0));

    int64_t offset = scale * _mm_extract_epi64(offsets.v[0], 0) + _mm_extract_epi64(constOffset.v[0], 0);
    RetScalar *ptr = (RetScalar *)(p + offset);
    r[0] = *ptr;

    offset = scale * _mm_extract_epi64(offsets.v[0], 1) + _mm_extract_epi64(constOffset.v[0], 1);
    ptr = (RetScalar *)(p + offset);
    r[1] = *ptr;

    offset = scale * _mm_extract_epi64(offsets.v[1], 0) + _mm_extract_epi64(constOffset.v[1], 0);
    ptr = (RetScalar *)(p + offset);
    r[2] = *ptr;

    offset = scale * _mm_extract_epi64(offsets.v[1], 1) + _mm_extract_epi64(constOffset.v[1], 1);
    ptr = (RetScalar *)(p + offset);
    r[3] = *ptr;
#else
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        int64_t offset = scale * _mm_extract_epi64(offsets.v[0], 0) + _mm_extract_epi64(constOffset.v[0], 0);
        RetScalar *ptr = (RetScalar *)(p + offset);
        r[0] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        int64_t offset = scale * _mm_extract_epi64(offsets.v[0], 1) + _mm_extract_epi64(constOffset.v[0], 1);
        RetScalar *ptr = (RetScalar *)(p + offset);
        r[1] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        int64_t offset = scale * _mm_extract_epi64(offsets.v[1], 0) + _mm_extract_epi64(constOffset.v[1], 0);
        RetScalar *ptr = (RetScalar *)(p + offset);
        r[2] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        int64_t offset = scale * _mm_extract_epi64(offsets.v[1], 1) + _mm_extract_epi64(constOffset.v[1], 1);
        RetScalar *ptr = (RetScalar *)(p + offset);
        r[3] = *ptr;
    }
#endif
    return RetVec(r[0], r[1], r[2], r[3]);

}

static FORCEINLINE __vec4_i8
__gather_base_offsets32_i8(unsigned char *b, __vec4_i32 offsets,
                           uint32_t scale,  __vec4_i32 constOffset, __vec4_i1 mask) {
    return lGatherBaseOffsets32(__vec4_i8(), uint8_t(), b, offsets, scale, 
                                constOffset, mask);
}

static FORCEINLINE __vec4_i8
__gather_base_offsets64_i8(unsigned char *b, __vec4_i64 offsets,
                           uint32_t scale, __vec4_i64 constOffset, __vec4_i1 mask) {
    return lGatherBaseOffsets64(__vec4_i8(), uint8_t(), b, offsets, scale, 
                                constOffset, mask);
}

static FORCEINLINE __vec4_i16
__gather_base_offsets32_i16(unsigned char *b, __vec4_i32 offsets,
                            uint32_t scale, __vec4_i32 constOffset, __vec4_i1 mask) {
    return lGatherBaseOffsets32(__vec4_i16(), uint16_t(), b, offsets, scale, 
                                constOffset, mask);
}

static FORCEINLINE __vec4_i16
 __gather_base_offsets64_i16(unsigned char *b, __vec4_i64 offsets,
                             uint32_t scale, __vec4_i64 constOffset, __vec4_i1 mask) {
    return lGatherBaseOffsets64(__vec4_i16(), uint16_t(), b, offsets, scale, 
                                constOffset, mask);
}

static FORCEINLINE __vec4_i32
__gather_base_offsets32_i32(uint8_t *p, __vec4_i32 offsets, uint32_t scale,
                            __vec4_i32 constOffset, __vec4_i1 mask) {
    __m128i r = _mm_set_epi32(0, 0, 0, 0);
#if 1
    // "Fast gather"...
    offsets = __select(mask, offsets, __smear_i32(0));
    constOffset = __select(mask, constOffset, __smear_i32(0));

    int offset = scale * _mm_extract_epi32(offsets.v, 0) +
        _mm_extract_epi32(constOffset.v, 0);
    uint32_t *ptr = (uint32_t *)(p + offset);
    r = _mm_insert_epi32(r, *ptr, 0);

    offset = scale * _mm_extract_epi32(offsets.v, 1) +
        _mm_extract_epi32(constOffset.v, 1);
    ptr = (uint32_t *)(p + offset);
    r = _mm_insert_epi32(r, *ptr, 1);

    offset = scale * _mm_extract_epi32(offsets.v, 2) +
        _mm_extract_epi32(constOffset.v, 2);
    ptr = (uint32_t *)(p + offset);
    r = _mm_insert_epi32(r, *ptr, 2);

    offset = scale * _mm_extract_epi32(offsets.v, 3) +
        _mm_extract_epi32(constOffset.v, 3);
    ptr = (uint32_t *)(p + offset);
    r = _mm_insert_epi32(r, *ptr, 3);
#else
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        int offset = scale * _mm_extract_epi32(offsets.v, 0) +
            _mm_extract_epi32(constOffset.v, 0);
        uint32_t *ptr = (uint32_t *)(p + offset);
        r = _mm_insert_epi32(r, *ptr, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        int offset = scale * _mm_extract_epi32(offsets.v, 1) +
            _mm_extract_epi32(constOffset.v, 1);
        uint32_t *ptr = (uint32_t *)(p + offset);
        r = _mm_insert_epi32(r, *ptr, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        int offset = scale * _mm_extract_epi32(offsets.v, 2) +
            _mm_extract_epi32(constOffset.v, 2);
        uint32_t *ptr = (uint32_t *)(p + offset);
        r = _mm_insert_epi32(r, *ptr, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        int offset = scale * _mm_extract_epi32(offsets.v, 3) +
            _mm_extract_epi32(constOffset.v, 3);
        uint32_t *ptr = (uint32_t *)(p + offset);
        r = _mm_insert_epi32(r, *ptr, 3);
    }
#endif
    return r;
}

static FORCEINLINE __vec4_i32
__gather_base_offsets64_i32(unsigned char *p, __vec4_i64 offsets,
                            uint32_t scale, __vec4_i64 delta, __vec4_i1 mask) {
    return lGatherBaseOffsets64(__vec4_i32(), uint32_t(), p, offsets, scale, 
                                delta, mask);
}

static FORCEINLINE __vec4_i64
__gather_base_offsets32_i64(unsigned char *p, __vec4_i32 offsets,
                            uint32_t scale, __vec4_i32 delta, __vec4_i1 mask) {
    return lGatherBaseOffsets32(__vec4_i64(), uint64_t(), p, offsets, scale, 
                                delta, mask);
}

static FORCEINLINE __vec4_i64
__gather_base_offsets64_i64(unsigned char *p, __vec4_i64 offsets,
                            uint32_t scale, __vec4_i64 delta, __vec4_i1 mask) {
    return lGatherBaseOffsets64(__vec4_i64(), uint64_t(), p, offsets, scale, 
                                delta, mask);
}

template<typename RetVec, typename RetScalar>
static FORCEINLINE RetVec lGather32(RetVec, RetScalar, __vec4_i32 ptrs, 
                                    __vec4_i1 mask) {
    RetScalar r[4];
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        RetScalar *ptr = (RetScalar *)_mm_extract_epi32(ptrs.v, 0);
        r[0] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        RetScalar *ptr = (RetScalar *)_mm_extract_epi32(ptrs.v, 1);
        r[1] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        RetScalar *ptr = (RetScalar *)_mm_extract_epi32(ptrs.v, 2);
        r[2] = *ptr;
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        RetScalar *ptr = (RetScalar *)_mm_extract_epi32(ptrs.v, 3);
        r[3] = *ptr;
    }

    return RetVec(r[0], r[1], r[2], r[3]);
}

template<typename RetVec, typename RetScalar>
static FORCEINLINE RetVec lGather64(RetVec, RetScalar, __vec4_i64 ptrs, 
                                    __vec4_i1 mask) {
    RetScalar r[4];
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        RetScalar *ptr = (RetScalar *)_mm_extract_epi64(ptrs.v[0], 0);
        r[0] = *ptr;
    }
    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        RetScalar *ptr = (RetScalar *)_mm_extract_epi64(ptrs.v[0], 1);
        r[1] = *ptr;
    }
    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        RetScalar *ptr = (RetScalar *)_mm_extract_epi64(ptrs.v[1], 0);
        r[2] = *ptr;
    }
    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        RetScalar *ptr = (RetScalar *)_mm_extract_epi64(ptrs.v[1], 1);
        r[3] = *ptr;
    }
    return RetVec(r[0], r[1], r[2], r[3]);
}


static FORCEINLINE __vec4_i8 __gather32_i8(__vec4_i32 ptrs, __vec4_i1 mask) {
    return lGather32(__vec4_i8(), uint8_t(), ptrs, mask);
}

static FORCEINLINE __vec4_i8 __gather64_i8(__vec4_i64 ptrs, __vec4_i1 mask) {
    return lGather64(__vec4_i8(), uint8_t(), ptrs, mask);
}

static FORCEINLINE __vec4_i16 __gather32_i16(__vec4_i32 ptrs, __vec4_i1 mask) {
    return lGather32(__vec4_i16(), uint16_t(), ptrs, mask);
}

static FORCEINLINE __vec4_i16 __gather64_i16(__vec4_i64 ptrs, __vec4_i1 mask) {
    return lGather64(__vec4_i16(), uint16_t(), ptrs, mask);
}

static FORCEINLINE __vec4_i32 __gather32_i32(__vec4_i32 ptrs, __vec4_i1 mask) {
    __m128i r = _mm_set_epi32(0, 0, 0, 0);
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        int32_t *ptr = (int32_t *)_mm_extract_epi32(ptrs.v, 0);
        r = _mm_insert_epi32(r, *ptr, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        int32_t *ptr = (int32_t *)_mm_extract_epi32(ptrs.v, 1);
        r = _mm_insert_epi32(r, *ptr, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        int32_t *ptr = (int32_t *)_mm_extract_epi32(ptrs.v, 2);
        r = _mm_insert_epi32(r, *ptr, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        int32_t *ptr = (int32_t *)_mm_extract_epi32(ptrs.v, 3);
        r = _mm_insert_epi32(r, *ptr, 3);
    }

    return r;
}

static FORCEINLINE __vec4_i32 __gather64_i32(__vec4_i64 ptrs, __vec4_i1 mask) {
    __m128i r = _mm_set_epi32(0, 0, 0, 0);

    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        int32_t *ptr = (int32_t *)_mm_extract_epi64(ptrs.v[0], 0);
        r = _mm_insert_epi32(r, *ptr, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        int32_t *ptr = (int32_t *)_mm_extract_epi64(ptrs.v[0], 1);
        r = _mm_insert_epi32(r, *ptr, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        int32_t *ptr = (int32_t *)_mm_extract_epi64(ptrs.v[1], 0);
        r = _mm_insert_epi32(r, *ptr, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        int32_t *ptr = (int32_t *)_mm_extract_epi64(ptrs.v[1], 1);
        r = _mm_insert_epi32(r, *ptr, 3);
    }

    return r;
}

static FORCEINLINE __vec4_i64 __gather32_i64(__vec4_i32 ptrs, __vec4_i1 mask) {
    return lGather32(__vec4_i64(), uint64_t(), ptrs, mask);
}

static FORCEINLINE __vec4_i64 __gather64_i64(__vec4_i64 ptrs, __vec4_i1 mask) {
    return lGather64(__vec4_i64(), uint64_t(), ptrs, mask);
}

// scatter
  
#define SCATTER32_64(SUFFIX, TYPE, EXTRACT)                         \
static FORCEINLINE void                                             \
__scatter_base_offsets32_##SUFFIX (unsigned char *b, __vec4_i32 offsets, \
                                   uint32_t scale, __vec4_i32 constOffset, \
                                   __vec4_##SUFFIX val, __vec4_i1 mask) { \
    uint32_t m = _mm_extract_ps(mask.v, 0);                             \
    if (m != 0) {                                                       \
        TYPE *ptr = (TYPE *)(b + scale * _mm_extract_epi32(offsets.v, 0) + \
                             _mm_extract_epi32(constOffset.v, 0));      \
        *ptr = EXTRACT(val.v, 0);                                       \
    }                                                                   \
    m = _mm_extract_ps(mask.v, 1);                                      \
    if (m != 0) {                                                       \
        TYPE *ptr = (TYPE *)(b + scale * _mm_extract_epi32(offsets.v, 1) + \
                             _mm_extract_epi32(constOffset.v, 1));      \
        *ptr = EXTRACT(val.v, 1);                                       \
    }                                                                   \
    m = _mm_extract_ps(mask.v, 2);                                      \
    if (m != 0) {                                                       \
        TYPE *ptr = (TYPE *)(b + scale * _mm_extract_epi32(offsets.v, 2) + \
                             _mm_extract_epi32(constOffset.v, 2));      \
        *ptr = EXTRACT(val.v, 2);                                       \
    }                                                                   \
    m = _mm_extract_ps(mask.v, 3);                                      \
    if (m != 0) {                                                       \
        TYPE *ptr = (TYPE *)(b + scale * _mm_extract_epi32(offsets.v, 3) + \
                             _mm_extract_epi32(constOffset.v, 3));      \
        *ptr = EXTRACT(val.v, 3);                                       \
    }                                                                   \
}                                                                       \
static FORCEINLINE void                                                \
__scatter_base_offsets64_##SUFFIX(unsigned char *p, __vec4_i64 offsets, \
                                  uint32_t scale, __vec4_i64 constOffset, \
                                  __vec4_##SUFFIX val, __vec4_i1 mask) { \
    uint32_t m = _mm_extract_ps(mask.v, 0);                            \
    if (m != 0) {                                                      \
        int64_t offset = scale * _mm_extract_epi64(offsets.v[0], 0) +  \
            _mm_extract_epi64(constOffset.v[0], 0);                    \
        TYPE *ptr = (TYPE *)(p + offset);                              \
        *ptr = EXTRACT(val.v, 0);                                      \
    }                                                                  \
    m = _mm_extract_ps(mask.v, 1);                                     \
    if (m != 0) {                                                      \
        int64_t offset = scale * _mm_extract_epi64(offsets.v[0], 1) +  \
            _mm_extract_epi64(constOffset.v[0], 1);                    \
        TYPE *ptr = (TYPE *)(p + offset);                              \
        *ptr = EXTRACT(val.v, 1);                                      \
    }                                                                  \
    m = _mm_extract_ps(mask.v, 2);                                     \
    if (m != 0) {                                                      \
        int64_t offset = scale * _mm_extract_epi64(offsets.v[1], 0) +  \
            _mm_extract_epi64(constOffset.v[1], 0);                    \
        TYPE *ptr = (TYPE *)(p + offset);                              \
        *ptr = EXTRACT(val.v, 2);                                      \
    }                                                                  \
    m = _mm_extract_ps(mask.v, 3);                                     \
    if (m != 0) {                                                      \
        int64_t offset = scale * _mm_extract_epi64(offsets.v[1], 1) +  \
            _mm_extract_epi64(constOffset.v[1], 1);                    \
        TYPE *ptr = (TYPE *)(p + offset);                              \
        *ptr = EXTRACT(val.v, 3);                                      \
    }                                                                  \
}


SCATTER32_64(i8, int8_t, _mm_extract_epi8)
SCATTER32_64(i16, int16_t, _mm_extract_epi16)
SCATTER32_64(i32, int32_t, _mm_extract_epi32)


static FORCEINLINE void
__scatter_base_offsets32_i64(unsigned char *p, __vec4_i32 offsets, 
                             uint32_t scale, __vec4_i32 constOffset, __vec4_i64 val, 
                             __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        int32_t offset = scale * _mm_extract_epi32(offsets.v, 0) +
            _mm_extract_epi32(constOffset.v, 0);
        uint64_t *ptr = (uint64_t *)(p + offset);
        *ptr = _mm_extract_epi64(val.v[0], 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        int32_t offset = scale * _mm_extract_epi32(offsets.v, 1) +
            _mm_extract_epi32(constOffset.v, 1);
        uint64_t *ptr = (uint64_t *)(p + offset);
        *ptr = _mm_extract_epi64(val.v[0], 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        int32_t offset = scale * _mm_extract_epi32(offsets.v, 2) +
            _mm_extract_epi32(constOffset.v, 2);
        uint64_t *ptr = (uint64_t *)(p + offset);
        *ptr = _mm_extract_epi64(val.v[1], 0);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        int32_t offset = scale * _mm_extract_epi32(offsets.v, 3) +
            _mm_extract_epi32(constOffset.v, 3);
        uint64_t *ptr = (uint64_t *)(p + offset);
        *ptr = _mm_extract_epi64(val.v[1], 1);
    }
}

static FORCEINLINE void
__scatter_base_offsets64_i64(unsigned char *p, __vec4_i64 offsets, 
                             uint32_t scale, __vec4_i64 constOffset,
                             __vec4_i64 val, __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        int64_t offset = scale * _mm_extract_epi64(offsets.v[0], 0) +
            _mm_extract_epi64(constOffset.v[0], 0);
        uint64_t *ptr = (uint64_t *)(p + offset);
        *ptr = _mm_extract_epi64(val.v[0], 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        int64_t offset = scale * _mm_extract_epi64(offsets.v[0], 1) +
            _mm_extract_epi64(constOffset.v[0], 1);
        uint64_t *ptr = (uint64_t *)(p + offset);
        *ptr = _mm_extract_epi64(val.v[0], 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        int64_t offset = scale * _mm_extract_epi64(offsets.v[1], 0) +
            _mm_extract_epi64(constOffset.v[1], 0);
        uint64_t *ptr = (uint64_t *)(p + offset);
        *ptr = _mm_extract_epi64(val.v[1], 0);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        int64_t offset = scale * _mm_extract_epi64(offsets.v[1], 1) +
            _mm_extract_epi64(constOffset.v[1], 1);
        uint64_t *ptr = (uint64_t *)(p + offset);
        *ptr = _mm_extract_epi64(val.v[1], 1);
    }
}

static FORCEINLINE void __scatter32_i8(__vec4_i32 ptrs, __vec4_i8 val,
                                       __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        uint8_t *ptr = (uint8_t *)_mm_extract_epi32(ptrs.v, 0);
        *ptr = _mm_extract_epi8(val.v, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        uint8_t *ptr = (uint8_t *)_mm_extract_epi32(ptrs.v, 1);
        *ptr = _mm_extract_epi8(val.v, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        uint8_t *ptr = (uint8_t *)_mm_extract_epi32(ptrs.v, 2);
        *ptr = _mm_extract_epi8(val.v, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        uint8_t *ptr = (uint8_t *)_mm_extract_epi32(ptrs.v, 3);
        *ptr = _mm_extract_epi8(val.v, 3);
    }
}

static FORCEINLINE void __scatter64_i8(__vec4_i64 ptrs, __vec4_i8 val,
                                       __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        uint8_t *ptr = (uint8_t *)_mm_extract_epi64(ptrs.v[0], 0);
        *ptr = _mm_extract_epi8(val.v, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        uint8_t *ptr = (uint8_t *)_mm_extract_epi64(ptrs.v[0], 1);
        *ptr = _mm_extract_epi8(val.v, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        uint8_t *ptr = (uint8_t *)_mm_extract_epi64(ptrs.v[1], 0);
        *ptr = _mm_extract_epi8(val.v, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        uint8_t *ptr = (uint8_t *)_mm_extract_epi64(ptrs.v[1], 1);
        *ptr = _mm_extract_epi8(val.v, 3);
    }
}

static FORCEINLINE void __scatter32_i16(__vec4_i32 ptrs, __vec4_i16 val,
                                        __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        uint16_t *ptr = (uint16_t *)_mm_extract_epi32(ptrs.v, 0);
        *ptr = _mm_extract_epi16(val.v, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        uint16_t *ptr = (uint16_t *)_mm_extract_epi32(ptrs.v, 1);
        *ptr = _mm_extract_epi16(val.v, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        uint16_t *ptr = (uint16_t *)_mm_extract_epi32(ptrs.v, 2);
        *ptr = _mm_extract_epi16(val.v, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        uint16_t *ptr = (uint16_t *)_mm_extract_epi32(ptrs.v, 3);
        *ptr = _mm_extract_epi16(val.v, 3);
    }
}

static FORCEINLINE void __scatter64_i16(__vec4_i64 ptrs, __vec4_i16 val,
                                        __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        uint16_t *ptr = (uint16_t *)_mm_extract_epi64(ptrs.v[0], 0);
        *ptr = _mm_extract_epi16(val.v, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        uint16_t *ptr = (uint16_t *)_mm_extract_epi64(ptrs.v[0], 1);
        *ptr = _mm_extract_epi16(val.v, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        uint16_t *ptr = (uint16_t *)_mm_extract_epi64(ptrs.v[1], 0);
        *ptr = _mm_extract_epi16(val.v, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        uint16_t *ptr = (uint16_t *)_mm_extract_epi64(ptrs.v[1], 1);
        *ptr = _mm_extract_epi16(val.v, 3);
    }
}

static FORCEINLINE void __scatter32_i32(__vec4_i32 ptrs, __vec4_i32 val,
                                        __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        uint32_t *ptr = (uint32_t *)_mm_extract_epi32(ptrs.v, 0);
        *ptr = _mm_extract_epi32(val.v, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        uint32_t *ptr = (uint32_t *)_mm_extract_epi32(ptrs.v, 1);
        *ptr = _mm_extract_epi32(val.v, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        uint32_t *ptr = (uint32_t *)_mm_extract_epi32(ptrs.v, 2);
        *ptr = _mm_extract_epi32(val.v, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        uint32_t *ptr = (uint32_t *)_mm_extract_epi32(ptrs.v, 3);
        *ptr = _mm_extract_epi32(val.v, 3);
    }
}

static FORCEINLINE void __scatter64_i32(__vec4_i64 ptrs, __vec4_i32 val,
                                        __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        uint32_t *ptr = (uint32_t *)_mm_extract_epi64(ptrs.v[0], 0);
        *ptr = _mm_extract_epi32(val.v, 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        uint32_t *ptr = (uint32_t *)_mm_extract_epi64(ptrs.v[0], 1);
        *ptr = _mm_extract_epi32(val.v, 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        uint32_t *ptr = (uint32_t *)_mm_extract_epi64(ptrs.v[1], 0);
        *ptr = _mm_extract_epi32(val.v, 2);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        uint32_t *ptr = (uint32_t *)_mm_extract_epi64(ptrs.v[1], 1);
        *ptr = _mm_extract_epi32(val.v, 3);
    }
}

static FORCEINLINE void __scatter32_i64(__vec4_i32 ptrs, __vec4_i64 val, 
                                        __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        uint64_t *ptr = (uint64_t *)_mm_extract_epi32(ptrs.v, 0);
        *ptr = _mm_extract_epi64(val.v[0], 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        uint64_t *ptr = (uint64_t *)_mm_extract_epi32(ptrs.v, 1);
        *ptr = _mm_extract_epi64(val.v[0], 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        uint64_t *ptr = (uint64_t *)_mm_extract_epi32(ptrs.v, 2);
        *ptr = _mm_extract_epi64(val.v[1], 0);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        uint64_t *ptr = (uint64_t *)_mm_extract_epi32(ptrs.v, 3);
        *ptr = _mm_extract_epi64(val.v[1], 1);
    }
}

static FORCEINLINE void __scatter64_i64(__vec4_i64 ptrs, __vec4_i64 val,
                                        __vec4_i1 mask) {
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0) {
        uint64_t *ptr = (uint64_t *)_mm_extract_epi64(ptrs.v[0], 0);
        *ptr = _mm_extract_epi64(val.v[0], 0);
    }

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0) {
        uint64_t *ptr = (uint64_t *)_mm_extract_epi64(ptrs.v[0], 1);
        *ptr = _mm_extract_epi64(val.v[0], 1);
    }

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0) {
        uint64_t *ptr = (uint64_t *)_mm_extract_epi64(ptrs.v[1], 0);
        *ptr = _mm_extract_epi64(val.v[1], 0);
    }

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0) {
        uint64_t *ptr = (uint64_t *)_mm_extract_epi64(ptrs.v[1], 1);
        *ptr = _mm_extract_epi64(val.v[1], 1);
    }
}

///////////////////////////////////////////////////////////////////////////
// packed load/store

static FORCEINLINE int32_t __packed_load_active(int32_t *ptr, __vec4_i32 *val,
                                                __vec4_i1 mask) {
    int count = 0;
    uint32_t m = _mm_extract_ps(mask.v, 0); 
    if (m != 0)
        val->v = _mm_insert_epi32(val->v, ptr[count++], 0);

    m = _mm_extract_ps(mask.v, 1); 
    if (m != 0)
        val->v = _mm_insert_epi32(val->v, ptr[count++], 1);

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        val->v = _mm_insert_epi32(val->v, ptr[count++], 2);

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        val->v = _mm_insert_epi32(val->v, ptr[count++], 3);

    return count;
}

static FORCEINLINE int32_t __packed_store_active(int32_t *ptr, __vec4_i32 val,
                                                 __vec4_i1 mask) {
    int count = 0;
    uint32_t m = _mm_extract_ps(mask.v, 0);
    if (m != 0)
        ptr[count++] = _mm_extract_epi32(val.v, 0);

    m = _mm_extract_ps(mask.v, 1);
    if (m != 0)
        ptr[count++] = _mm_extract_epi32(val.v, 1);

    m = _mm_extract_ps(mask.v, 2);
    if (m != 0)
        ptr[count++] = _mm_extract_epi32(val.v, 2);

    m = _mm_extract_ps(mask.v, 3);
    if (m != 0)
        ptr[count++] = _mm_extract_epi32(val.v, 3);

    return count;
}

static FORCEINLINE int32_t __packed_load_active(uint32_t *ptr, __vec4_i32 *val,
                                                __vec4_i1 mask) {
    return __packed_load_active((int32_t *)ptr, val, mask);
}

static FORCEINLINE int32_t __packed_store_active(uint32_t *ptr, __vec4_i32 val,
                                                 __vec4_i1 mask) {
    return __packed_store_active((int32_t *)ptr, val, mask);
}


///////////////////////////////////////////////////////////////////////////
// aos/soa

// FIXME: these all are correct but could be much more efficient with
// actual use of SSE shuffles and the like

static FORCEINLINE void __soa_to_aos3_float(__vec4_f v0, __vec4_f v1, __vec4_f v2,
                                            float *ptr) {
    for (int i = 0; i < 4; ++i) {
        *ptr++ = __extract_element(v0, i);
        *ptr++ = __extract_element(v1, i);
        *ptr++ = __extract_element(v2, i);
    }
}

static FORCEINLINE void __aos_to_soa3_float(float *ptr, __vec4_f *out0, 
                                            __vec4_f *out1, __vec4_f *out2) {
    for (int i = 0; i < 4; ++i) {
        __insert_element(out0, i, *ptr++);
        __insert_element(out1, i, *ptr++);
        __insert_element(out2, i, *ptr++);
    }
}

static FORCEINLINE void __soa_to_aos4_float(__vec4_f v0, __vec4_f v1, __vec4_f v2,
                                            __vec4_f v3, float *ptr) {
    for (int i = 0; i < 4; ++i) {
        *ptr++ = __extract_element(v0, i);
        *ptr++ = __extract_element(v1, i);
        *ptr++ = __extract_element(v2, i);
        *ptr++ = __extract_element(v3, i);
    }
}

static FORCEINLINE void __aos_to_soa4_float(float *ptr, __vec4_f *out0, __vec4_f *out1,
                                            __vec4_f *out2, __vec4_f *out3) {
    for (int i = 0; i < 4; ++i) {
        __insert_element(out0, i, *ptr++);
        __insert_element(out1, i, *ptr++);
        __insert_element(out2, i, *ptr++);
        __insert_element(out3, i, *ptr++);
    }
}

///////////////////////////////////////////////////////////////////////////
// prefetch

static FORCEINLINE void __prefetch_read_uniform_1(unsigned char *ptr) {
    _mm_prefetch((char *)ptr, _MM_HINT_T0);
}

static FORCEINLINE void __prefetch_read_uniform_2(unsigned char *ptr) {
    _mm_prefetch((char *)ptr, _MM_HINT_T1);
}

static FORCEINLINE void __prefetch_read_uniform_3(unsigned char *ptr) {
    _mm_prefetch((char *)ptr, _MM_HINT_T2);
}

static FORCEINLINE void __prefetch_read_uniform_nt(unsigned char *ptr) {
    _mm_prefetch((char *)ptr, _MM_HINT_NTA);
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
