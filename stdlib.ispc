// -*- mode: c++ -*-
/*
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

/** @file stdlib.ispc

    @brief Portion of the ispc standard library implementation that's in
           ispc code 
*/

#ifdef ISPC_TARGET_GENERIC
#define IntMaskType bool
#define UIntMaskType bool
#else
#define IntMaskType int32
#define UIntMaskType unsigned int32
#endif

///////////////////////////////////////////////////////////////////////////
// Low level primitives

__declspec(safe,cost0)
static inline float floatbits(unsigned int a) {
    return __floatbits_varying_int32(a);
}

__declspec(safe,cost0)
static inline uniform float floatbits(uniform unsigned int a) {
    return __floatbits_uniform_int32(a);
}

__declspec(safe,cost0)
static inline float floatbits(int a) {
    return __floatbits_varying_int32(a);
}

__declspec(safe,cost0)
static inline uniform float floatbits(uniform int a) {
    return __floatbits_uniform_int32(a);
}

__declspec(safe,cost0)
static inline double doublebits(unsigned int64 a) {
    return __doublebits_varying_int64(a);
}

__declspec(safe,cost0)
static inline uniform double doublebits(uniform unsigned int64 a) {
    return __doublebits_uniform_int64(a);
}

__declspec(safe,cost0)
static inline unsigned int intbits(float a) {
    return __intbits_varying_float(a);
}

__declspec(safe,cost0)
static inline uniform unsigned int intbits(uniform float a) {
    return __intbits_uniform_float(a);
}

__declspec(safe,cost0)
static inline unsigned int64 intbits(double d) {
    return __intbits_varying_double(d);
}

__declspec(safe,cost0)
static inline uniform unsigned int64 intbits(uniform double d) {
    return __intbits_uniform_double(d);
}

__declspec(safe)
static inline float broadcast(float v, uniform int i) {
    return __broadcast_float(v, i);
}

__declspec(safe)
static inline int8 broadcast(int8 v, uniform int i) {
    return __broadcast_i8(v, i);
}

__declspec(safe)
static inline int16 broadcast(int16 v, uniform int i) {
    return __broadcast_i16(v, i);
}

__declspec(safe) 
static inline int32 broadcast(int32 v, uniform int i) {
    return __broadcast_i32(v, i);
}

__declspec(safe) 
static inline double broadcast(double v, uniform int i) {
    return __broadcast_double(v, i);
}

__declspec(safe) 
static inline int64 broadcast(int64 v, uniform int i) {
    return __broadcast_i64(v, i);
}

__declspec(safe) 
static inline float rotate(float v, uniform int i) {
    return __rotate_float(v, i);
}

__declspec(safe) 
static inline int8 rotate(int8 v, uniform int i) {
    return __rotate_i8(v, i);
}

__declspec(safe) 
static inline int16 rotate(int16 v, uniform int i) {
    return __rotate_i16(v, i);
}

__declspec(safe) 
static inline int32 rotate(int32 v, uniform int i) {
    return __rotate_i32(v, i);
}

__declspec(safe) 
static inline double rotate(double v, uniform int i) {
    return __rotate_double(v, i);
}

__declspec(safe) 
static inline int64 rotate(int64 v, uniform int i) {
    return __rotate_i64(v, i);
}

__declspec(safe) 
static inline float shuffle(float v, int i) {
    return __shuffle_float(v, i);
}

__declspec(safe) 
static inline int8 shuffle(int8 v, int i) {
    return __shuffle_i8(v, i);
}

__declspec(safe) 
static inline int16 shuffle(int16 v, int i) {
    return __shuffle_i16(v, i);
}

__declspec(safe) 
static inline int32 shuffle(int32 v, int i) {
    return __shuffle_i32(v, i);
}

__declspec(safe) 
static inline double shuffle(double v, int i) {
    return __shuffle_double(v, i);
}

__declspec(safe) 
static inline int64 shuffle(int64 v, int i) {
    return __shuffle_i64(v, i);
}

__declspec(safe) 
static inline float shuffle(float v0, float v1, int i) {
    return __shuffle2_float(v0, v1, i);
}

__declspec(safe) 
static inline int8 shuffle(int8 v0, int8 v1, int i) {
    return __shuffle2_i8(v0, v1, i);
}

__declspec(safe) 
static inline int16 shuffle(int16 v0, int16 v1, int i) {
    return __shuffle2_i16(v0, v1, i);
}

__declspec(safe) 
static inline int32 shuffle(int32 v0, int32 v1, int i) {
    return __shuffle2_i32(v0, v1, i);
}

__declspec(safe) 
static inline double shuffle(double v0, double v1, int i) {
    return __shuffle2_double(v0, v1, i);
}

__declspec(safe) 
static inline int64 shuffle(int64 v0, int64 v1, int i) {
    return __shuffle2_i64(v0, v1, i);
}

// x[i]
__declspec(safe,cost1) 
static inline uniform float extract(float x, uniform int i) {
    return floatbits(__extract_int32((int)intbits(x), i));
}

__declspec(safe,cost1) 
static inline uniform int8 extract(int8 x, uniform int i) {
    return __extract_int8(x, i);
}

__declspec(safe,cost1) 
static inline uniform unsigned int8 extract(unsigned int8 x, uniform int i) {
    return __extract_int8(x, (unsigned int)i);
}

__declspec(safe,cost1) 
static inline uniform int16 extract(int16 x, uniform int i) {
    return __extract_int16(x, i);
}

__declspec(safe,cost1) 
static inline uniform unsigned int16 extract(unsigned int16 x, uniform int i) {
    return __extract_int16(x, (unsigned int)i);
}

__declspec(safe,cost1) 
static inline uniform int32 extract(int32 x, uniform int i) {
    return __extract_int32(x, i);
}

__declspec(safe,cost1) 
static inline uniform unsigned int32 extract(unsigned int32 x, uniform int i) {
    return __extract_int32(x, (unsigned int)i);
}

__declspec(safe,cost1) 
static inline uniform double extract(double x, uniform int i) {
    return doublebits(__extract_int64((int64)intbits(x), i));
}

__declspec(safe,cost1) 
static inline uniform int64 extract(int64 x, uniform int i) {
    return __extract_int64(x, i);
}

__declspec(safe,cost1) 
static inline uniform unsigned int64 extract(unsigned int64 x, uniform int i) {
    return __extract_int64(x, (unsigned int)i);
}

// x[i] = v
__declspec(safe,cost1) 
static inline float insert(float x, uniform int i, uniform float v) {
    return floatbits(__insert_int32((int)intbits(x), i, (int)intbits(v)));
}

__declspec(safe,cost1) 
static inline int8 insert(int8 x, uniform int i, uniform int8 v) {
    return __insert_int8(x, i, v);
}

__declspec(safe,cost1) 
static inline unsigned int8 insert(unsigned int8 x, uniform int i, 
                                    uniform unsigned int8 v) {
    return __insert_int8(x, (unsigned int)i, v);
}

__declspec(safe,cost1) 
static inline int16 insert(int16 x, uniform int i, uniform int16 v) {
    return __insert_int16(x, i, v);
}

__declspec(safe,cost1) 
static inline unsigned int16 insert(unsigned int16 x, uniform int i, 
                                    uniform unsigned int16 v) {
    return __insert_int16(x, (unsigned int)i, v);
}

__declspec(safe,cost1) 
static inline int32 insert(int32 x, uniform int i, uniform int32 v) {
    return __insert_int32(x, i, v);
}

__declspec(safe,cost1) 
static inline unsigned int32 insert(unsigned int32 x, uniform int i, 
                                    uniform unsigned int32 v) {
    return __insert_int32(x, (unsigned int)i, v);
}

__declspec(safe,cost1) 
static inline double insert(double x, uniform int i, uniform double v) {
    return doublebits(__insert_int64((int64)intbits(x), i, (int64)intbits(v)));
}

__declspec(safe,cost1) 
static inline int64 insert(int64 x, uniform int i, uniform int64 v) {
    return __insert_int64(x, i, v);
}

__declspec(safe,cost1) 
static inline unsigned int64 insert(unsigned int64 x, uniform int i, 
                                    uniform unsigned int64 v) {
    return __insert_int64(x, (unsigned int)i, v);
}

__declspec(safe,cost1) 
static inline uniform int32 sign_extend(uniform bool v) {
    return __sext_uniform_bool(v);
}

__declspec(safe,cost1) 
static inline int32 sign_extend(bool v) {
    return __sext_varying_bool(v);
}

__declspec(safe) 
static inline uniform bool any(bool v) {
    // We only care about whether "any" is true for the active program instances,
    // so we have to make v with the current program mask.
#ifdef ISPC_TARGET_GENERIC
    return __movmsk(v & __mask) != 0;
#else
    return __movmsk(__sext_varying_bool(v) & __mask) != 0;
#endif
}

__declspec(safe) 
static inline uniform bool all(bool v) {
    // As with any(), we need to explicitly mask v with the current program mask
    // so we're only looking at the current lanes
#ifdef ISPC_TARGET_GENERIC
    bool match = ((v & __mask) == __mask);
#else
    int32 match = __sext_varying_bool((__sext_varying_bool(v) & __mask) == __mask);
#endif
    return __movmsk(match) == (1 << programCount) - 1;
}

__declspec(safe) 
static inline uniform int32 popcnt(uniform int32 v) {
    return __popcnt_int32(v);
}

__declspec(safe) 
static inline uniform int popcnt(uniform int64 v) {
    return (int32)__popcnt_int64(v);
}

__declspec(safe) 
static inline int popcnt(int v) {
    int r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, popcnt(extract(v, i)));
    return __mask ? r : 0;
}

__declspec(safe) 
static inline int popcnt(int64 v) {
    int r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, popcnt(extract(v, i)));
    return __mask ? r : 0;
}

__declspec(safe) 
static inline uniform int popcnt(bool v) {
    // As with any() and all(), only count across the active lanes
#ifdef ISPC_TARGET_GENERIC
    return __popcnt_int32(__movmsk(v & __mask));
#else
    return __popcnt_int32(__movmsk(__sext_varying_bool(v) & __mask));
#endif
}

__declspec(safe) 
static inline uniform int lanemask() {
    return __movmsk(__mask);
}

///////////////////////////////////////////////////////////////////////////
// memcpy/memmove/memset

static inline void memcpy(void * uniform dst, void * uniform src, 
                          uniform int32 count) {
    __memcpy32((int8 * uniform)dst, (int8 * uniform)src, count);
}

static inline void memcpy64(void * uniform dst, void * uniform src, 
                          uniform int64 count) {
    __memcpy64((int8 * uniform)dst, (int8 * uniform)src, count);
}

static inline void memcpy(void * varying dst, void * varying src, 
                          int32 count) {
    void * uniform da[programCount];
    void * uniform sa[programCount];

    da[programIndex] = dst;
    sa[programIndex] = src;

    __foreach_active (i) {
        void * uniform d = da[i], * uniform s = sa[i];
        __memcpy32((int8 * uniform)d, (int8 * uniform)s, extract(count, i));
    }
}

static inline void memcpy64(void * varying dst, void * varying src, 
                            int64 count) {
    void * uniform da[programCount];
    void * uniform sa[programCount];

    da[programIndex] = dst;
    sa[programIndex] = src;

    __foreach_active (i) {
        void * uniform d = da[i], * uniform s = sa[i];
        __memcpy64((int8 * uniform)d, (int8 * uniform)s, extract(count, i));
    }
}

static inline void memmove(void * uniform dst, void * uniform src, 
                          uniform int32 count) {
    __memmove32((int8 * uniform)dst, (int8 * uniform)src, count);
}

static inline void memmove64(void * uniform dst, void * uniform src, 
                             uniform int64 count) {
    __memmove64((int8 * uniform)dst, (int8 * uniform)src, count);
}

static inline void memmove(void * varying dst, void * varying src, 
                          int32 count) {
    void * uniform da[programCount];
    void * uniform sa[programCount];

    da[programIndex] = dst;
    sa[programIndex] = src;

    __foreach_active (i) {
        void * uniform d = da[i], * uniform s = sa[i];
        __memmove32((int8 * uniform)d, (int8 * uniform)s, extract(count, i));
    }
}

static inline void memmove64(void * varying dst, void * varying src, 
                             int64 count) {
    void * uniform da[programCount];
    void * uniform sa[programCount];

    da[programIndex] = dst;
    sa[programIndex] = src;

    __foreach_active (i) {
        void * uniform d = da[i], * uniform s = sa[i];
        __memmove64((int8 * uniform)d, (int8 * uniform)s, extract(count, i));
    }
}

static inline void memset(void * uniform ptr, uniform int8 val, 
                          uniform int32 count) {
    __memset32((int8 * uniform)ptr, val, count);
}

static inline void memset64(void * uniform ptr, uniform int8 val, 
                          uniform int64 count) {
    __memset64((int8 * uniform)ptr, val, count);
}

static inline void memset(void * varying ptr, int8 val, int32 count) {
    void * uniform pa[programCount];
    pa[programIndex] = ptr;

    __foreach_active (i) {
        __memset32((int8 * uniform)pa[i], extract(val, i), extract(count, i));
    }
}

static inline void memset64(void * varying ptr, int8 val, int64 count) {
    void * uniform pa[programCount];
    pa[programIndex] = ptr;

    __foreach_active (i) {
        __memset64((int8 * uniform)pa[i], extract(val, i), extract(count, i));
    }
}

///////////////////////////////////////////////////////////////////////////
// count leading/trailing zeros

__declspec(safe,cost1) 
static inline uniform unsigned int32
count_leading_zeros(uniform unsigned int32 v) {
    return __count_leading_zeros_i32(v);
}

__declspec(safe,cost1) 
static inline uniform unsigned int64
count_leading_zeros(uniform unsigned int64 v) {
    return __count_leading_zeros_i64(v);
}

__declspec(safe,cost1) 
static inline uniform unsigned int32
count_trailing_zeros(uniform unsigned int32 v) {
    return __count_trailing_zeros_i32(v);
}

__declspec(safe,cost1) 
static inline uniform unsigned int64
count_trailing_zeros(uniform unsigned int64 v) {
    return __count_trailing_zeros_i64(v);
}

__declspec(safe,cost1) 
static inline uniform int32
count_leading_zeros(uniform int32 v) {
    return __count_leading_zeros_i32(v);
}

__declspec(safe,cost1) 
static inline uniform int64
count_leading_zeros(uniform int64 v) {
    return __count_leading_zeros_i64(v);
}

__declspec(safe,cost1) 
static inline uniform int32
count_trailing_zeros(uniform int32 v) {
    return __count_trailing_zeros_i32(v);
}

__declspec(safe,cost1) 
static inline uniform int64
count_trailing_zeros(uniform int64 v) {
    return __count_trailing_zeros_i64(v);
}

__declspec(safe) 
static inline unsigned int32
count_leading_zeros(unsigned int32 v) {
    unsigned int32 r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, __count_leading_zeros_i32(extract(v, i)));
    return r;
}

__declspec(safe) 
static inline unsigned int64
count_leading_zeros(unsigned int64 v) {
    unsigned int64 r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, __count_leading_zeros_i64(extract(v, i)));
    return r;
}

__declspec(safe) 
static inline unsigned int32
count_trailing_zeros(unsigned int32 v) {
    unsigned int32 r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, __count_trailing_zeros_i32(extract(v, i)));
    return r;
}

__declspec(safe) 
static inline unsigned int64
count_trailing_zeros(unsigned int64 v) {
    unsigned int64 r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, __count_trailing_zeros_i64(extract(v, i)));
    return r;
}

__declspec(safe) 
static inline int32
count_leading_zeros(int32 v) {
    int32 r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, __count_leading_zeros_i32(extract(v, i)));
    return r;
}

__declspec(safe) 
static inline int64
count_leading_zeros(int64 v) {
    int64 r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, __count_leading_zeros_i64(extract(v, i)));
    return r;
}

__declspec(safe) 
static inline int32
count_trailing_zeros(int32 v) {
    int32 r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, __count_trailing_zeros_i32(extract(v, i)));
    return r;
}

__declspec(safe) 
static inline int64
count_trailing_zeros(int64 v) {
    int64 r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, __count_trailing_zeros_i64(extract(v, i)));
    return r;
}

///////////////////////////////////////////////////////////////////////////
// AOS/SOA conversion

static inline void
aos_to_soa3(uniform float a[], varying float * uniform v0, 
            varying float * uniform v1, varying float * uniform v2) {
    __aos_to_soa3_float(a, v0, v1, v2);
}

static inline void
soa_to_aos3(float v0, float v1, float v2, uniform float a[]) {
    __soa_to_aos3_float(v0, v1, v2, a);
}

static inline void
aos_to_soa4(uniform float a[], varying float * uniform v0,
            varying float * uniform v1, varying float * uniform v2,
            varying float * uniform v3) {
    __aos_to_soa4_float(a, v0, v1, v2, v3);
}

static inline void
soa_to_aos4(float v0, float v1, float v2, float v3, uniform float a[]) {
    __soa_to_aos4_float(v0, v1, v2, v3, a);
}

static inline void
aos_to_soa3(uniform int32 a[], varying int32 * uniform v0,
            varying int32 * uniform v1, varying int32 * uniform v2) {
    aos_to_soa3((uniform float * uniform)a, (varying float * uniform)v0, 
                (varying float * uniform)v1, (varying float * uniform)v2);
}

static inline void
soa_to_aos3(int32 v0, int32 v1, int32 v2, uniform int32 a[]) {
    soa_to_aos3(floatbits(v0), floatbits(v1), floatbits(v2),
                (uniform float * uniform)a);
}

static inline void
aos_to_soa4(uniform int32 a[], varying int32 * uniform v0, 
            varying int32 * uniform v1, varying int32 * uniform v2, 
            varying int32 * uniform v3) {
    aos_to_soa4((uniform float * uniform)a, (varying float * uniform )v0, 
                (varying float * uniform)v1, (varying float * uniform)v2, 
                (varying float * uniform)v3);
}

static inline void
soa_to_aos4(int32 v0, int32 v1, int32 v2, int32 v3, uniform int32 a[]) {
    soa_to_aos4(floatbits(v0), floatbits(v1), floatbits(v2), floatbits(v3), 
                (uniform float * uniform)a);
}

///////////////////////////////////////////////////////////////////////////
// Prefetching

__declspec(safe,cost1) 
static inline void prefetch_l1(const void * uniform ptr) {
    __prefetch_read_uniform_1((uniform int8 * uniform)ptr);
}

__declspec(safe,cost1) 
static inline void prefetch_l2(const void * uniform ptr) {
    __prefetch_read_uniform_2((uniform int8 * uniform)ptr);
}

__declspec(safe,cost1) 
static inline void prefetch_l3(const void * uniform ptr) {
    __prefetch_read_uniform_3((uniform int8 * uniform)ptr);
}

__declspec(safe,cost1) 
static inline void prefetch_nt(const void * uniform ptr) {
     __prefetch_read_uniform_nt((uniform int8 * uniform)ptr);
}

static inline void prefetch_l1(const void * varying ptr) {
    const void * uniform ptrArray[programCount];
    ptrArray[programIndex] = ptr;

    __foreach_active (i) {
        const void * uniform p = ptrArray[i];
        prefetch_l1(p);
    }
}

static inline void prefetch_l2(const void * varying ptr) {
    const void * uniform ptrArray[programCount];
    ptrArray[programIndex] = ptr;

    __foreach_active (i) {
        const void * uniform p = ptrArray[i];
        prefetch_l2(p);
    }
}

static inline void prefetch_l3(const void * varying ptr) {
    const void * uniform ptrArray[programCount];
    ptrArray[programIndex] = ptr;

    __foreach_active (i) {
        const void * uniform p = ptrArray[i];
        prefetch_l3(p);
    }
}

static inline void prefetch_nt(const void * varying ptr) {
    const void * uniform ptrArray[programCount];
    ptrArray[programIndex] = ptr;

    __foreach_active (i) {
        const void * uniform p = ptrArray[i];
        prefetch_nt(p);
    }
}

///////////////////////////////////////////////////////////////////////////
// non-short-circuiting alternatives

__declspec(safe,cost1)
static inline bool and(bool a, bool b) {
    return a && b;
}

__declspec(safe,cost1)
static inline uniform bool and(uniform bool a, uniform bool b) {
    return a && b;
}

__declspec(safe,cost1)
static inline bool or(bool a, bool b) {
    return a || b;
}

__declspec(safe,cost1)
static inline uniform bool or(uniform bool a, uniform bool b) {
    return a || b;
}

__declspec(safe,cost1)
static inline int8 select(bool c, int8 a, int8 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline int8 select(uniform bool c, int8 a, int8 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline uniform int8 select(uniform bool c, uniform int8 a,
                                  uniform int8 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline int16 select(bool c, int16 a, int16 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline int16 select(uniform bool c, int16 a, int16 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline uniform int16 select(uniform bool c, uniform int16 a,
                                   uniform int16 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline int32 select(bool c, int32 a, int32 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline int32 select(uniform bool c, int32 a, int32 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline uniform int32 select(uniform bool c, uniform int32 a,
                                   uniform int32 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline int64 select(bool c, int64 a, int64 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline int64 select(uniform bool c, int64 a, int64 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline uniform int64 select(uniform bool c, uniform int64 a,
                                   uniform int64 b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline float select(bool c, float a, float b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline float select(uniform bool c, float a, float b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline uniform float select(uniform bool c, uniform float a,
                                   uniform float b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline double select(bool c, double a, double b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline double select(uniform bool c, double a, double b) {
    return c ? a : b;
}

__declspec(safe,cost1)
static inline uniform double select(uniform bool c, uniform double a,
                                    uniform double b) {
    return c ? a : b;
}

///////////////////////////////////////////////////////////////////////////
// Horizontal ops / reductions

__declspec(safe) 
static inline uniform float reduce_add(float x) {
    // zero the lanes where the mask is off
    return __reduce_add_float(__mask ? x : 0.);
}


__declspec(safe) 
static inline uniform float reduce_min(float v) {
    // For the lanes where the mask is off, replace the given value with
    // infinity, so that it doesn't affect the result.
    int iflt_max = 0x7f800000; // infinity
    // Must use __floatbits_varying_int32, not floatbits(), since with the
    // latter the current mask enters into the returned result...
    return __reduce_min_float(__mask ? v : __floatbits_varying_int32(iflt_max));
}

__declspec(safe) 
static inline uniform float reduce_max(float v) {
    // For the lanes where the mask is off, replace the given value with
    // negative infinity, so that it doesn't affect the result.
    const int iflt_neg_max = 0xff800000; // -infinity
    // Must use __floatbits_varying_int32, not floatbits(), since with the
    // latter the current mask enters into the returned result...
    return __reduce_max_float(__mask ? v : __floatbits_varying_int32(iflt_neg_max));
}

__declspec(safe) 
static inline uniform int reduce_add(int x) {
    // Zero out the values for lanes that aren't running
    return __reduce_add_int32(__mask ? x : 0);
}

__declspec(safe) 
static inline uniform int reduce_min(int v) {
    // Set values for non-running lanes to the maximum integer value so
    // they don't affect the result.
    int int_max = 0x7fffffff;
    return __reduce_min_int32(__mask ? v : int_max);
}

__declspec(safe) 
static inline uniform int reduce_max(int v) {
    // Set values for non-running lanes to the minimum integer value so
    // they don't affect the result.
    int int_min = 0x80000000;
    return __reduce_max_int32(__mask ? v : int_min);
}

__declspec(safe) 
static inline uniform unsigned int reduce_add(unsigned int x) {
    // Set values for non-running lanes to zero so they don't affect the
    // result.
    return __reduce_add_uint32(__mask ? x : 0);
}

__declspec(safe) 
static inline uniform unsigned int reduce_min(unsigned int v) {
    // Set values for non-running lanes to the maximum unsigned integer
    // value so they don't affect the result.
    unsigned int uint_max = 0xffffffff;
    return __reduce_min_uint32(__mask ? v : uint_max);
}

__declspec(safe) 
static inline uniform unsigned int reduce_max(unsigned int v) {
    // Set values for non-running lanes to zero so they don't affect the
    // result.
    return __reduce_max_uint32(__mask ? v : 0);
}

__declspec(safe) 
static inline uniform double reduce_add(double x) {
    // zero the lanes where the mask is off
    return __reduce_add_double(__mask ? x : 0.);
}

__declspec(safe) 
static inline uniform double reduce_min(double v) {
    int64 iflt_max = 0x7ff0000000000000; // infinity
    // Must use __doublebits_varying_int64, not doublebits(), since with the
    // latter the current mask enters into the returned result...
    return __reduce_min_double(__mask ? v : __doublebits_varying_int64(iflt_max));
}

__declspec(safe) 
static inline uniform double reduce_max(double v) {
    const int64 iflt_neg_max = 0xfff0000000000000; // -infinity
    // Must use __doublebits_varying_int64, not doublebits(), since with the
    // latter the current mask enters into the returned result...
    return __reduce_max_double(__mask ? v : __doublebits_varying_int64(iflt_neg_max));
}

__declspec(safe) 
static inline uniform int64 reduce_add(int64 x) {
    // Zero out the values for lanes that aren't running
    return __reduce_add_int64(__mask ? x : 0);
}

__declspec(safe) 
static inline uniform int64 reduce_min(int64 v) {
    // Set values for non-running lanes to the maximum integer value so
    // they don't affect the result.
    int64 int_max = 0x7fffffffffffffff;
    return __reduce_min_int64(__mask ? v : int_max);
}

__declspec(safe) 
static inline uniform int64 reduce_max(int64 v) {
    // Set values for non-running lanes to the minimum integer value so
    // they don't affect the result.
    int64 int_min = 0x8000000000000000;
    return __reduce_max_int64(__mask ? v : int_min);
}

__declspec(safe) 
static inline uniform unsigned int64 reduce_add(unsigned int64 x) {
    // Set values for non-running lanes to zero so they don't affect the
    // result.
    return __reduce_add_int64(__mask ? x : 0);
}

__declspec(safe) 
static inline uniform unsigned int64 reduce_min(unsigned int64 v) {
    // Set values for non-running lanes to the maximum unsigned integer
    // value so they don't affect the result.
    unsigned int64 uint_max = 0xffffffffffffffff;
    return __reduce_min_uint64(__mask ? v : uint_max);
}

__declspec(safe) 
static inline uniform unsigned int64 reduce_max(unsigned int64 v) {
    // Set values for non-running lanes to zero so they don't affect the
    // result.
    return __reduce_max_uint64(__mask ? v : 0);
}

#define REDUCE_EQUAL(TYPE, FUNCTYPE, MASKTYPE)                     \
__declspec(safe)                                                   \
static inline uniform bool reduce_equal(TYPE v) {                  \
    uniform TYPE unusedValue;                                      \
    return __reduce_equal_##FUNCTYPE(v, &unusedValue, (MASKTYPE)__mask); \
}                                                                  \
__declspec(safe)                                                   \
static inline uniform bool reduce_equal(TYPE v, uniform TYPE * uniform value) { \
    return __reduce_equal_##FUNCTYPE(v, value, (MASKTYPE)__mask);       \
}

REDUCE_EQUAL(int32, int32, IntMaskType)
REDUCE_EQUAL(unsigned int32, int32, UIntMaskType)
REDUCE_EQUAL(float, float, IntMaskType)
REDUCE_EQUAL(int64, int64, IntMaskType)
REDUCE_EQUAL(unsigned int64, int64, UIntMaskType)
REDUCE_EQUAL(double, double, IntMaskType)

static int32 exclusive_scan_add(int32 v) {
    return __exclusive_scan_add_i32(v, (IntMaskType)__mask);
}

static unsigned int32 exclusive_scan_add(unsigned int32 v) {
    return __exclusive_scan_add_i32((int32)v, (IntMaskType)__mask);
}

static float exclusive_scan_add(float v) {
    return __exclusive_scan_add_float(v, __mask);
}

static int64 exclusive_scan_add(int64 v) {
    return __exclusive_scan_add_i64(v, (IntMaskType)__mask);
}

static unsigned int64 exclusive_scan_add(unsigned int64 v) {
    return __exclusive_scan_add_i64(v, (UIntMaskType)__mask);
}

static double exclusive_scan_add(double v) {
    return __exclusive_scan_add_double(v, __mask);
}

static int32 exclusive_scan_and(int32 v) {
    return __exclusive_scan_and_i32(v, (IntMaskType)__mask);
}

static unsigned int32 exclusive_scan_and(unsigned int32 v) {
    return __exclusive_scan_and_i32(v, (UIntMaskType)__mask);
}

static int64 exclusive_scan_and(int64 v) {
    return __exclusive_scan_and_i64(v, (IntMaskType)__mask);
}

static unsigned int64 exclusive_scan_and(unsigned int64 v) {
    return __exclusive_scan_and_i64(v, (UIntMaskType)__mask);
}

static int32 exclusive_scan_or(int32 v) {
    return __exclusive_scan_or_i32(v, (IntMaskType)__mask);
}

static unsigned int32 exclusive_scan_or(unsigned int32 v) {
    return __exclusive_scan_or_i32(v, (UIntMaskType)__mask);
}

static int64 exclusive_scan_or(int64 v) {
    return __exclusive_scan_or_i64(v, (IntMaskType)__mask);
}

static unsigned int64 exclusive_scan_or(unsigned int64 v) {
    return __exclusive_scan_or_i64(v, (UIntMaskType)__mask);
}

///////////////////////////////////////////////////////////////////////////
// packed load, store

static inline uniform int 
packed_load_active(uniform unsigned int a[],
                   varying unsigned int * uniform vals) {
    return __packed_load_active(a, vals, (UIntMaskType)__mask);
}

static inline uniform int
packed_store_active(uniform unsigned int a[],
                    unsigned int vals) {
    return __packed_store_active(a, vals, (UIntMaskType)__mask);
}

static inline uniform int 
packed_load_active(uniform int a[], varying int * uniform vals) {
    return __packed_load_active(a, vals, (IntMaskType)__mask);
}

static inline uniform int 
packed_store_active(uniform int a[], int vals) {
    return __packed_store_active(a, vals, (IntMaskType)__mask);
}

///////////////////////////////////////////////////////////////////////////
// System information

static inline uniform int num_cores() {
    return __num_cores();
}

__declspec(safe) 
static inline uniform int64 clock() {
    return __clock();
}

///////////////////////////////////////////////////////////////////////////
// Floating-Point Math

__declspec(safe,cost1) 
static inline float abs(float a) {
    // Floating-point hack: zeroing the high bit clears the sign
    unsigned int i = intbits(a);
    i &= 0x7fffffff;
    return floatbits(i);
}

__declspec(safe,cost1) 
static inline uniform float abs(uniform float a) {
    uniform unsigned int i = intbits(a);
    i &= 0x7fffffff;
    return floatbits(i);
}

__declspec(safe,cost1) 
static inline double abs(double a) {
    // zeroing the high bit clears the sign
    unsigned int64 i = intbits(a);
    i &= 0x7fffffffffffffff;
    return doublebits(i);
}

__declspec(safe,cost1) 
static inline uniform double abs(uniform double a) {
    uniform unsigned int64 i = intbits(a);
    i &= 0x7fffffffffffffff;
    return doublebits(i);
}

__declspec(safe,cost1) 
static inline unsigned int signbits(float x) {
    unsigned int i = intbits(x);
    return (i & 0x80000000);
}

__declspec(safe,cost1) 
static inline uniform unsigned int signbits(uniform float x) {
    uniform unsigned int i = intbits(x);
    return (i & 0x80000000);
}

__declspec(safe,cost1) 
static inline unsigned int64 signbits(double x) {
    unsigned int64 i = intbits(x);
    return (i & 0x8000000000000000);
}

__declspec(safe,cost1) 
static inline uniform unsigned int64 signbits(uniform double x) {
    uniform unsigned int64 i = intbits(x);
    return (i & 0x8000000000000000);
}

__declspec(safe,cost2) 
static inline float round(float x) {
    return __round_varying_float(x);
}

__declspec(safe,cost2) 
static inline uniform float round(uniform float x) {
    return __round_uniform_float(x);
}

__declspec(safe,cost2) 
static inline double round(double x) {
    return __round_varying_double(x);
}

__declspec(safe,cost2) 
static inline uniform double round(uniform double x) {
    return __round_uniform_double(x);
}

__declspec(safe,cost2) 
static inline float floor(float x) {
    return __floor_varying_float(x);
}

__declspec(safe,cost2) 
static inline uniform float floor(uniform float x) {
    return __floor_uniform_float(x);
}

__declspec(safe,cost2) 
static inline double floor(double x) {
    return __floor_varying_double(x);
}

__declspec(safe,cost2) 
static inline uniform double floor(uniform double x) {
    return __floor_uniform_double(x);
}

__declspec(safe,cost2) 
static inline float ceil(float x) {
    return __ceil_varying_float(x);
}

__declspec(safe,cost2) 
static inline uniform float ceil(uniform float x) {
    return __ceil_uniform_float(x);
}

__declspec(safe,cost2) 
static inline double ceil(double x) {
    return __ceil_varying_double(x);
}

__declspec(safe,cost2) 
static inline uniform double ceil(uniform double x) {
    return __ceil_uniform_double(x);
}

__declspec(safe) 
static inline float rcp(float v) {
    return __rcp_varying_float(v);
}

__declspec(safe) 
static inline uniform float rcp(uniform float v) {
    return __rcp_uniform_float(v);
}

///////////////////////////////////////////////////////////////////////////
// min/max

// float

__declspec(safe,cost1) 
static inline float min(float a, float b) {
    return __min_varying_float(a, b);
}

__declspec(safe,cost1) 
static inline uniform float min(uniform float a, uniform float b) {
    return __min_uniform_float(a, b);
}

__declspec(safe,cost1) 
static inline float max(float a, float b) {
    return __max_varying_float(a, b);
}

__declspec(safe,cost1) 
static inline uniform float max(uniform float a, uniform float b) {
    return __max_uniform_float(a, b);
}


// double

__declspec(safe) 
static inline double min(double a, double b) {
    return __min_varying_double(a, b);
}

__declspec(safe) 
static inline uniform double min(uniform double a, uniform double b) {
    return __min_uniform_double(a, b);
}

__declspec(safe) 
static inline double max(double a, double b) {
    return __max_varying_double(a, b);
}

__declspec(safe) 
static inline uniform double max(uniform double a, uniform double b) {
    return __max_uniform_double(a, b);
}

// int8

__declspec(safe,cost2) 
static inline uniform unsigned int8 min(uniform unsigned int8 a,
                                        uniform unsigned int8 b) {
    return (a < b) ? a : b;
}

__declspec(safe,cost2) 
static inline uniform unsigned int8 max(uniform unsigned int8 a, 
                                        uniform unsigned int8 b) {
    return (a > b) ? a : b;
}

__declspec(safe,cost2)
static inline uniform int8 min(uniform int8 a, uniform int8 b) {
    return (a < b) ? a : b;
}

__declspec(safe,cost2)
static inline uniform int8 max(uniform int8 a, uniform int8 b) {
    return (a > b) ? a : b;
}

__declspec(safe,cost2)
static inline unsigned int8 min(unsigned int8 a, unsigned int8 b) {
    return (a < b) ? a : b;
}

__declspec(safe,cost2)
static inline unsigned int8 max(unsigned int8 a, unsigned int8 b) {
    return (a > b) ? a : b;
}

__declspec(safe,cost2)
static inline int8 min(int8 a, int8 b) {
    return (a < b) ? a : b;
}

__declspec(safe,cost2)
static inline int8 max(int8 a, int8 b) {
    return (a > b) ? a : b;
}

// int16

__declspec(safe,cost2)
static inline uniform unsigned int16 min(uniform unsigned int16 a, 
                                         uniform unsigned int16 b) {
    return (a < b) ? a : b;
}

__declspec(safe,cost2)
static inline uniform unsigned int16 max(uniform unsigned int16 a, 
                                         uniform unsigned int16 b) {
    return (a > b) ? a : b;
}

__declspec(safe,cost2)
static inline uniform int16 min(uniform int16 a, uniform int16 b) {
    return (a < b) ? a : b;
}

__declspec(safe,cost2)
static inline uniform int16 max(uniform int16 a, uniform int16 b) {
    return (a > b) ? a : b;
}

__declspec(safe,cost2)
static inline unsigned int16 min(unsigned int16 a, unsigned int16 b) {
    return (a < b) ? a : b;
}

__declspec(safe,cost2)
static inline unsigned int16 max(unsigned int16 a, unsigned int16 b) {
    return (a > b) ? a : b;
}

__declspec(safe,cost2)
static inline int16 min(int16 a, int16 b) {
    return (a < b) ? a : b;
}

__declspec(safe,cost2)
static inline int16 max(int16 a, int16 b) {
    return (a > b) ? a : b;
}

// int32

__declspec(safe,cost1)
static inline unsigned int min(unsigned int a, unsigned int b) {
    return __min_varying_uint32(a, b);
}

__declspec(safe,cost1)
static inline uniform unsigned int min(uniform unsigned int a, uniform unsigned int b) {
    return __min_uniform_uint32(a, b);
}

__declspec(safe,cost1)
static inline unsigned int max(unsigned int a, unsigned int b) {
    return __max_varying_uint32(a, b);
}

__declspec(safe,cost1)
static inline uniform unsigned int max(uniform unsigned int a, uniform unsigned int b) {
    return __max_uniform_uint32(a, b);
}

__declspec(safe,cost1)
static inline int min(int a, int b) {
    return __min_varying_int32(a, b);
}

__declspec(safe,cost1)
static inline uniform int min(uniform int a, uniform int b) {
    return __min_uniform_int32(a, b);
}

__declspec(safe,cost1)
static inline int max(int a, int b) {
    return __max_varying_int32(a, b);
}

__declspec(safe,cost1)
static inline uniform int max(uniform int a, uniform int b) {
    return __max_uniform_int32(a, b);
}

// int64

__declspec(safe,cost1)
static inline unsigned int64 min(unsigned int64 a, unsigned int64 b) {
    return __min_varying_uint64(a, b);
}

__declspec(safe,cost1)
static inline uniform unsigned int64 min(uniform unsigned int64 a, uniform unsigned int64 b) {
    return __min_uniform_uint64(a, b);
}

__declspec(safe,cost1)
static inline unsigned int64 max(unsigned int64 a, unsigned int64 b) {
    return __max_varying_uint64(a, b);
}

__declspec(safe,cost1)
static inline uniform unsigned int64 max(uniform unsigned int64 a, uniform unsigned int64 b) {
    return __max_uniform_uint64(a, b);
}

__declspec(safe,cost1)
static inline int64 min(int64 a, int64 b) {
    return __min_varying_int64(a, b);
}

__declspec(safe,cost1)
static inline uniform int64 min(uniform int64 a, uniform int64 b) {
    return __min_uniform_int64(a, b);
}

__declspec(safe,cost1)
static inline int64 max(int64 a, int64 b) {
    return __max_varying_int64(a, b);
}

__declspec(safe,cost1)
static inline uniform int64 max(uniform int64 a, uniform int64 b) {
    return __max_uniform_int64(a, b);
}

///////////////////////////////////////////////////////////////////////////
// clamps

// float

__declspec(safe,cost2)
static inline float clamp(float v, float low, float high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform float clamp(uniform float v, uniform float low, uniform float high) {
    return min(max(v, low), high);
}

// int8

__declspec(safe,cost2)
static inline unsigned int8 clamp(unsigned int8 v, unsigned int8 low, 
                                   unsigned int8 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform unsigned int8 clamp(uniform unsigned int8 v, 
                                           uniform unsigned int8 low, 
                                           uniform unsigned int8 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline int8 clamp(int8 v, int8 low, int8 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform int8 clamp(uniform int8 v, uniform int8 low, 
                                  uniform int8 high) {
    return min(max(v, low), high);
}

// int16

__declspec(safe,cost2)
static inline unsigned int16 clamp(unsigned int16 v, unsigned int16 low, 
                                   unsigned int16 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform unsigned int16 clamp(uniform unsigned int16 v, 
                                           uniform unsigned int16 low, 
                                           uniform unsigned int16 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline int16 clamp(int16 v, int16 low, int16 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform int16 clamp(uniform int16 v, uniform int16 low, 
                                  uniform int16 high) {
    return min(max(v, low), high);
}

// int32

__declspec(safe,cost2)
static inline unsigned int clamp(unsigned int v, unsigned int low, unsigned int high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform unsigned int clamp(uniform unsigned int v, uniform unsigned int low, 
                                         uniform unsigned int high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline int clamp(int v, int low, int high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform int clamp(uniform int v, uniform int low, uniform int high) {
    return min(max(v, low), high);
}

// int64

__declspec(safe,cost2)
static inline unsigned int64 clamp(unsigned int64 v, unsigned int64 low, 
                                   unsigned int64 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform unsigned int64 clamp(uniform unsigned int64 v, 
                                           uniform unsigned int64 low, 
                                           uniform unsigned int64 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline int64 clamp(int64 v, int64 low, int64 high) {
    return min(max(v, low), high);
}

__declspec(safe,cost2)
static inline uniform int64 clamp(uniform int64 v, uniform int64 low, 
                                  uniform int64 high) {
    return min(max(v, low), high);
}

///////////////////////////////////////////////////////////////////////////
// Global atomics and memory barriers

static inline void memory_barrier() {
    __memory_barrier();
}

#define DEFINE_ATOMIC_OP(TA,TB,OPA,OPB,MASKTYPE)                        \
static inline TA atomic_##OPA##_global(uniform TA * uniform ptr, TA value) { \
    TA ret = __atomic_##OPB##_##TB##_global(ptr, value, (MASKTYPE)__mask); \
    return ret;                                                         \
}                                                                       \
static inline uniform TA atomic_##OPA##_global(uniform TA * uniform ptr, \
                                               uniform TA value) {      \
    uniform TA ret = __atomic_##OPB##_uniform_##TB##_global(ptr, value); \
    return ret;                                                         \
}                                                                       \
static inline TA atomic_##OPA##_global(uniform TA * varying ptr, TA value) { \
    uniform TA * uniform ptrArray[programCount];                        \
    ptrArray[programIndex] = ptr;                                       \
    TA ret;                                                             \
    __foreach_active (i) {                                              \
        uniform TA * uniform p = ptrArray[i];                           \
        uniform TA v = extract(value, i);                               \
        uniform TA r = __atomic_##OPB##_uniform_##TB##_global(p, v);    \
        ret = insert(ret, i, r);                                        \
    }                                                                   \
    return ret;                                                         \
}                                                                       \

#define DEFINE_ATOMIC_SWAP(TA,TB)                                       \
static inline TA atomic_swap_global(uniform TA * uniform ptr, TA value) { \
    uniform int i = 0;                                                  \
    TA ret[programCount];                                               \
    TA memVal;                                                          \
    uniform int lastSwap;                                               \
    uniform int mask = lanemask();                                      \
    /* First, have the first running program instance (if any) perform  \
       the swap with memory with its value of "value"; record the       \
       value returned. */                                               \
    for (; i < programCount; ++i) {                                     \
        if ((mask & (1 << i)) == 0)                                     \
            continue;                                                   \
        memVal = __atomic_swap_uniform_##TB##_global(ptr, extract(value, i)); \
        lastSwap = i;                                                   \
        break;                                                          \
    }                                                                   \
    /* Now, for all of the remaining running program instances, set the \
       return value of the last instance that did a swap with this      \
       instance's value of "value"; this gives the same effect as if the \
       current instance had executed a hardware atomic swap right before \
       the last one that did a swap. */                                 \
    for (; i < programCount; ++i) {                                     \
        if ((mask & (1 << i)) == 0)                                     \
            continue;                                                   \
        ret[lastSwap] = extract(value, i);                              \
        lastSwap = i;                                                   \
    }                                                                   \
    /* And the last instance that wanted to swap gets the value we      \
       originally got back from memory... */                            \
    ret[lastSwap] = memVal;                                             \
    return ret[programIndex];                                           \
}                                                                       \
static inline uniform TA atomic_swap_global(uniform TA * uniform ptr,   \
                                            uniform TA value) {         \
    uniform TA ret = __atomic_swap_uniform_##TB##_global(ptr, value);   \
    return ret;                                                         \
}                                                                       \
static inline TA atomic_swap_global(uniform TA * varying ptr, TA value) { \
    uniform TA * uniform ptrArray[programCount];                        \
    ptrArray[programIndex] = ptr;                                       \
    TA ret;                                                             \
    __foreach_active (i) {                                              \
        uniform TA * uniform p = ptrArray[i];                           \
        uniform TA v = extract(value, i);                               \
        uniform TA r = __atomic_swap_uniform_##TB##_global(p, v);       \
        ret = insert(ret, i, r);                                        \
    }                                                                   \
    return ret;                                                         \
}                                                                       \

#define DEFINE_ATOMIC_MINMAX_OP(TA,TB,OPA,OPB)                          \
static inline TA atomic_##OPA##_global(uniform TA * uniform ptr, TA value) { \
    uniform TA oneval = reduce_##OPA(value);                            \
    TA ret;                                                             \
    if (lanemask() != 0)                                                \
        ret = __atomic_##OPB##_uniform_##TB##_global(ptr, oneval);      \
    return ret;                                                         \
}                                                                       \
static inline uniform TA atomic_##OPA##_global(uniform TA * uniform ptr, \
                                               uniform TA value) {      \
    uniform TA ret = __atomic_##OPB##_uniform_##TB##_global(ptr, value); \
    return ret;                                                         \
}                                                                       \
static inline TA atomic_##OPA##_global(uniform TA * varying ptr,        \
                                       TA value) {                      \
    uniform TA * uniform ptrArray[programCount];                        \
    ptrArray[programIndex] = ptr;                                       \
    TA ret;                                                             \
    __foreach_active (i) {                                              \
        uniform TA * uniform p = ptrArray[i];                           \
        uniform TA v = extract(value, i);                               \
        uniform TA r = __atomic_##OPB##_uniform_##TB##_global(p, v);    \
        ret = insert(ret, i, r);                                        \
    }                                                                   \
    return ret;                                                         \
}

DEFINE_ATOMIC_OP(int32,int32,add,add,IntMaskType)
DEFINE_ATOMIC_OP(int32,int32,subtract,sub,IntMaskType)
DEFINE_ATOMIC_MINMAX_OP(int32,int32,min,min)
DEFINE_ATOMIC_MINMAX_OP(int32,int32,max,max)
DEFINE_ATOMIC_OP(int32,int32,and,and,IntMaskType)
DEFINE_ATOMIC_OP(int32,int32,or,or,IntMaskType)
DEFINE_ATOMIC_OP(int32,int32,xor,xor,IntMaskType)
DEFINE_ATOMIC_SWAP(int32,int32)

// For everything but atomic min and max, we can use the same
// implementations for unsigned as for signed.
DEFINE_ATOMIC_OP(unsigned int32,int32,add,add,UIntMaskType)
DEFINE_ATOMIC_OP(unsigned int32,int32,subtract,sub,UIntMaskType)
DEFINE_ATOMIC_MINMAX_OP(unsigned int32,uint32,min,umin)
DEFINE_ATOMIC_MINMAX_OP(unsigned int32,uint32,max,umax)
DEFINE_ATOMIC_OP(unsigned int32,int32,and,and,UIntMaskType)
DEFINE_ATOMIC_OP(unsigned int32,int32,or,or,UIntMaskType)
DEFINE_ATOMIC_OP(unsigned int32,int32,xor,xor,UIntMaskType)
DEFINE_ATOMIC_SWAP(unsigned int32,int32)

DEFINE_ATOMIC_SWAP(float,float)

DEFINE_ATOMIC_OP(int64,int64,add,add,IntMaskType)
DEFINE_ATOMIC_OP(int64,int64,subtract,sub,IntMaskType)
DEFINE_ATOMIC_MINMAX_OP(int64,int64,min,min)
DEFINE_ATOMIC_MINMAX_OP(int64,int64,max,max)
DEFINE_ATOMIC_OP(int64,int64,and,and,IntMaskType)
DEFINE_ATOMIC_OP(int64,int64,or,or,IntMaskType)
DEFINE_ATOMIC_OP(int64,int64,xor,xor,IntMaskType)
DEFINE_ATOMIC_SWAP(int64,int64)

// For everything but atomic min and max, we can use the same
// implementations for unsigned as for signed.
DEFINE_ATOMIC_OP(unsigned int64,int64,add,add,UIntMaskType)
DEFINE_ATOMIC_OP(unsigned int64,int64,subtract,sub,UIntMaskType)
DEFINE_ATOMIC_MINMAX_OP(unsigned int64,uint64,min,umin)
DEFINE_ATOMIC_MINMAX_OP(unsigned int64,uint64,max,umax)
DEFINE_ATOMIC_OP(unsigned int64,int64,and,and,UIntMaskType)
DEFINE_ATOMIC_OP(unsigned int64,int64,or,or,UIntMaskType)
DEFINE_ATOMIC_OP(unsigned int64,int64,xor,xor,UIntMaskType)
DEFINE_ATOMIC_SWAP(unsigned int64,int64)

DEFINE_ATOMIC_SWAP(double,double)

#undef DEFINE_ATOMIC_OP
#undef DEFINE_ATOMIC_MINMAX_OP
#undef DEFINE_ATOMIC_SWAP

#define ATOMIC_DECL_CMPXCHG(TA, TB, MASKTYPE)                           \
static inline uniform TA atomic_compare_exchange_global(               \
         uniform TA * uniform ptr, uniform TA oldval, uniform TA newval) { \
    uniform TA ret =                                                    \
        __atomic_compare_exchange_uniform_##TB##_global(ptr, oldval, newval); \
    return ret;                                                         \
}                                                                       \
static inline TA atomic_compare_exchange_global(                           \
         uniform TA * uniform ptr, TA oldval, TA newval) {                 \
    TA ret = __atomic_compare_exchange_##TB##_global(ptr, oldval, newval,  \
                                                     (MASKTYPE)__mask);    \
    return ret;                                                            \
} \
static inline TA atomic_compare_exchange_global(               \
         uniform TA * varying ptr, TA oldval, TA newval) { \
    uniform TA * uniform ptrArray[programCount];                        \
    ptrArray[programIndex] = ptr;                                       \
    TA ret;                                                             \
    __foreach_active (i) {                                              \
        uniform TA r =                                                  \
            __atomic_compare_exchange_uniform_##TB##_global(ptrArray[i], \
                                                            extract(oldval, i), \
                                                            extract(newval, i)); \
        ret = insert(ret, i, r);                                        \
    }                                                                   \
    return ret;                                                         \
}

ATOMIC_DECL_CMPXCHG(int32, int32, IntMaskType)
ATOMIC_DECL_CMPXCHG(unsigned int32, int32, UIntMaskType)
ATOMIC_DECL_CMPXCHG(float, float, IntMaskType)
ATOMIC_DECL_CMPXCHG(int64, int64, IntMaskType)
ATOMIC_DECL_CMPXCHG(unsigned int64, int64, UIntMaskType)
ATOMIC_DECL_CMPXCHG(double, double, IntMaskType)

#undef ATOMIC_DECL_CMPXCHG

// void * variants of swap and compare exchange

static inline void *atomic_swap_global(void ** uniform ptr,
                                       void * value) {
    return (void *)atomic_swap_global((intptr_t * uniform)ptr,
                                      (intptr_t)value);
}

static inline void * uniform atomic_swap_global(void ** uniform ptr,
                                                void * uniform value) {
    return (void * uniform)atomic_swap_global((intptr_t * uniform)ptr,
                                              (uniform intptr_t)value);
}

static inline void *atomic_swap_global(void ** ptr, void * value) {
    return (void *)atomic_swap_global((intptr_t *)ptr,
                                      (intptr_t)value);
}

static inline void * 
atomic_compare_exchange_global(void ** uniform ptr, 
                               void * oldval, void * newval) {
    return (void *)atomic_compare_exchange_global((intptr_t * uniform)ptr,
                                                  (intptr_t)oldval,
                                                  (intptr_t)newval);
}

static inline void * uniform
atomic_compare_exchange_global(void ** uniform ptr, void * uniform oldval, 
                               void * uniform newval) { 
    return (void * uniform)atomic_compare_exchange_global((intptr_t * uniform)ptr,
                                                          (uniform intptr_t)oldval,
                                                          (uniform intptr_t)newval);
}

static inline void *
atomic_compare_exchange_global(void ** ptr, void * oldval,
                               void * newval) {
    return (void *)atomic_compare_exchange_global((intptr_t *)ptr,
                                                  (intptr_t)oldval,
                                                  (intptr_t)newval);
}

///////////////////////////////////////////////////////////////////////////
// local atomics

#define LOCAL_ATOMIC(TYPE,NAME,OPFUNC)                                  \
static inline uniform TYPE atomic_##NAME##_local(uniform TYPE * uniform ptr, \
                                                 uniform TYPE value) {  \
    uniform TYPE ret = *ptr;                                           \
    *ptr = OPFUNC(*ptr, value);                                        \
     return ret;                                                       \
}                                                                      \
static inline TYPE atomic_##NAME##_local(uniform TYPE * uniform ptr, TYPE value) { \
    TYPE ret;                                                          \
    __foreach_active (i) {                                             \
        ret = insert(ret, i, *ptr);                                    \
        *ptr = OPFUNC(*ptr, extract(value, i));                        \
    }                                                                  \
    return ret;                                                        \
}                                                                      \
static inline TYPE atomic_##NAME##_local(uniform TYPE * p, TYPE value) {    \
    TYPE ret;                                                          \
    uniform TYPE * uniform ptrs[programCount];                         \
    ptrs[programIndex] = p;                                            \
    __foreach_active (i) {                                             \
        ret = insert(ret, i, *ptrs[i]);                                \
        *ptrs[i] = OPFUNC(*ptrs[i], extract(value, i));                \
    }                                                                  \
    return ret;                                                        \
}

static inline uniform int32 __add(uniform int32 a, uniform int32 b) { return a+b; }
static inline uniform int32 __sub(uniform int32 a, uniform int32 b) { return a-b; }
static inline uniform int32 __and(uniform int32 a, uniform int32 b) { return a & b; }
static inline uniform int32 __or(uniform int32 a, uniform int32 b) { return a | b; }
static inline uniform int32 __xor(uniform int32 a, uniform int32 b) { return a ^ b; }
static inline uniform int32 __swap(uniform int32 a, uniform int32 b) { return b; }

static inline uniform unsigned int32 __add(uniform unsigned int32 a, 
                                           uniform unsigned int32 b) { return a+b; }
static inline uniform unsigned int32 __sub(uniform unsigned int32 a, 
                                           uniform unsigned int32 b) { return a-b; }
static inline uniform unsigned int32 __and(uniform unsigned int32 a, 
                                           uniform unsigned int32 b) { return a & b; }
static inline uniform unsigned int32 __or(uniform unsigned int32 a, 
                                          uniform unsigned int32 b) { return a | b; }
static inline uniform unsigned int32 __xor(uniform unsigned int32 a, 
                                           uniform unsigned int32 b) { return a ^ b; }
static inline uniform unsigned int32 __swap(uniform unsigned int32 a, 
                                            uniform unsigned int32 b) { return b; }


static inline uniform float __add(uniform float a, uniform float b) { return a+b; }
static inline uniform float __sub(uniform float a, uniform float b) { return a-b; }
static inline uniform float __swap(uniform float a, uniform float b) { return b; }

static inline uniform int64 __add(uniform int64 a, uniform int64 b) { return a+b; }
static inline uniform int64 __sub(uniform int64 a, uniform int64 b) { return a-b; }
static inline uniform int64 __and(uniform int64 a, uniform int64 b) { return a & b; }
static inline uniform int64 __or(uniform int64 a, uniform int64 b) { return a | b; }
static inline uniform int64 __xor(uniform int64 a, uniform int64 b) { return a ^ b; }
static inline uniform int64 __swap(uniform int64 a, uniform int64 b) { return b; }

static inline uniform unsigned int64 __add(uniform unsigned int64 a, 
                                           uniform unsigned int64 b) { return a+b; }
static inline uniform unsigned int64 __sub(uniform unsigned int64 a, 
                                           uniform unsigned int64 b) { return a-b; }
static inline uniform unsigned int64 __and(uniform unsigned int64 a, 
                                           uniform unsigned int64 b) { return a & b; }
static inline uniform unsigned int64 __or(uniform unsigned int64 a, 
                                          uniform unsigned int64 b) { return a | b; }
static inline uniform unsigned int64 __xor(uniform unsigned int64 a, 
                                           uniform unsigned int64 b) { return a ^ b; }
static inline uniform unsigned int64 __swap(uniform unsigned int64 a, 
                                            uniform unsigned int64 b) { return b; }

static inline uniform double __add(uniform double a, uniform double b) { return a+b; }
static inline uniform double __sub(uniform double a, uniform double b) { return a-b; }
static inline uniform double __swap(uniform double a, uniform double b) { return a-b; }

LOCAL_ATOMIC(int32, add, __add)
LOCAL_ATOMIC(int32, subtract, __sub)
LOCAL_ATOMIC(int32, and, __and)
LOCAL_ATOMIC(int32, or, __or)
LOCAL_ATOMIC(int32, xor, __xor)
LOCAL_ATOMIC(int32, min, min)
LOCAL_ATOMIC(int32, max, max)
LOCAL_ATOMIC(int32, swap, __swap)

LOCAL_ATOMIC(unsigned int32, add, __add)
LOCAL_ATOMIC(unsigned int32, subtract, __sub)
LOCAL_ATOMIC(unsigned int32, and, __and)
LOCAL_ATOMIC(unsigned int32, or, __or)
LOCAL_ATOMIC(unsigned int32, xor, __xor)
LOCAL_ATOMIC(unsigned int32, min, min)
LOCAL_ATOMIC(unsigned int32, max, max)
LOCAL_ATOMIC(unsigned int32, swap, __swap)

LOCAL_ATOMIC(float, add, __add)
LOCAL_ATOMIC(float, subtract, __sub)
LOCAL_ATOMIC(float, min, min)
LOCAL_ATOMIC(float, max, max)
LOCAL_ATOMIC(float, swap, __swap)

LOCAL_ATOMIC(int64, add, __add)
LOCAL_ATOMIC(int64, subtract, __sub)
LOCAL_ATOMIC(int64, and, __and)
LOCAL_ATOMIC(int64, or, __or)
LOCAL_ATOMIC(int64, xor, __xor)
LOCAL_ATOMIC(int64, min, min)
LOCAL_ATOMIC(int64, max, max)
LOCAL_ATOMIC(int64, swap, __swap)

LOCAL_ATOMIC(unsigned int64, add, __add)
LOCAL_ATOMIC(unsigned int64, subtract, __sub)
LOCAL_ATOMIC(unsigned int64, and, __and)
LOCAL_ATOMIC(unsigned int64, or, __or)
LOCAL_ATOMIC(unsigned int64, xor, __xor)
LOCAL_ATOMIC(unsigned int64, min, min)
LOCAL_ATOMIC(unsigned int64, max, max)
LOCAL_ATOMIC(unsigned int64, swap, __swap)

LOCAL_ATOMIC(double, add, __add)
LOCAL_ATOMIC(double, subtract, __sub)
LOCAL_ATOMIC(double, min, min)
LOCAL_ATOMIC(double, max, max)
LOCAL_ATOMIC(double, swap, __swap)

// compare exchange
#define LOCAL_CMPXCHG(TYPE)                                             \
static inline uniform TYPE atomic_compare_exchange_local(uniform TYPE * uniform ptr, \
                                                         uniform TYPE cmp, \
                                                         uniform TYPE update) { \
    uniform TYPE old = *ptr;                                               \
    if (old == cmp)                                                     \
        *ptr = update;                                                  \
    return old;                                                         \
}                                                                       \
static inline TYPE atomic_compare_exchange_local(uniform TYPE * uniform ptr, \
                                                 TYPE cmp, TYPE update) { \
    TYPE ret;                                                          \
    __foreach_active (i) {                                             \
        uniform TYPE old = *ptr;                                       \
        if (old == extract(cmp, i))                                    \
            *ptr = extract(update, i);                                 \
        ret = insert(ret, i, old);                                     \
    }                                                                  \
    return ret;                                                        \
}                                                                       \
static inline TYPE atomic_compare_exchange_local(uniform TYPE * varying p, \
                                                 TYPE cmp, TYPE update) { \
    uniform TYPE * uniform ptrs[programCount];                          \
    ptrs[programIndex] = p;                                            \
    TYPE ret;                                                          \
    __foreach_active (i) {                                             \
        uniform TYPE old = *ptrs[i];                                   \
        if (old == extract(cmp, i))                                    \
            *ptrs[i] = extract(update, i);                             \
        ret = insert(ret, i, old);                                     \
    }                                                                  \
    return ret;                                                        \
}

LOCAL_CMPXCHG(int32)
LOCAL_CMPXCHG(unsigned int32)
LOCAL_CMPXCHG(float)
LOCAL_CMPXCHG(int64)
LOCAL_CMPXCHG(unsigned int64)
LOCAL_CMPXCHG(double)

#undef LOCAL_ATOMIC
#undef LOCAL_CMPXCHG

// void * variants of swap and compare exchange

static inline void *atomic_swap_local(void ** uniform ptr,
                                      void * value) {
    return (void *)atomic_swap_local((intptr_t * uniform)ptr,
                                      (intptr_t)value);
}

static inline void * uniform atomic_swap_local(void ** uniform ptr,
                                               void * uniform value) {
    return (void * uniform)atomic_swap_local((intptr_t * uniform)ptr,
                                              (uniform intptr_t)value);
}

static inline void *atomic_swap_local(void ** ptr, void * value) {
    return (void *)atomic_swap_local((intptr_t *)ptr,
                                      (intptr_t)value);
}

static inline void * 
atomic_compare_exchange_local(void ** uniform ptr, 
                              void * oldval, void * newval) {
    return (void *)atomic_compare_exchange_local((intptr_t * uniform)ptr,
                                                  (intptr_t)oldval,
                                                  (intptr_t)newval);
}

static inline void * uniform
atomic_compare_exchange_local(void ** uniform ptr, void * uniform oldval, 
                              void * uniform newval) { 
    return (void * uniform)atomic_compare_exchange_local((intptr_t * uniform)ptr,
                                                          (uniform intptr_t)oldval,
                                                          (uniform intptr_t)newval);
}

static inline void *
atomic_compare_exchange_local(void ** ptr, void * oldval,
                              void * newval) {
    return (void *)atomic_compare_exchange_local((intptr_t *)ptr,
                                                  (intptr_t)oldval,
                                                  (intptr_t)newval);
}

///////////////////////////////////////////////////////////////////////////
// Transcendentals (float precision)

__declspec(safe)
static inline float sqrt(float v) {
    return __sqrt_varying_float(v);
}

__declspec(safe)
static inline uniform float sqrt(uniform float v) {
    return __sqrt_uniform_float(v);
}

__declspec(safe)
static inline float rsqrt(float v) {
    return __rsqrt_varying_float(v);
}

__declspec(safe)
static inline uniform float rsqrt(uniform float v) {
    return __rsqrt_uniform_float(v);
}

__declspec(safe)
static inline float ldexp(float x, int n) {
    unsigned int ex = 0x7F800000u;
    unsigned int ix = intbits(x);
    ex &= ix;              // extract old exponent;
    ix = ix & ~0x7F800000u;  // clear exponent
    n = (n << 23) + ex;
    ix |= n; // insert new exponent
    return floatbits(ix);
}

__declspec(safe)
static inline uniform float ldexp(uniform float x, uniform int n) {
    uniform unsigned int ex = 0x7F800000u;
    uniform unsigned int ix = intbits(x);
    ex &= ix;              // extract old exponent;
    ix = ix & ~0x7F800000u;  // clear exponent
    n = (n << 23) + ex;
    ix |= n; // insert new exponent
    return floatbits(ix);
}

__declspec(safe)
static inline float frexp(float x, varying int * uniform pw2) {
    unsigned int ex = 0x7F800000u;              // exponent mask
    unsigned int ix = intbits(x);
    ex &= ix;
    ix &= ~0x7F800000u;  // clear exponent
    *pw2 = (int)(ex >> 23) - 126; // compute exponent
    ix |= 0x3F000000u;         // insert exponent +1 in x
    return floatbits(ix);
}

__declspec(safe)
static inline uniform float frexp(uniform float x, uniform int * uniform pw2) {
    uniform unsigned int ex = 0x7F800000u;              // exponent mask
    uniform unsigned int ix = intbits(x);
    ex &= ix;
    ix &= ~0x7F800000u;  // clear exponent
    *pw2 = (uniform int)(ex >> 23) - 126; // compute exponent
    ix |= 0x3F000000u;         // insert exponent +1 in x
    return floatbits(ix);
}

// Most of the transcendental implementations in ispc code here come from
// Solomon Boulos's "syrah": https://github.com/boulos/syrah/

__declspec(safe)
static inline float sin(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_sin(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_sinf(extract(x_full, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        static const float pi_over_two_vec = 1.57079637050628662109375;
        static const float two_over_pi_vec = 0.636619746685028076171875;
        float scaled = x_full * two_over_pi_vec;
        float k_real = floor(scaled);
        int k = (int)k_real;

        // Reduced range version of x
        float x = x_full - k_real * pi_over_two_vec;
        int k_mod4 = k & 3;
        bool sin_usecos = (k_mod4 == 1 || k_mod4 == 3);
        bool flip_sign = (k_mod4 > 1);

        // These coefficients are from sollya with fpminimax(sin(x)/x, [|0, 2,
        // 4, 6, 8, 10|], [|single...|], [0;Pi/2]);
        static const float sin_c2 = -0.16666667163372039794921875;
        static const float sin_c4 = 8.333347737789154052734375e-3;
        static const float sin_c6 = -1.9842604524455964565277099609375e-4;
        static const float sin_c8 = 2.760012648650445044040679931640625e-6;
        static const float sin_c10 = -2.50293279435709337121807038784027099609375e-8;

        static const float cos_c2 = -0.5;
        static const float cos_c4 = 4.166664183139801025390625e-2;
        static const float cos_c6 = -1.388833043165504932403564453125e-3;
        static const float cos_c8 = 2.47562347794882953166961669921875e-5;
        static const float cos_c10 = -2.59630184018533327616751194000244140625e-7;

        float outside = sin_usecos ? 1 : x;
        float c2 = sin_usecos ? cos_c2 : sin_c2;
        float c4 = sin_usecos ? cos_c4 : sin_c4;
        float c6 = sin_usecos ? cos_c6 : sin_c6;
        float c8 = sin_usecos ? cos_c8 : sin_c8;
        float c10 = sin_usecos ? cos_c10 : sin_c10;

        float x2 = x * x;
        float formula = x2 * c10 + c8;
        formula = x2 * formula + c6;
        formula = x2 * formula + c4;
        formula = x2 * formula + c2;
        formula = x2 * formula + 1;
        formula *= outside;

        formula = flip_sign ? -formula : formula;
        return formula;
    }
}


__declspec(safe)
static inline uniform float sin(uniform float x_full) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_sinf(x_full);
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        static const uniform float pi_over_two_vec = 1.57079637050628662109375;
        static const uniform float two_over_pi_vec = 0.636619746685028076171875;
        uniform float scaled = x_full * two_over_pi_vec;
        uniform float k_real = floor(scaled);
        uniform int k = (int)k_real;

        // Reduced range version of x
        uniform float x = x_full - k_real * pi_over_two_vec;
        uniform int k_mod4 = k & 3;
        uniform bool sin_usecos = (k_mod4 == 1 || k_mod4 == 3);
        uniform bool flip_sign = (k_mod4 > 1);

        // These coefficients are from sollya with fpminimax(sin(x)/x, [|0, 2,
        // 4, 6, 8, 10|], [|single...|], [0;Pi/2]);
        static const uniform float sin_c2 = -0.16666667163372039794921875;
        static const uniform float sin_c4 = 8.333347737789154052734375e-3;
        static const uniform float sin_c6 = -1.9842604524455964565277099609375e-4;
        static const uniform float sin_c8 = 2.760012648650445044040679931640625e-6;
        static const uniform float sin_c10 = -2.50293279435709337121807038784027099609375e-8;

        static const uniform float cos_c2 = -0.5;
        static const uniform float cos_c4 = 4.166664183139801025390625e-2;
        static const uniform float cos_c6 = -1.388833043165504932403564453125e-3;
        static const uniform float cos_c8 = 2.47562347794882953166961669921875e-5;
        static const uniform float cos_c10 = -2.59630184018533327616751194000244140625e-7;

        uniform float outside, c2, c4, c6, c8, c10;
        if (sin_usecos) {
            outside = 1.;
            c2 =  cos_c2;
            c4 =  cos_c4;
            c6 =  cos_c6;
            c8 =  cos_c8;
            c10 = cos_c10;
        }
        else {
            outside = x;
            c2 = sin_c2;
            c4 = sin_c4;
            c6 = sin_c6;
            c8 = sin_c8;
            c10 = sin_c10;
        }

        uniform float x2 = x * x;
        uniform float formula = x2 * c10 + c8;
        formula = x2 * formula + c6;
        formula = x2 * formula + c4;
        formula = x2 * formula + c2;
        formula = x2 * formula + 1.;
        formula *= outside;

        formula = flip_sign ? -formula : formula;
        return formula;
    }
}


__declspec(safe)
static inline float asin(float x) {
    bool isneg = x < 0;
    x = abs(x);

    bool isnan = (x > 1);

    float v;
    if (__math_lib == __math_lib_svml ||
        __math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_asinf(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc)
        // sollya
        // fpminimax(((asin(x)-pi/2)/-sqrt(1-x)), [|0,1,2,3,4,5,6,7,8,9,10|],
        //           [|single...|], [1e-20;.9999999999999999]);
        // avg error: 8.5716801e-09, max error: 2.1373853e-07
        v = 1.57079637050628662109375f + 
            x * (-0.21460501849651336669921875f + 
            x * (8.9116774499416351318359375e-2f + 
            x * (-5.146093666553497314453125e-2f + 
            x * (3.7269376218318939208984375e-2f + 
            x * (-3.5882405936717987060546875e-2f + 
            x * (4.14929799735546112060546875e-2f +
            x * (-4.25077490508556365966796875e-2f +
            x * (3.05023305118083953857421875e-2f + 
            x * (-1.2897425331175327301025390625e-2f +
            x * 2.38926825113594532012939453125e-3f)))))))));
    else if (__math_lib == __math_lib_ispc_fast)
        // sollya
        // fpminimax(((asin(x)-pi/2)/-sqrt(1-x)), [|0,1,2,3,4,5|],[|single...|],
        //           [1e-20;.9999999999999999]);
        // avg error: 1.1105439e-06, max error 1.3187528e-06
        v = 1.57079517841339111328125f + 
             x * (-0.21450997889041900634765625f + 
             x * (8.78556668758392333984375e-2f + 
             x * (-4.489909112453460693359375e-2f + 
             x * (1.928029954433441162109375e-2f + 
             x * (-4.3095736764371395111083984375e-3f)))));

    v *= -sqrt(1.f - x);
    v = v + 1.57079637050628662109375;
    if (v < 0) v = 0;
    // v = max(0, v);

    if (isneg) v = -v;
    if (isnan) v = floatbits(0x7fc00000);

    return v;
}


__declspec(safe)
static inline uniform float asin(uniform float x) {
    uniform bool isneg = x < 0;
    x = abs(x);

    uniform bool isnan = (x > 1);

    uniform float v;
    if (__math_lib == __math_lib_svml ||
        __math_lib == __math_lib_system) {
        return __stdlib_asinf(x);
    }
    else if (__math_lib == __math_lib_ispc)
        // sollya
        // fpminimax(((asin(x)-pi/2)/-sqrt(1-x)), [|0,1,2,3,4,5,6,7,8,9,10|],
        //           [|single...|], [1e-20;.9999999999999999]);
        // avg error: 8.5716801e-09, max error: 2.1373853e-07
        v = 1.57079637050628662109375f + 
            x * (-0.21460501849651336669921875f + 
            x * (8.9116774499416351318359375e-2f + 
            x * (-5.146093666553497314453125e-2f + 
            x * (3.7269376218318939208984375e-2f + 
            x * (-3.5882405936717987060546875e-2f + 
            x * (4.14929799735546112060546875e-2f +
            x * (-4.25077490508556365966796875e-2f +
            x * (3.05023305118083953857421875e-2f + 
            x * (-1.2897425331175327301025390625e-2f +
            x * 2.38926825113594532012939453125e-3f)))))))));
    else if (__math_lib == __math_lib_ispc_fast)
        // sollya
        // fpminimax(((asin(x)-pi/2)/-sqrt(1-x)), [|0,1,2,3,4,5|],[|single...|],
        //           [1e-20;.9999999999999999]);
        // avg error: 1.1105439e-06, max error 1.3187528e-06
        v = 1.57079517841339111328125f + 
             x * (-0.21450997889041900634765625f + 
             x * (8.78556668758392333984375e-2f + 
             x * (-4.489909112453460693359375e-2f + 
             x * (1.928029954433441162109375e-2f + 
             x * (-4.3095736764371395111083984375e-3f)))));

    v *= -sqrt(1.f - x);
    v = v + 1.57079637050628662109375;
    if (v < 0) v = 0;
    // v = max(0, v);

    if (isneg) v = -v;
    if (isnan) v = floatbits(0x7fc00000);

    return v;
}


__declspec(safe)
static inline float cos(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_cos(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_cosf(extract(x_full, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        static const float pi_over_two_vec = 1.57079637050628662109375;
        static const float two_over_pi_vec = 0.636619746685028076171875;
        float scaled = x_full * two_over_pi_vec;
        float k_real = floor(scaled);
        int k = (int)k_real;

        // Reduced range version of x
        float x = x_full - k_real * pi_over_two_vec;

        int k_mod4 = k & 3;
        bool cos_usecos = (k_mod4 == 0 || k_mod4 == 2);
        bool flip_sign = (k_mod4 == 1 || k_mod4 == 2);

        const float sin_c2 = -0.16666667163372039794921875;
        const float sin_c4 = 8.333347737789154052734375e-3;
        const float sin_c6 = -1.9842604524455964565277099609375e-4;
        const float sin_c8 = 2.760012648650445044040679931640625e-6;
        const float sin_c10 = -2.50293279435709337121807038784027099609375e-8;

        const float cos_c2 = -0.5;
        const float cos_c4 = 4.166664183139801025390625e-2;
        const float cos_c6 = -1.388833043165504932403564453125e-3;
        const float cos_c8 = 2.47562347794882953166961669921875e-5;
        const float cos_c10 = -2.59630184018533327616751194000244140625e-7;

        float outside = cos_usecos ? 1. : x;
        float c2 = cos_usecos ? cos_c2 : sin_c2;
        float c4 = cos_usecos ? cos_c4 : sin_c4;
        float c6 = cos_usecos ? cos_c6 : sin_c6;
        float c8 = cos_usecos ? cos_c8 : sin_c8;
        float c10 = cos_usecos ? cos_c10 : sin_c10;

        float x2 = x * x;
        float formula = x2 * c10 + c8;
        formula = x2 * formula + c6;
        formula = x2 * formula + c4;
        formula = x2 * formula + c2;
        formula = x2 * formula + 1.;
        formula *= outside;

        formula = flip_sign ? -formula : formula;
        return formula;
    }
}


__declspec(safe)
static inline uniform float cos(uniform float x_full) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_cosf(x_full);
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        static const uniform float pi_over_two_vec = 1.57079637050628662109375;
        static const uniform float two_over_pi_vec = 0.636619746685028076171875;
        uniform float scaled = x_full * two_over_pi_vec;
        uniform float k_real = floor(scaled);
        uniform int k = (int)k_real;

        // Reduced range version of x
        uniform float x = x_full - k_real * pi_over_two_vec;

        uniform int k_mod4 = k & 3;
        uniform bool cos_usecos = (k_mod4 == 0 || k_mod4 == 2);
        uniform bool flip_sign = (k_mod4 == 1 || k_mod4 == 2);

        const uniform float sin_c2 = -0.16666667163372039794921875;
        const uniform float sin_c4 = 8.333347737789154052734375e-3;
        const uniform float sin_c6 = -1.9842604524455964565277099609375e-4;
        const uniform float sin_c8 = 2.760012648650445044040679931640625e-6;
        const uniform float sin_c10 = -2.50293279435709337121807038784027099609375e-8;

        const uniform float cos_c2 = -0.5;
        const uniform float cos_c4 = 4.166664183139801025390625e-2;
        const uniform float cos_c6 = -1.388833043165504932403564453125e-3;
        const uniform float cos_c8 = 2.47562347794882953166961669921875e-5;
        const uniform float cos_c10 = -2.59630184018533327616751194000244140625e-7;

        uniform float outside, c2, c4, c6, c8, c10;
        if (cos_usecos) {
            outside = 1.;
            c2 = cos_c2;
            c4 = cos_c4;
            c6 = cos_c6;
            c8 = cos_c8;
            c10 = cos_c10;
        }
        else {
            outside = x;
            c2 = sin_c2;
            c4 = sin_c4;
            c6 = sin_c6;
            c8 = sin_c8;
            c10 = sin_c10;
        }

        uniform float x2 = x * x;
        uniform float formula = x2 * c10 + c8;
        formula = x2 * formula + c6;
        formula = x2 * formula + c4;
        formula = x2 * formula + c2;
        formula = x2 * formula + 1.;
        formula *= outside;

        formula = flip_sign ? -formula : formula;
        return formula;
    }
}


__declspec(safe)
static inline float acos(float v) {
    return 1.57079637050628662109375 - asin(v);
}


__declspec(safe)
static inline uniform float acos(uniform float v) {
    return 1.57079637050628662109375 - asin(v);
}


__declspec(safe)
static inline void sincos(float x_full, varying float * uniform sin_result, 
                          varying float * uniform cos_result) {
    if (__math_lib == __math_lib_svml) {
        __svml_sincos(x_full, sin_result, cos_result);
    }
    else if (__math_lib == __math_lib_system) {
        __foreach_active (i) {
            uniform float s, c;
            __stdlib_sincosf(extract(x_full, i), &s, &c);
            *sin_result = insert(*sin_result, i, s);
            *cos_result = insert(*cos_result, i, c);
        }
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        const float pi_over_two_vec = 1.57079637050628662109375;
        const float two_over_pi_vec = 0.636619746685028076171875;
        float scaled = x_full * two_over_pi_vec;
        float k_real = floor(scaled);
        int k = (int)k_real;

        // Reduced range version of x
        float x = x_full - k_real * pi_over_two_vec;
        int k_mod4 = k & 3;
        bool cos_usecos = (k_mod4 == 0 || k_mod4 == 2);
        bool sin_usecos = (k_mod4 == 1 || k_mod4 == 3);
        bool sin_flipsign = (k_mod4 > 1);
        bool cos_flipsign = (k_mod4 == 1 || k_mod4 == 2);

        const float one_vec = 1.;
        const float sin_c2 = -0.16666667163372039794921875;
        const float sin_c4 = 8.333347737789154052734375e-3;
        const float sin_c6 = -1.9842604524455964565277099609375e-4;
        const float sin_c8 = 2.760012648650445044040679931640625e-6;
        const float sin_c10 = -2.50293279435709337121807038784027099609375e-8;

        const float cos_c2 = -0.5;
        const float cos_c4 = 4.166664183139801025390625e-2;
        const float cos_c6 = -1.388833043165504932403564453125e-3;
        const float cos_c8 = 2.47562347794882953166961669921875e-5;
        const float cos_c10 = -2.59630184018533327616751194000244140625e-7;

        float x2 = x * x;

        float sin_formula = x2 * sin_c10 + sin_c8;
        float cos_formula = x2 * cos_c10 + cos_c8;
        sin_formula = x2 * sin_formula + sin_c6;
        cos_formula = x2 * cos_formula + cos_c6;

        sin_formula = x2 * sin_formula + sin_c4;
        cos_formula = x2 * cos_formula + cos_c4;

        sin_formula = x2 * sin_formula + sin_c2;
        cos_formula = x2 * cos_formula + cos_c2;

        sin_formula = x2 * sin_formula + one_vec;
        cos_formula = x2 * cos_formula + one_vec;

        sin_formula *= x;

        *sin_result = sin_usecos ? cos_formula : sin_formula;
        *cos_result = cos_usecos ? cos_formula : sin_formula;

        *sin_result = sin_flipsign ? -*sin_result : *sin_result;
        *cos_result = cos_flipsign ? -*cos_result : *cos_result;
    }
}


__declspec(safe)
static inline void sincos(uniform float x_full, uniform float * uniform sin_result,
                          uniform float * uniform cos_result) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        __stdlib_sincosf(x_full, sin_result, cos_result);
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        const uniform float pi_over_two_vec = 1.57079637050628662109375;
        const uniform float two_over_pi_vec = 0.636619746685028076171875;
        uniform float scaled = x_full * two_over_pi_vec;
        uniform float k_real = floor(scaled);
        uniform int k = (uniform int)k_real;

        // Reduced range version of x
        uniform float x = x_full - k_real * pi_over_two_vec;
        uniform int k_mod4 = k & 3;
        uniform bool cos_usecos = (k_mod4 == 0 || k_mod4 == 2);
        uniform bool sin_usecos = (k_mod4 == 1 || k_mod4 == 3);
        uniform bool sin_flipsign = (k_mod4 > 1);
        uniform bool cos_flipsign = (k_mod4 == 1 || k_mod4 == 2);

        const uniform float one_vec = 1.;
        const uniform float sin_c2 = -0.16666667163372039794921875;
        const uniform float sin_c4 = 8.333347737789154052734375e-3;
        const uniform float sin_c6 = -1.9842604524455964565277099609375e-4;
        const uniform float sin_c8 = 2.760012648650445044040679931640625e-6;
        const uniform float sin_c10 = -2.50293279435709337121807038784027099609375e-8;

        const uniform float cos_c2 = -0.5;
        const uniform float cos_c4 = 4.166664183139801025390625e-2;
        const uniform float cos_c6 = -1.388833043165504932403564453125e-3;
        const uniform float cos_c8 = 2.47562347794882953166961669921875e-5;
        const uniform float cos_c10 = -2.59630184018533327616751194000244140625e-7;

        uniform float x2 = x * x;

        uniform float sin_formula = x2 * sin_c10 + sin_c8;
        uniform float cos_formula = x2 * cos_c10 + cos_c8;
        sin_formula = x2 * sin_formula + sin_c6;
        cos_formula = x2 * cos_formula + cos_c6;

        sin_formula = x2 * sin_formula + sin_c4;
        cos_formula = x2 * cos_formula + cos_c4;

        sin_formula = x2 * sin_formula + sin_c2;
        cos_formula = x2 * cos_formula + cos_c2;

        sin_formula = x2 * sin_formula + one_vec;
        cos_formula = x2 * cos_formula + one_vec;

        sin_formula *= x;

        *sin_result = sin_usecos ? cos_formula : sin_formula;
        *cos_result = cos_usecos ? cos_formula : sin_formula;

        *sin_result = sin_flipsign ? -*sin_result : *sin_result;
        *cos_result = cos_flipsign ? -*cos_result : *cos_result;
    }
}


__declspec(safe)
static inline float tan(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_tan(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_tanf(extract(x_full, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        const float pi_over_four_vec = 0.785398185253143310546875;
        const float four_over_pi_vec = 1.27323949337005615234375;

        bool x_lt_0 = x_full < 0.;
        float y = x_lt_0 ? -x_full : x_full;
        float scaled = y * four_over_pi_vec;

        float k_real = floor(scaled);
        int k = (int)k_real;

        float x = y - k_real * pi_over_four_vec;

        // if k & 1, x -= Pi/4
        bool need_offset = (k & 1) != 0;
        x = need_offset ? x - pi_over_four_vec : x;

        // if k & 3 == (0 or 3) let z = tan_In...(y) otherwise z = -cot_In0To...
        int k_mod4 = k & 3;
        bool use_cotan = (k_mod4 == 1) || (k_mod4 == 2);

        const float one_vec = 1.0;

        const float tan_c2 = 0.33333075046539306640625;
        const float tan_c4 = 0.13339905440807342529296875;
        const float tan_c6 = 5.3348250687122344970703125e-2;
        const float tan_c8 = 2.46033705770969390869140625e-2;
        const float tan_c10 = 2.892402000725269317626953125e-3;
        const float tan_c12 = 9.500005282461643218994140625e-3;

        const float cot_c2 = -0.3333333432674407958984375;
        const float cot_c4 = -2.222204394638538360595703125e-2;
        const float cot_c6 = -2.11752182804048061370849609375e-3;
        const float cot_c8 = -2.0846328698098659515380859375e-4;
        const float cot_c10 = -2.548247357481159269809722900390625e-5;
        const float cot_c12 = -3.5257363606433500535786151885986328125e-7;

        float x2 = x * x;
        float z;
        cif (use_cotan) {
            float cot_val = x2 * cot_c12 + cot_c10;
            cot_val = x2 * cot_val + cot_c8;
            cot_val = x2 * cot_val + cot_c6;
            cot_val = x2 * cot_val + cot_c4;
            cot_val = x2 * cot_val + cot_c2;
            cot_val = x2 * cot_val + one_vec;
            // The equation is for x * cot(x) but we need -x * cot(x) for the tan part.
            cot_val /= -x;
            z = cot_val;
        } else {
            float tan_val = x2 * tan_c12 + tan_c10;
            tan_val = x2 * tan_val + tan_c8;
            tan_val = x2 * tan_val + tan_c6;
            tan_val = x2 * tan_val + tan_c4;
            tan_val = x2 * tan_val + tan_c2;
            tan_val = x2 * tan_val + one_vec;
            // Equation was for tan(x)/x
            tan_val *= x;
            z = tan_val;
        }
        return x_lt_0 ? -z : z;
    }
}


__declspec(safe)
static inline uniform float tan(uniform float x_full) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_tanf(x_full);
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        const uniform float pi_over_four_vec = 0.785398185253143310546875;
        const uniform float four_over_pi_vec = 1.27323949337005615234375;

        uniform bool x_lt_0 = x_full < 0.;
        uniform float y = x_lt_0 ? -x_full : x_full;
        uniform float scaled = y * four_over_pi_vec;

        uniform float k_real = floor(scaled);
        uniform int k = (int)k_real;

        uniform float x = y - k_real * pi_over_four_vec;

        // if k & 1, x -= Pi/4
        uniform bool need_offset = (k & 1) != 0;
        x = need_offset ? x - pi_over_four_vec : x;

        // if k & 3 == (0 or 3) let z = tan_In...(y) otherwise z = -cot_In0To...
        uniform int k_mod4 = k & 3;
        uniform bool use_cotan = (k_mod4 == 1) || (k_mod4 == 2);

        const uniform float one_vec = 1.0;

        const uniform float tan_c2 = 0.33333075046539306640625;
        const uniform float tan_c4 = 0.13339905440807342529296875;
        const uniform float tan_c6 = 5.3348250687122344970703125e-2;
        const uniform float tan_c8 = 2.46033705770969390869140625e-2;
        const uniform float tan_c10 = 2.892402000725269317626953125e-3;
        const uniform float tan_c12 = 9.500005282461643218994140625e-3;

        const uniform float cot_c2 = -0.3333333432674407958984375;
        const uniform float cot_c4 = -2.222204394638538360595703125e-2;
        const uniform float cot_c6 = -2.11752182804048061370849609375e-3;
        const uniform float cot_c8 = -2.0846328698098659515380859375e-4;
        const uniform float cot_c10 = -2.548247357481159269809722900390625e-5;
        const uniform float cot_c12 = -3.5257363606433500535786151885986328125e-7;

        uniform float x2 = x * x;
        uniform float z;
        if (use_cotan) {
            uniform float cot_val = x2 * cot_c12 + cot_c10;
            cot_val = x2 * cot_val + cot_c8;
            cot_val = x2 * cot_val + cot_c6;
            cot_val = x2 * cot_val + cot_c4;
            cot_val = x2 * cot_val + cot_c2;
            cot_val = x2 * cot_val + one_vec;
            // The equation is for x * cot(x) but we need -x * cot(x) for the tan part.
            cot_val /= -x;
            z = cot_val;
        } else {
            uniform float tan_val = x2 * tan_c12 + tan_c10;
            tan_val = x2 * tan_val + tan_c8;
            tan_val = x2 * tan_val + tan_c6;
            tan_val = x2 * tan_val + tan_c4;
            tan_val = x2 * tan_val + tan_c2;
            tan_val = x2 * tan_val + one_vec;
            // Equation was for tan(x)/x
            tan_val *= x;
            z = tan_val;
        }
        return x_lt_0 ? -z : z;
    }
}


__declspec(safe)
static inline float atan(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_atan(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_atanf(extract(x_full, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        const float pi_over_two_vec = 1.57079637050628662109375;
        // atan(-x) = -atan(x) (so flip from negative to positive first)
        // if x > 1 -> atan(x) = Pi/2 - atan(1/x)
        bool x_neg = x_full < 0;
        float x_flipped = x_neg ? -x_full : x_full;

        bool x_gt_1 = x_flipped > 1.;
        float x = x_gt_1 ? 1./x_flipped : x_flipped;

        // These coefficients approximate atan(x)/x
        const float atan_c0 = 0.99999988079071044921875;
        const float atan_c2 = -0.3333191573619842529296875;
        const float atan_c4 = 0.199689209461212158203125;
        const float atan_c6 = -0.14015688002109527587890625;
        const float atan_c8 = 9.905083477497100830078125e-2;
        const float atan_c10 = -5.93664981424808502197265625e-2;
        const float atan_c12 = 2.417283318936824798583984375e-2;
        const float atan_c14 = -4.6721356920897960662841796875e-3;

        float x2 = x * x;
        float result = x2 * atan_c14 + atan_c12;
        result = x2 * result + atan_c10;
        result = x2 * result + atan_c8;
        result = x2 * result + atan_c6;
        result = x2 * result + atan_c4;
        result = x2 * result + atan_c2;
        result = x2 * result + atan_c0;
        result *= x;

        result = x_gt_1 ? pi_over_two_vec - result : result;
        result = x_neg ? -result : result;
        return result;
    }
}


__declspec(safe)
static inline uniform float atan(uniform float x_full) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_atanf(x_full);
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        const uniform float pi_over_two_vec = 1.57079637050628662109375;
        // atan(-x) = -atan(x) (so flip from negative to positive first)
        // if x > 1 -> atan(x) = Pi/2 - atan(1/x)
        uniform bool x_neg = x_full < 0;
        uniform float x_flipped = x_neg ? -x_full : x_full;

        uniform bool x_gt_1 = x_flipped > 1.;
        uniform float x = x_gt_1 ? 1./x_flipped : x_flipped;

        // These coefficients approximate atan(x)/x
        const uniform float atan_c0 = 0.99999988079071044921875;
        const uniform float atan_c2 = -0.3333191573619842529296875;
        const uniform float atan_c4 = 0.199689209461212158203125;
        const uniform float atan_c6 = -0.14015688002109527587890625;
        const uniform float atan_c8 = 9.905083477497100830078125e-2;
        const uniform float atan_c10 = -5.93664981424808502197265625e-2;
        const uniform float atan_c12 = 2.417283318936824798583984375e-2;
        const uniform float atan_c14 = -4.6721356920897960662841796875e-3;

        uniform float x2 = x * x;
        uniform float result = x2 * atan_c14 + atan_c12;
        result = x2 * result + atan_c10;
        result = x2 * result + atan_c8;
        result = x2 * result + atan_c6;
        result = x2 * result + atan_c4;
        result = x2 * result + atan_c2;
        result = x2 * result + atan_c0;
        result *= x;

        result = x_gt_1 ? pi_over_two_vec - result : result;
        result = x_neg ? -result : result;
        return result;
    }
}


__declspec(safe)
static inline float atan2(float y, float x) {
    if (__math_lib == __math_lib_svml) {
        return __svml_atan2(y, x);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_atan2f(extract(y, i), extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        const float pi_vec = 3.1415926536;
        const float pi_over_two_vec = 1.5707963267;
        // atan2(y, x) =
        //
        // atan2(y > 0, x = +-0) ->  Pi/2
        // atan2(y < 0, x = +-0) -> -Pi/2
        // atan2(y = +-0, x < +0) -> +-Pi
        // atan2(y = +-0, x >= +0) -> +-0
        //
        // atan2(y >= 0, x < 0) ->  Pi + atan(y/x)
        // atan2(y <  0, x < 0) -> -Pi + atan(y/x)
        // atan2(y, x > 0) -> atan(y/x)
        //
        // and then a bunch of code for dealing with infinities.
        float y_over_x = y/x;
        float atan_arg = atan(y_over_x);
        bool x_lt_0 = x < 0;
        bool y_lt_0 = y < 0;
        float offset = x_lt_0 ? (y_lt_0 ? -pi_vec : pi_vec) : 0;
        return offset + atan_arg;
    }
}


__declspec(safe)
static inline uniform float atan2(uniform float y, uniform float x) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_atan2f(y, x);
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        const uniform float pi_vec = 3.1415927410125732421875;
        const uniform float pi_over_two_vec = 1.57079637050628662109375;

        uniform float y_over_x = y/x;
        uniform float atan_arg = atan(y_over_x);
        uniform bool x_lt_0 = x < 0;
        uniform bool y_lt_0 = y < 0;
        uniform float offset = x_lt_0 ? (y_lt_0 ? -pi_vec : pi_vec) : 0;
        return offset + atan_arg;
    }
}


__declspec(safe)
static inline float exp(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_exp(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_expf(extract(x_full, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc_fast) {
        float z = floor(1.44269504088896341f * x_full + 0.5f); 
        int n;
        x_full -= z * 0.693359375f;
        x_full -= z * -2.12194440e-4f;
        n = (int)z;

        z = x_full * x_full;
        z = (((((1.9875691500E-4f  * x_full + 1.3981999507E-3f) * x_full +
                8.3334519073E-3f) * x_full + 4.1665795894E-2f) * x_full +
              1.6666665459E-1f) * x_full + 5.0000001201E-1f) * z + x_full + 1.f;
        x_full = ldexp(z, n);
        return x_full;
    }
    else if (__math_lib == __math_lib_ispc) {
        const float ln2_part1 = 0.6931457519;
        const float ln2_part2 = 1.4286067653e-6;
        const float one_over_ln2 = 1.44269502162933349609375;

        float scaled = x_full * one_over_ln2;
        float k_real = floor(scaled);
        int k = (int)k_real;

        // Reduced range version of x
        float x = x_full - k_real * ln2_part1;
        x -= k_real * ln2_part2;

        // These coefficients are for e^x in [0, ln(2)]
        const float one = 1.;
        const float c2 = 0.4999999105930328369140625;
        const float c3 = 0.166668415069580078125;
        const float c4 = 4.16539050638675689697265625e-2;
        const float c5 = 8.378830738365650177001953125e-3;
        const float c6 = 1.304379315115511417388916015625e-3;
        const float c7 = 2.7555381529964506626129150390625e-4;

        float result = x * c7 + c6;
        result = x * result + c5;
        result = x * result + c4;
        result = x * result + c3;
        result = x * result + c2;
        result = x * result + one;
        result = x * result + one;

        // Compute 2^k (should differ for float and double, but I'll avoid
        // it for now and just do floats)
        const int fpbias = 127;
        int biased_n = k + fpbias;
        bool overflow = k > fpbias;
        // Minimum exponent is -126, so if k is <= -127 (k + 127 <= 0)
        // we've got underflow. -127 * ln(2) -> -88.02. So the most
        // negative float input that doesn't result in zero is like -88.
        bool underflow = (biased_n <= 0);
        const int InfBits = 0x7f800000;
        biased_n <<= 23;
        // Reinterpret this thing as float
        float two_to_the_n = floatbits(biased_n);
        // Handle both doubles and floats (hopefully eliding the copy for float)
        float elemtype_2n = two_to_the_n;
        result *= elemtype_2n;
        result = overflow ? floatbits(InfBits) : result;
        result = underflow ? 0. : result;
        return result;
    }
}

__declspec(safe)
static inline uniform float exp(uniform float x_full) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_expf(x_full);
    }
    else if (__math_lib == __math_lib_ispc_fast) {
        uniform float z = floor(1.44269504088896341f * x_full + 0.5f); 
        uniform int n;
        x_full -= z * 0.693359375f;
        x_full -= z * -2.12194440e-4f;
        n = (int)z;

        z = x_full * x_full;
        z = (((((1.9875691500E-4f  * x_full + 1.3981999507E-3f) * x_full +
                8.3334519073E-3f) * x_full + 4.1665795894E-2f) * x_full +
              1.6666665459E-1f) * x_full + 5.0000001201E-1f) * z + x_full + 1.f;
        x_full = ldexp(z, n);
        return x_full;
    }
    else if (__math_lib == __math_lib_ispc) {
        const uniform float ln2_part1 = 0.6931457519;
        const uniform float ln2_part2 = 1.4286067653e-6;
        const uniform float one_over_ln2 = 1.44269502162933349609375;

        uniform float scaled = x_full * one_over_ln2;
        uniform float k_real = floor(scaled);
        uniform int k = (uniform int)k_real;

        // Reduced range version of x
        uniform float x = x_full - k_real * ln2_part1;
        x -= k_real * ln2_part2;

        // These coefficients are for e^x in [0, ln(2)]
        const uniform float one = 1.;
        const uniform float c2 = 0.4999999105930328369140625;
        const uniform float c3 = 0.166668415069580078125;
        const uniform float c4 = 4.16539050638675689697265625e-2;
        const uniform float c5 = 8.378830738365650177001953125e-3;
        const uniform float c6 = 1.304379315115511417388916015625e-3;
        const uniform float c7 = 2.7555381529964506626129150390625e-4;

        uniform float result = x * c7 + c6;
        result = x * result + c5;
        result = x * result + c4;
        result = x * result + c3;
        result = x * result + c2;
        result = x * result + one;
        result = x * result + one;

        // Compute 2^k (should differ for uniform float and double, but I'll avoid
        // it for now and just do uniform floats)
        const uniform int fpbias = 127;
        uniform int biased_n = k + fpbias;
        uniform bool overflow = k > fpbias;
        // Minimum exponent is -126, so if k is <= -127 (k + 127 <= 0)
        // we've got underflow. -127 * ln(2) -> -88.02. So the most
        // negative uniform float input that doesn't result in zero is like -88.
        uniform bool underflow = (biased_n <= 0);
        const uniform int InfBits = 0x7f800000;
        biased_n <<= 23;
        // Reuniform interpret this thing as uniform float
        uniform float two_to_the_n = floatbits(biased_n);
        // Handle both doubles and uniform floats (hopefully eliding the copy for uniform float)
        uniform float elemtype_2n = two_to_the_n;
        result *= elemtype_2n;
        result = overflow ? floatbits(InfBits) : result;
        result = underflow ? 0. : result;
        return result;
    }
}

// Range reduction for logarithms takes log(x) -> log(2^n * y) -> n
// * log(2) + log(y) where y is the reduced range (usually in [1/2,
// 1)).
__declspec(safe)
static inline void __range_reduce_log(float input, varying float * uniform reduced, 
                                      varying int * uniform exponent) {
    int int_version = intbits(input);
    // single precision = SEEE EEEE EMMM MMMM MMMM MMMM MMMM MMMM
    // exponent mask    = 0111 1111 1000 0000 0000 0000 0000 0000
    //                    0x7  0xF  0x8  0x0  0x0  0x0  0x0  0x0
    // non-exponent     = 1000 0000 0111 1111 1111 1111 1111 1111
    //                  = 0x8  0x0  0x7  0xF  0xF  0xF  0xF  0xF

    //const int exponent_mask(0x7F800000)
    static const int nonexponent_mask = 0x807FFFFF;

    // We want the reduced version to have an exponent of -1 which is -1 + 127 after biasing or 126
    static const int exponent_neg1 = (126 << 23);
    // NOTE(boulos): We don't need to mask anything out since we know
    // the sign bit has to be 0. If it's 1, we need to return infinity/nan
    // anyway (log(x), x = +-0 -> infinity, x < 0 -> NaN).
    int biased_exponent = int_version >> 23; // This number is [0, 255] but it means [-127, 128]

    int offset_exponent = biased_exponent + 1; // Treat the number as if it were 2^{e+1} * (1.m)/2
    *exponent = offset_exponent - 127; // get the real value

    // Blend the offset_exponent with the original input (do this in
    // int for now, until I decide if float can have & and &not)
    int blended = (int_version & nonexponent_mask) | (exponent_neg1);
    *reduced = floatbits(blended);
}



__declspec(safe)
static inline void __range_reduce_log(uniform float input, uniform float * uniform reduced, 
                                      uniform int * uniform exponent) {
    uniform int int_version = intbits(input);
    static const uniform int nonexponent_mask = 0x807FFFFF;

    static const uniform int exponent_neg1 = (126 << 23);
    uniform int biased_exponent = int_version >> 23;
    uniform int offset_exponent = biased_exponent + 1;
    *exponent = offset_exponent - 127; // get the real value

    uniform int blended = (int_version & nonexponent_mask) | (exponent_neg1);
    *reduced = floatbits(blended);
}


__declspec(safe)
static inline float log(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_log(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_logf(extract(x_full, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc_fast) {
        int e;
        x_full = frexp(x_full, &e);
    
        int x_smaller_SQRTHF = (0.707106781186547524f > x_full) ? 0xffffffff : 0;
        e += x_smaller_SQRTHF;
        int ix_add = intbits(x_full);
        ix_add &= x_smaller_SQRTHF;
        x_full += floatbits(ix_add) - 1.f;

        float z = x_full * x_full;
        float y =
            ((((((((7.0376836292E-2f * x_full
                    + -1.1514610310E-1f) * x_full
                   + 1.1676998740E-1f) * x_full
                  + -1.2420140846E-1f) * x_full
                 + 1.4249322787E-1f) * x_full
                + -1.6668057665E-1f) * x_full
               + 2.0000714765E-1f) * x_full
              + -2.4999993993E-1f) * x_full
             + 3.3333331174E-1f) * x_full * z;

        float fe = (float)e;
        y += fe * -2.12194440e-4;
        y -= 0.5f * z;
        z  = x_full + y;
        return z + 0.693359375 * fe;
    } 
    else if (__math_lib == __math_lib_ispc) {
        float reduced;
        int exponent;

        const int NaN_bits = 0x7fc00000;
        const int Neg_Inf_bits = 0xFF800000;
        const float NaN = floatbits(NaN_bits);
        const float neg_inf = floatbits(Neg_Inf_bits);
        bool use_nan = x_full < 0.;
        bool use_inf = x_full == 0.;
        bool exceptional = use_nan || use_inf;
        const float one = 1.0;

        float patched = exceptional ? one : x_full;
        __range_reduce_log(patched, &reduced, &exponent);

        const float ln2 = 0.693147182464599609375;

        float x1 = one - reduced;
        const float c1 = 0.50000095367431640625;
        const float c2 = 0.33326041698455810546875;
        const float c3 = 0.2519190013408660888671875;
        const float c4 = 0.17541764676570892333984375;
        const float c5 = 0.3424419462680816650390625;
        const float c6 = -0.599632322788238525390625;
        const float c7 = +1.98442304134368896484375;
        const float c8 = -2.4899270534515380859375;
        const float c9 = +1.7491014003753662109375;

        float result = x1 * c9 + c8;
        result = x1 * result + c7;
        result = x1 * result + c6;
        result = x1 * result + c5;
        result = x1 * result + c4;
        result = x1 * result + c3;
        result = x1 * result + c2;
        result = x1 * result + c1;
        result = x1 * result + one;

        // Equation was for -(ln(red)/(1-red))
        result *= -x1;
        result += (float)(exponent) * ln2;

        return exceptional ? (use_nan ? NaN : neg_inf) : result;
    }
}

__declspec(safe)
static inline uniform float log(uniform float x_full) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_logf(x_full);
    }
    else if (__math_lib == __math_lib_ispc_fast) {
        uniform int e;
        x_full = frexp(x_full, &e);
    
        uniform int x_smaller_SQRTHF = (0.707106781186547524f > x_full) ? 0xffffffff : 0;
        e += x_smaller_SQRTHF;
        uniform int ix_add = intbits(x_full);
        ix_add &= x_smaller_SQRTHF;
        x_full += floatbits(ix_add) - 1.f;

        uniform float z = x_full * x_full;
        uniform float y =
            ((((((((7.0376836292E-2f * x_full
                    + -1.1514610310E-1f) * x_full
                   + 1.1676998740E-1f) * x_full
                  + -1.2420140846E-1f) * x_full
                 + 1.4249322787E-1f) * x_full
                + -1.6668057665E-1f) * x_full
               + 2.0000714765E-1f) * x_full
              + -2.4999993993E-1f) * x_full
             + 3.3333331174E-1f) * x_full * z;

        uniform float fe = (uniform float)e;
        y += fe * -2.12194440e-4;
        y -= 0.5f * z;
        z  = x_full + y;
        return z + 0.693359375 * fe;
    }
    else if (__math_lib == __math_lib_ispc) {
        uniform float reduced;
        uniform int exponent;

        const uniform int NaN_bits = 0x7fc00000;
        const uniform int Neg_Inf_bits = 0xFF800000;
        const uniform float NaN = floatbits(NaN_bits);
        const uniform float neg_inf = floatbits(Neg_Inf_bits);
        uniform bool use_nan = x_full < 0.;
        uniform bool use_inf = x_full == 0.;
        uniform bool exceptional = use_nan || use_inf;
        const uniform float one = 1.0;

        uniform float patched = exceptional ? one : x_full;
        __range_reduce_log(patched, &reduced, &exponent);

        const uniform float ln2 = 0.693147182464599609375;

        uniform float x1 = one - reduced;
        const uniform float c1 = 0.50000095367431640625;
        const uniform float c2 = 0.33326041698455810546875;
        const uniform float c3 = 0.2519190013408660888671875;
        const uniform float c4 = 0.17541764676570892333984375;
        const uniform float c5 = 0.3424419462680816650390625;
        const uniform float c6 = -0.599632322788238525390625;
        const uniform float c7 = +1.98442304134368896484375;
        const uniform float c8 = -2.4899270534515380859375;
        const uniform float c9 = +1.7491014003753662109375;

        uniform float result = x1 * c9 + c8;
        result = x1 * result + c7;
        result = x1 * result + c6;
        result = x1 * result + c5;
        result = x1 * result + c4;
        result = x1 * result + c3;
        result = x1 * result + c2;
        result = x1 * result + c1;
        result = x1 * result + one;

        // Equation was for -(ln(red)/(1-red))
        result *= -x1;
        result += (uniform float)(exponent) * ln2;

        return exceptional ? (use_nan ? NaN : neg_inf) : result;
    }
}

__declspec(safe)
static inline float pow(float a, float b) {
    if (__math_lib == __math_lib_svml) {
        return __svml_pow(a, b);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        __foreach_active (i) {
            uniform float r = __stdlib_powf(extract(a, i), extract(b, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        return exp(b * log(a));
    }
}

__declspec(safe)
static inline uniform float pow(uniform float a, uniform float b) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_powf(a, b);
    }
    else if (__math_lib == __math_lib_ispc || 
             __math_lib == __math_lib_ispc_fast) {
        return exp(b * log(a));
    }
}

///////////////////////////////////////////////////////////////////////////
// Transcendentals (double precision)

__declspec(safe)
static inline double sqrt(double v) {
    return __sqrt_varying_double(v);
}

__declspec(safe)
static inline uniform double sqrt(uniform double v) {
    return __sqrt_uniform_double(v);
}

__declspec(safe)
static inline double ldexp(double x, int n) {
    unsigned int64 ex = 0x7ff0000000000000;
    unsigned int64 ix = intbits(x);
    ex &= ix;
    ix = ix & ~0x7ff0000000000000;  // clear exponent
    int64 n64 = ((int64)n << 52) + ex;
    ix |= n64; // insert new exponent
    return doublebits(ix);
}

__declspec(safe)
static inline uniform double ldexp(uniform double x, uniform int n) {
    uniform unsigned int64 ex = 0x7ff0000000000000;
    uniform unsigned int64 ix = intbits(x);
    ex &= ix;
    ix = ix & ~0x7ff0000000000000;  // clear exponent
    uniform int64 n64 = ((int64)n << 52) + ex;
    ix |= n64; // insert new exponent
    return doublebits(ix);
}

__declspec(safe)
static inline double frexp(double x, varying int * uniform pw2) {
    unsigned int64 ex = 0x7ff0000000000000;              // exponent mask
    unsigned int64 ix = intbits(x);
    ex &= ix;
    ix &= ~0x7ff0000000000000;  // clear exponent
    *pw2 = (int)(ex >> 52) - 1022; // compute exponent
    ix |= 0x3fe0000000000000;         // insert exponent +1 in x
    return doublebits(ix);
}

__declspec(safe)
static inline uniform double frexp(uniform double x, uniform int * uniform pw2) {
    uniform unsigned int64 ex = 0x7ff0000000000000;              // exponent mask
    uniform unsigned int64 ix = intbits(x);
    ex &= ix;
    ix &= ~0x7ff0000000000000;  // clear exponent
    *pw2 = (int)(ex >> 52) - 1022; // compute exponent
    ix |= 0x3fe0000000000000;         // insert exponent +1 in x
    return doublebits(ix);
}

__declspec(safe)
static inline double sin(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return sin((float)x);
    else {
        double ret;
        __foreach_active (i) {
            uniform double r = __stdlib_sin(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

__declspec(safe)
static inline uniform double sin(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return sin((float)x);
    else
        return __stdlib_sin(x);
}

__declspec(safe)
static inline double cos(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return cos((float)x);
    else {
        double ret;
        __foreach_active (i) {
            uniform double r = __stdlib_cos(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

__declspec(safe)
static inline uniform double cos(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return cos((float)x);
    else
        return __stdlib_cos(x);
}

__declspec(safe)
static inline void sincos(double x, varying double * uniform sin_result,
                          varying double * uniform cos_result) {
    if (__math_lib == __math_lib_ispc_fast) {
        float sr, cr;
        sincos((float)x, &sr, &cr);
        *sin_result = sr;
        *cos_result = cr;
    }
    else {
        __foreach_active (i) {
            uniform double sr, cr;
            __stdlib_sincos(extract(x, i), &sr, &cr);
            *sin_result = insert(*sin_result, i, sr);
            *cos_result = insert(*cos_result, i, cr);
        }
    }
}

__declspec(safe)
static inline void sincos(uniform double x, uniform double * uniform sin_result,
                          uniform double * uniform cos_result) {
    if (__math_lib == __math_lib_ispc_fast) {
        uniform float sr, cr;
        sincos((uniform float)x, &sr, &cr);
        *sin_result = sr;
        *cos_result = cr;
    }
    else
        __stdlib_sincos(x, sin_result, cos_result);
}

__declspec(safe)
static inline double tan(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return tan((float)x);
    else {
        double ret;
        __foreach_active (i) {
            uniform double r = __stdlib_tan(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

__declspec(safe)
static inline uniform double tan(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return tan((float)x);
    else
        return __stdlib_tan(x);
}

__declspec(safe)
static inline double atan(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return atan((float)x);
    else {
        double ret;
        __foreach_active (i) {
            uniform double r = __stdlib_atan(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

__declspec(safe)
static inline uniform double atan(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return atan((float)x);
    else
        return __stdlib_atan(x);
}

__declspec(safe)
static inline double atan2(double y, double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return atan2((float)y, (float)x);
    else {
        double ret;
        __foreach_active (i) {
            uniform double r = __stdlib_atan2(extract(y, i), extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

__declspec(safe)
static inline uniform double atan2(uniform double y, uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return atan2((float)y, (float)x);
    else
        return __stdlib_atan2(y, x);
}

__declspec(safe)
static inline double exp(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return exp((float)x);
    else {
        double ret;
        __foreach_active (i) {
            uniform double r = __stdlib_exp(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

__declspec(safe)
static inline uniform double exp(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return exp((float)x);
    else
        return __stdlib_exp(x);
}

__declspec(safe)
static inline double log(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return log((float)x);
    else {
        double ret;
        __foreach_active (i) {
            uniform double r = __stdlib_log(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

__declspec(safe)
static inline uniform double log(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return log((float)x);
    else
        return __stdlib_log(x);
}

__declspec(safe)
static inline double pow(double a, double b) {
    if (__math_lib == __math_lib_ispc_fast)
        return pow((float)a, (float)b);
    else {
        double ret;
        __foreach_active (i) {
            uniform double r = __stdlib_pow(extract(a, i), extract(b, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

__declspec(safe)
static inline uniform double pow(uniform double a, uniform double b) {
    if (__math_lib == __math_lib_ispc_fast)
        return pow((float)a, (float)b);
    else
        return __stdlib_pow(a, b);
}

///////////////////////////////////////////////////////////////////////////
// half-precision floats

__declspec(safe)
static inline uniform float half_to_float(uniform unsigned int16 h) {
    if (__have_native_half) {
        return __half_to_float_uniform(h);
    }
    else {
        // https://gist.github.com/2144712
        // Fabian "ryg" Giesen.
        static const uniform unsigned int32 shifted_exp = 0x7c00 << 13; // exponent mask after shift

        uniform int32 o = ((int32)(h & 0x7fff)) << 13;     // exponent/mantissa bits
        uniform unsigned int32 exp = shifted_exp & o;   // just the exponent
        o += (127 - 15) << 23;        // exponent adjust

        // handle exponent special cases
        if (exp == shifted_exp) // Inf/NaN?
            o += (128 - 16) << 23;    // extra exp adjust
        else if (exp == 0) { // Zero/Denormal?
            o += 1 << 23;             // extra exp adjust
            o = intbits(floatbits(o) - floatbits(113 << 23)); // renormalize
        }

        o |= ((int32)(h & 0x8000)) << 16;    // sign bit
        return floatbits(o);
    }
}

__declspec(safe)
static inline float half_to_float(unsigned int16 h) {
    if (__have_native_half) {
        return __half_to_float_varying((unsigned int16)h);
    }
    else {
        // https://gist.github.com/2144712
        // Fabian "ryg" Giesen.

        const unsigned int32 shifted_exp = 0x7c00 << 13; // exponent mask after shift

        int32 o = ((int32)(h & 0x7fff)) << 13;     // exponent/mantissa bits
        unsigned int32 exp = shifted_exp & o;   // just the exponent
        o += (127 - 15) << 23;        // exponent adjust

        int32 infnan_val = o + ((128 - 16) << 23);
        int32 zerodenorm_val = intbits(floatbits(o + (1<<23)) - floatbits(113 << 23));
        int32 reg_val = (exp == 0) ? zerodenorm_val : o;

        int32 sign_bit = ((int32)(h & 0x8000)) << 16;
        return floatbits(((exp == shifted_exp) ? infnan_val : reg_val) | sign_bit);
    }
}


__declspec(safe)
static inline uniform int16 float_to_half(uniform float f) {
    if (__have_native_half) {
        return __float_to_half_uniform(f);
    }
    else {
        // via Fabian "ryg" Giesen.
        // https://gist.github.com/2156668
        uniform unsigned int32 sign_mask = 0x80000000u;
        uniform int32 o;

        uniform int32 fint = intbits(f);
        uniform int32 sign = fint & sign_mask;
        fint ^= sign;

        // NOTE all the integer compares in this function can be safely
        // compiled into signed compares since all operands are below
        // 0x80000000. Important if you want fast straight SSE2 code (since
        // there's no unsigned PCMPGTD).

        // Inf or NaN (all exponent bits set)
        // NaN->qNaN and Inf->Inf
        // unconditional assignment here, will override with right value for
        // the regular case below.
        uniform int32 f32infty = 255 << 23;
        o = (fint > f32infty) ? 0x7e00 : 0x7c00; 

        // (De)normalized number or zero
        // update fint unconditionally to save the blending; we don't need it
        // anymore for the Inf/NaN case anyway.

        const uniform unsigned int32 round_mask = ~0xfffu; 
        const uniform int32 magic = 15 << 23;
        const uniform int32 f16infty = 31 << 23;

        uniform int32 fint2 = intbits(floatbits(fint & round_mask) * floatbits(magic)) - round_mask;
        fint2 = (fint2 > f16infty) ? f16infty : fint2; // Clamp to signed infinity if overflowed

        if (fint < f32infty)
            o = fint2 >> 13; // Take the bits!

        return (o | (sign >> 16));
    }
}


__declspec(safe)
static inline int16 float_to_half(float f) {
    if (__have_native_half) {
        return __float_to_half_varying(f);
    }
    else {
        // via Fabian "ryg" Giesen.
        // https://gist.github.com/2156668
        unsigned int32 sign_mask = 0x80000000u;
        int32 o;

        int32 fint = intbits(f);
        int32 sign = fint & sign_mask;
        fint ^= sign;

        // NOTE all the integer compares in this function can be safely
        // compiled into signed compares since all operands are below
        // 0x80000000. Important if you want fast straight SSE2 code (since
        // there's no unsigned PCMPGTD).

        // Inf or NaN (all exponent bits set)
        // NaN->qNaN and Inf->Inf
        // unconditional assignment here, will override with right value for
        // the regular case below.
        int32 f32infty = 255 << 23;
        o = (fint > f32infty) ? 0x7e00 : 0x7c00; 

        // (De)normalized number or zero
        // update fint unconditionally to save the blending; we don't need it
        // anymore for the Inf/NaN case anyway.

        const unsigned int32 round_mask = ~0xfffu; 
        const int32 magic = 15 << 23;
        const int32 f16infty = 31 << 23;

        // Shift exponent down, denormalize if necessary.
        // NOTE This represents half-float denormals using single precision denormals.
        // The main reason to do this is that there's no shift with per-lane variable
        // shifts in SSE*, which we'd otherwise need. It has some funky side effects
        // though:
        // - This conversion will actually respect the FTZ (Flush To Zero) flag in
        //   MXCSR - if it's set, no half-float denormals will be generated. I'm
        //   honestly not sure whether this is good or bad. It's definitely interesting.
        // - If the underlying HW doesn't support denormals (not an issue with Intel
        //   CPUs, but might be a problem on GPUs or PS3 SPUs), you will always get
        //   flush-to-zero behavior. This is bad, unless you're on a CPU where you don't
        //   care.
        // - Denormals tend to be slow. FP32 denormals are rare in practice outside of things
        //   like recursive filters in DSP - not a typical half-float application. Whether
        //   FP16 denormals are rare in practice, I don't know. Whatever slow path your HW
        //   may or may not have for denormals, this may well hit it.
        float fscale = floatbits(fint & round_mask) * floatbits(magic);
        fscale = min(fscale, floatbits((31 << 23) - 0x1000));
        int32 fint2 = intbits(fscale) - round_mask;

        if (fint < f32infty)
            o = fint2 >> 13; // Take the bits!

        return (o | (sign >> 16));
    }
}


__declspec(safe)
static inline uniform float half_to_float_fast(uniform unsigned int16 h) {
    if (__have_native_half) {
        return __half_to_float_uniform(h);
    }
    else {
        uniform unsigned int32 hs = h & (int32)0x8000u;     // Pick off sign bit
        uniform unsigned int32 hem = h & (int32)0x7fffu;    // Pick off exponent-mantissa bits

        uniform unsigned int32 xs = ((unsigned int32) hs) << 16;
        uniform unsigned int32 xem = ((unsigned int32) hem) << 13;

        xem += 0x38000000;      // (127 - 15) << 23

        return floatbits(xs | xem);
    }
}

__declspec(safe)
static inline float half_to_float_fast(unsigned int16 h) {
    if (__have_native_half) {
        return __half_to_float_varying(h);
    }
    else {
        unsigned int32 hs = h & (int32)0x8000u;     // Pick off sign bit
        unsigned int32 hem = h & (int32)0x7fffu;    // Pick off exponent-mantissa bits

        unsigned int32 xs = ((unsigned int32) hs) << 16;
        unsigned int32 xem = ((unsigned int32) hem) << 13;

        return floatbits(xs | (xem + 0x38000000 /* (127 - 15) << 23 */));
    }
}

__declspec(safe)
static inline uniform int16 float_to_half_fast(uniform float f) {
    if (__have_native_half) {
        return __float_to_half_uniform(f);
    }
    else {
        uniform int32 x = intbits(f);
        uniform unsigned int32 xs = x & 0x80000000u;  // Pick off sign bit
        uniform unsigned int32 xe = x & 0x7F800000u;  // Pick off exponent bits
        uniform unsigned int32 xm = x & 0x007FFFFFu;  // Pick off mantissa bits

        uniform unsigned int32 hs = (xs >> 16); // Sign bit
        // Exponent unbias the single, then bias the halfp
        uniform int32 hes = ((int)(xe >> 23)) - 127 + 15; 
        uniform unsigned int32 he = (hes << 10); // Exponent
        uniform int32 hm = (xm >> 13); // Mantissa
        uniform int32 ret = (hs | he | hm);

        if (xm & 0x00001000u) // Check for rounding
            // Round, might overflow to inf, this is OK
            ret += 1u; 

        return (int16)ret;
    }
}

__declspec(safe)
static inline int16 float_to_half_fast(float f) {
    if (__have_native_half) {
        return __float_to_half_varying(f);
    }
    else {
        int32 x = intbits(f);
        unsigned int32 xs = x & 0x80000000u;  // Pick off sign bit
        unsigned int32 xe = x & 0x7F800000u;  // Pick off exponent bits
        unsigned int32 xm = x & 0x007FFFFFu;  // Pick off mantissa bits

        unsigned int32 hs = (xs >> 16); // Sign bit
        // Exponent unbias the single, then bias the halfp
        int32 hes = ((int)(xe >> 23)) - 127 + 15; 
        unsigned int32 he = (hes << 10); // Exponent
        int32 hm = (xm >> 13); // Mantissa
        int32 ret = (hs | he | hm);

        if (xm & 0x00001000u) // Check for rounding
            // Round, might overflow to inf, this is OK
            ret += 1u; 

        return (int16)ret;
    }
}

///////////////////////////////////////////////////////////////////////////
// float -> srgb8

// https://gist.github.com/2246678, from Fabian "rygorous" Giesen.
//
// The basic ideas are still the same, only this time, we squeeze
// everything into the table, even the linear part of the range; since we
// are approximating the function as piecewise linear anyway, this is
// fairly easy.
//
// In the exact version of the conversion, any value that produces an
// output float less than 0.5 will be rounded to an integer of
// zero. Inverting the linear part of the transform, we get:
//
//   log2(0.5 / (255 * 12.92)) =~ -12.686
//
// which in turn means that any value smaller than about 2^(-12.687) will
// return 0.  What this means is that we can adapt the clamping code to
// just clamp to [2^(-13), 1-eps] and we're covered. This means our table
// needs to cover a range of 13 different exponents from -13 to -1.
//
// The table lookup, storage and interpolation works exactly the same way
// as in the code above.
//
// Max error for the whole function (integer-rounded result minus "exact"
// value, as computed in floats using the official formula): 0.544403 at
// 0x3e9f8000

__declspec(safe)
static inline int
float_to_srgb8(float in)
{
    static const uniform unsigned int table[104] = {
        0x0073000d, 0x007a000d, 0x0080000d, 0x0087000d,
        0x008d000d, 0x0094000d, 0x009a000d, 0x00a1000d,
        0x00a7001a, 0x00b4001a, 0x00c1001a, 0x00ce001a,
        0x00da001a, 0x00e7001a, 0x00f4001a, 0x0101001a,
        0x010e0033, 0x01280033, 0x01410033, 0x015b0033,
        0x01750033, 0x018f0033, 0x01a80033, 0x01c20033,
        0x01dc0067, 0x020f0067, 0x02430067, 0x02760067,
        0x02aa0067, 0x02dd0067, 0x03110067, 0x03440067,
        0x037800ce, 0x03df00ce, 0x044600ce, 0x04ad00ce,
        0x051400ce, 0x057b00c5, 0x05dd00bc, 0x063b00b5,
        0x06970158, 0x07420142, 0x07e30130, 0x087b0120,
        0x090b0112, 0x09940106, 0x0a1700fc, 0x0a9500f2,
        0x0b0f01cb, 0x0bf401ae, 0x0ccb0195, 0x0d950180,
        0x0e56016e, 0x0f0d015e, 0x0fbc0150, 0x10630143,
        0x11070264, 0x1238023e, 0x1357021d, 0x14660201,
        0x156601e9, 0x165a01d3, 0x174401c0, 0x182401af,
        0x18fe0331, 0x1a9602fe, 0x1c1502d2, 0x1d7e02ad,
        0x1ed4028d, 0x201a0270, 0x21520256, 0x227d0240,
        0x239f0443, 0x25c003fe, 0x27bf03c4, 0x29a10392,
        0x2b6a0367, 0x2d1d0341, 0x2ebe031f, 0x304d0300,
        0x31d105b0, 0x34a80555, 0x37520507, 0x39d504c5,
        0x3c37048b, 0x3e7c0458, 0x40a8042a, 0x42bd0401,
        0x44c20798, 0x488e071e, 0x4c1c06b6, 0x4f76065d,
        0x52a50610, 0x55ac05cc, 0x5892058f, 0x5b590559,
        0x5e0c0a23, 0x631c0980, 0x67db08f6, 0x6c55087f,
        0x70940818, 0x74a007bd, 0x787d076c, 0x7c330723,
    };

    static const uniform unsigned int almost_one = 0x3f7fffff;
    
    // Clamp to [2^(-13), 1-eps]; these two values map to 0 and 1, respectively.
    in = max(in, 0.0f);
    in = min(in, floatbits(almost_one));

    // Do the table lookup and unpack bias, scale
    unsigned int tab = table[(intbits(in) - 0x39000000u) >> 20];
    unsigned int bias = (tab >> 16) << 9;
    unsigned int scale = tab & 0xffff;

    // Grab next-highest mantissa bits and perform linear interpolation
    unsigned int t = (intbits(in) >> 12) & 0xff;
    return (bias + scale*t) >> 16;
}


__declspec(safe)
static inline uniform int
float_to_srgb8(uniform float in)
{
    static const uniform unsigned int table[104] = {
        0x0073000d, 0x007a000d, 0x0080000d, 0x0087000d,
        0x008d000d, 0x0094000d, 0x009a000d, 0x00a1000d,
        0x00a7001a, 0x00b4001a, 0x00c1001a, 0x00ce001a,
        0x00da001a, 0x00e7001a, 0x00f4001a, 0x0101001a,
        0x010e0033, 0x01280033, 0x01410033, 0x015b0033,
        0x01750033, 0x018f0033, 0x01a80033, 0x01c20033,
        0x01dc0067, 0x020f0067, 0x02430067, 0x02760067,
        0x02aa0067, 0x02dd0067, 0x03110067, 0x03440067,
        0x037800ce, 0x03df00ce, 0x044600ce, 0x04ad00ce,
        0x051400ce, 0x057b00c5, 0x05dd00bc, 0x063b00b5,
        0x06970158, 0x07420142, 0x07e30130, 0x087b0120,
        0x090b0112, 0x09940106, 0x0a1700fc, 0x0a9500f2,
        0x0b0f01cb, 0x0bf401ae, 0x0ccb0195, 0x0d950180,
        0x0e56016e, 0x0f0d015e, 0x0fbc0150, 0x10630143,
        0x11070264, 0x1238023e, 0x1357021d, 0x14660201,
        0x156601e9, 0x165a01d3, 0x174401c0, 0x182401af,
        0x18fe0331, 0x1a9602fe, 0x1c1502d2, 0x1d7e02ad,
        0x1ed4028d, 0x201a0270, 0x21520256, 0x227d0240,
        0x239f0443, 0x25c003fe, 0x27bf03c4, 0x29a10392,
        0x2b6a0367, 0x2d1d0341, 0x2ebe031f, 0x304d0300,
        0x31d105b0, 0x34a80555, 0x37520507, 0x39d504c5,
        0x3c37048b, 0x3e7c0458, 0x40a8042a, 0x42bd0401,
        0x44c20798, 0x488e071e, 0x4c1c06b6, 0x4f76065d,
        0x52a50610, 0x55ac05cc, 0x5892058f, 0x5b590559,
        0x5e0c0a23, 0x631c0980, 0x67db08f6, 0x6c55087f,
        0x70940818, 0x74a007bd, 0x787d076c, 0x7c330723,
    };

    static const uniform unsigned int almost_one = 0x3f7fffff;
    
    // Clamp to [2^(-13), 1-eps]; these two values map to 0 and 1, respectively.
    in = max(in, 0.0f);
    in = min(in, floatbits(almost_one));

    // Do the table lookup and unpack bias, scale
    uniform unsigned int tab = table[(intbits(in) - 0x39000000u) >> 20];
    uniform unsigned int bias = (tab >> 16) << 9;
    uniform unsigned int scale = tab & 0xffff;

    // Grab next-highest mantissa bits and perform linear interpolation
    uniform unsigned int t = (intbits(in) >> 12) & 0xff;
    return (bias + scale*t) >> 16;
}

///////////////////////////////////////////////////////////////////////////
// RNG stuff

struct RNGState {
    unsigned int z1, z2, z3, z4;
};

static inline unsigned int random(varying RNGState * uniform state)
{
    unsigned int b;

    b  = ((state->z1 << 6) ^ state->z1) >> 13;
    state->z1 = ((state->z1 & 4294967294U) << 18) ^ b;
    b  = ((state->z2 << 2) ^ state->z2) >> 27; 
    state->z2 = ((state->z2 & 4294967288U) << 2) ^ b;
    b  = ((state->z3 << 13) ^ state->z3) >> 21;
    state->z3 = ((state->z3 & 4294967280U) << 7) ^ b;
    b  = ((state->z4 << 3) ^ state->z4) >> 12;
    state->z4 = ((state->z4 & 4294967168U) << 13) ^ b;
    return (state->z1 ^ state->z2 ^ state->z3 ^ state->z4);
}

static inline uniform unsigned int random(uniform RNGState * uniform state)
{
    uniform unsigned int b;

    b  = ((state->z1 << 6) ^ state->z1) >> 13;
    state->z1 = ((state->z1 & 4294967294U) << 18) ^ b;
    b  = ((state->z2 << 2) ^ state->z2) >> 27; 
    state->z2 = ((state->z2 & 4294967288U) << 2) ^ b;
    b  = ((state->z3 << 13) ^ state->z3) >> 21;
    state->z3 = ((state->z3 & 4294967280U) << 7) ^ b;
    b  = ((state->z4 << 3) ^ state->z4) >> 12;
    state->z4 = ((state->z4 & 4294967168U) << 13) ^ b;
    return (state->z1 ^ state->z2 ^ state->z3 ^ state->z4);
}

static inline float frandom(varying RNGState * uniform state)
{
    unsigned int irand = random(state);
    irand &= (1<<23)-1;
    return floatbits(0x3F800000 | irand)-1.0f;
}

static inline uniform float frandom(uniform RNGState * uniform state)
{
    uniform unsigned int irand = random(state);
    irand &= (1<<23)-1;
    return floatbits(0x3F800000 | irand)-1.0f;
}

static inline uniform unsigned int __seed4(varying RNGState * uniform state, 
                                           uniform int start,
                                           uniform unsigned int seed) {
    uniform unsigned int c1 = 0xf0f0f0f0;
    uniform unsigned int c2 = 0x0f0f0f0f;

    state->z1 = insert(state->z1, start + 0, seed);
    state->z1 = insert(state->z1, start + 1, seed ^ c1);
    state->z1 = insert(state->z1, start + 2, (seed << 3) ^ c1);
    state->z1 = insert(state->z1, start + 3, (seed << 2) ^ c2);

    seed += 131;
    state->z2 = insert(state->z2, start + 0, seed);
    state->z2 = insert(state->z2, start + 1, seed ^ c1);
    state->z2 = insert(state->z2, start + 2, (seed << 3) ^ c1);
    state->z2 = insert(state->z2, start + 3, (seed << 2) ^ c2);

    seed ^= extract(state->z2, 2);
    state->z3 = insert(state->z3, start + 0, seed);
    state->z3 = insert(state->z3, start + 1, seed ^ c1);
    state->z3 = insert(state->z3, start + 2, (seed << 3) ^ c1);
    state->z3 = insert(state->z3, start + 3, (seed << 2) ^ c2);

    seed <<= 4;
    seed += 3;
    seed ^= extract(state->z1, 3);
    state->z4 = insert(state->z4, start + 0, seed);
    state->z4 = insert(state->z4, start + 1, seed ^ c1);
    state->z4 = insert(state->z4, start + 2, (seed << 3) ^ c1);
    state->z4 = insert(state->z4, start + 3, (seed << 2) ^ c2);

    return seed;
}

static inline void seed_rng(varying RNGState * uniform state, 
                            uniform unsigned int seed) {
    if (programCount == 1) {
        state->z1 = seed;
        state->z2 = seed ^ 0xbeeff00d;
        state->z3 = ((seed & 0xffff) << 16) | (seed >> 16);
        state->z4 = (((seed & 0xff) << 24) | ((seed & 0xff00)  << 8) |
                     ((seed & 0xff0000) >> 8) | (seed & 0xff000000) >> 24);
    }
    else {
        seed = __seed4(state, 0, seed);
        if (programCount == 8)
            __seed4(state, 4, seed ^ 0xbeeff00d);
        if (programCount == 16) {
            __seed4(state, 4,  seed ^ 0xbeeff00d);
            __seed4(state, 8,  ((seed & 0xffff) << 16) | (seed >> 16));
            __seed4(state, 12, (((seed & 0xff) << 24) | ((seed & 0xff00)  << 8) |
                                ((seed & 0xff0000) >> 8) | (seed & 0xff000000) >> 24));
        }
    }
}

static inline void seed_rng(uniform RNGState * uniform state, 
                            uniform unsigned int seed) {
    state->z1 = seed;
    state->z2 = seed ^ 0xbeeff00d;
    state->z3 = ((seed & 0xffff) << 16) | (seed >> 16);
    state->z4 = (((seed & 0xff) << 24) | ((seed & 0xff00)  << 8) |
                 ((seed & 0xff0000) >> 8) | (seed & 0xff000000) >> 24);
}


static inline void fastmath() {
    __fastmath();
}
