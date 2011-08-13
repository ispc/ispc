// -*- mode: c++ -*-
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

/** @file stdlib.ispc

    @brief Portion of the ispc standard library implementation that's in
           ispc code 
*/

///////////////////////////////////////////////////////////////////////////
// Low level primitives

static inline float floatbits(unsigned int a) {
    return __floatbits_varying_int32(a);
}

static inline uniform float floatbits(uniform unsigned int a) {
    return __floatbits_uniform_int32(a);
}

static inline float floatbits(int a) {
    return __floatbits_varying_int32(a);
}

static inline uniform float floatbits(uniform int a) {
    return __floatbits_uniform_int32(a);
}

static inline double doublebits(unsigned int64 a) {
    return __doublebits_varying_int64(a);
}

static inline uniform double doublebits(uniform unsigned int64 a) {
    return __doublebits_uniform_int64(a);
}

static inline unsigned int intbits(float a) {
    return __intbits_varying_float(a);
}

static inline uniform unsigned int intbits(uniform float a) {
    return __intbits_uniform_float(a);
}

static inline unsigned int64 intbits(double d) {
    return __intbits_varying_double(d);
}

static inline uniform unsigned int64 intbits(uniform double d) {
    return __intbits_uniform_double(d);
}

static inline float broadcast(float v, uniform int i) {
    return __broadcast_float(v, i);
}

static inline int8 broadcast(int8 v, uniform int i) {
    return __broadcast_int8(v, i);
}

static inline int16 broadcast(int16 v, uniform int i) {
    return __broadcast_int16(v, i);
}

static inline int32 broadcast(int32 v, uniform int i) {
    return __broadcast_int32(v, i);
}

static inline double broadcast(double v, uniform int i) {
    return __broadcast_double(v, i);
}

static inline int64 broadcast(int64 v, uniform int i) {
    return __broadcast_int64(v, i);
}

static inline float rotate(float v, uniform int i) {
    return __rotate_float(v, i);
}

static inline int8 rotate(int8 v, uniform int i) {
    return __rotate_int8(v, i);
}

static inline int16 rotate(int16 v, uniform int i) {
    return __rotate_int16(v, i);
}

static inline int32 rotate(int32 v, uniform int i) {
    return __rotate_int32(v, i);
}

static inline double rotate(double v, uniform int i) {
    return __rotate_double(v, i);
}

static inline int64 rotate(int64 v, uniform int i) {
    return __rotate_int64(v, i);
}

static inline float shuffle(float v, int i) {
    return __shuffle_float(v, i);
}

static inline int8 shuffle(int8 v, int i) {
    return __shuffle_int8(v, i);
}

static inline int16 shuffle(int16 v, int i) {
    return __shuffle_int16(v, i);
}

static inline int32 shuffle(int32 v, int i) {
    return __shuffle_int32(v, i);
}

static inline double shuffle(double v, int i) {
    return __shuffle_double(v, i);
}

static inline int64 shuffle(int64 v, int i) {
    return __shuffle_int64(v, i);
}

static inline float shuffle(float v0, float v1, int i) {
    return __shuffle2_float(v0, v1, i);
}

static inline int8 shuffle(int8 v0, int8 v1, int i) {
    return __shuffle2_int8(v0, v1, i);
}

static inline int16 shuffle(int16 v0, int16 v1, int i) {
    return __shuffle2_int16(v0, v1, i);
}

static inline int32 shuffle(int32 v0, int32 v1, int i) {
    return __shuffle2_int32(v0, v1, i);
}

static inline double shuffle(double v0, double v1, int i) {
    return __shuffle2_double(v0, v1, i);
}

static inline int64 shuffle(int64 v0, int64 v1, int i) {
    return __shuffle2_int64(v0, v1, i);
}

// x[i]
static inline uniform float extract(float x, uniform int i) {
    return floatbits(__extract_int32((int)intbits(x), i));
}

static inline uniform int8 extract(int8 x, uniform int i) {
    return __extract_int8(x, i);
}

static inline uniform unsigned int8 extract(unsigned int8 x, uniform int i) {
    return __extract_int8(x, (unsigned int)i);
}

static inline uniform int16 extract(int16 x, uniform int i) {
    return __extract_int16(x, i);
}

static inline uniform unsigned int16 extract(unsigned int16 x, uniform int i) {
    return __extract_int16(x, (unsigned int)i);
}

static inline uniform int32 extract(int32 x, uniform int i) {
    return __extract_int32(x, i);
}

static inline uniform unsigned int32 extract(unsigned int32 x, uniform int i) {
    return __extract_int32(x, (unsigned int)i);
}

static inline uniform double extract(double x, uniform int i) {
    return doublebits(__extract_int64((int64)intbits(x), i));
}

static inline uniform int64 extract(int64 x, uniform int i) {
    return __extract_int64(x, i);
}

static inline uniform unsigned int64 extract(unsigned int64 x, uniform int i) {
    return __extract_int64(x, (unsigned int)i);
}

// x[i] = v
static inline float insert(float x, uniform int i, uniform float v) {
    return floatbits(__insert_int32((int)intbits(x), i, (int)intbits(v)));
}

static inline int8 insert(int8 x, uniform int i, uniform int8 v) {
    return __insert_int8(x, i, v);
}

static inline unsigned int8 insert(unsigned int8 x, uniform int i, 
                                    uniform unsigned int8 v) {
    return __insert_int8(x, (unsigned int)i, v);
}

static inline int16 insert(int16 x, uniform int i, uniform int16 v) {
    return __insert_int16(x, i, v);
}

static inline unsigned int16 insert(unsigned int16 x, uniform int i, 
                                    uniform unsigned int16 v) {
    return __insert_int16(x, (unsigned int)i, v);
}

static inline int32 insert(int32 x, uniform int i, uniform int32 v) {
    return __insert_int32(x, i, v);
}

static inline unsigned int32 insert(unsigned int32 x, uniform int i, 
                                    uniform unsigned int32 v) {
    return __insert_int32(x, (unsigned int)i, v);
}

static inline double insert(double x, uniform int i, uniform double v) {
    return doublebits(__insert_int64((int64)intbits(x), i, (int64)intbits(v)));
}

static inline int64 insert(int64 x, uniform int i, uniform int64 v) {
    return __insert_int64(x, i, v);
}

static inline unsigned int64 insert(unsigned int64 x, uniform int i, 
                                    uniform unsigned int64 v) {
    return __insert_int64(x, (unsigned int)i, v);
}

static inline uniform int32 sign_extend(uniform bool v) {
    return __sext_uniform_bool(v);
}

static inline int32 sign_extend(bool v) {
    return __sext_varying_bool(v);
}

static inline uniform bool any(bool v) {
    // We only care about whether "any" is true for the active program instances,
    // so we have to make v with the current program mask.
    return __movmsk(__sext_varying_bool(v) & __mask) != 0;
}

static inline uniform bool all(bool v) {
    // As with any(), we need to explicitly mask v with the current program mask
    // so we're only looking at the current lanes
    int32 match = __sext_varying_bool((__sext_varying_bool(v) & __mask) == __mask);
    return __movmsk(match) == (1 << programCount) - 1;
}

static inline uniform int32 popcnt(uniform int32 v) {
    return __popcnt_int32(v);
}

static inline uniform int popcnt(uniform int64 v) {
    return (int32)__popcnt_int64(v);
}

static inline int popcnt(int v) {
    int r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, popcnt(extract(v, i)));
    return (r & __mask);
}

static inline int popcnt(int64 v) {
    int r;
    for (uniform int i = 0; i < programCount; ++i)
        r = insert(r, i, popcnt(extract(v, i)));
    return (r & __mask);
}

static inline uniform int popcnt(bool v) {
    // As with any() and all(), only count across the active lanes
    return __popcnt_int32(__movmsk(__sext_varying_bool(v) & __mask));
}

static inline uniform int lanemask() {
    return __movmsk(__mask);
}

///////////////////////////////////////////////////////////////////////////
// Prefetching

#define PREFETCHES(NAME, TYPE)                                  \
static inline void prefetch_l1(const reference TYPE ptr) {      \
    __prefetch_read_1_##NAME##_refsconst(ptr);                  \
}                                                               \
static inline void prefetch_l2(const reference TYPE ptr) {      \
    __prefetch_read_2_##NAME##_refsconst(ptr);                  \
}                                                               \
static inline void prefetch_l3(const reference TYPE ptr) {      \
    __prefetch_read_3_##NAME##_refsconst(ptr);                  \
}                                                               \
 static inline void prefetch_nt(const reference TYPE ptr) {     \
     __prefetch_read_nt_##NAME##_refsconst(ptr);                \
}

PREFETCHES(uniform_int8, uniform int8)
PREFETCHES(uniform_int16, uniform int16)
PREFETCHES(uniform_int32, uniform int32)
PREFETCHES(uniform_int64, uniform int64)
PREFETCHES(uniform_float, uniform float)
PREFETCHES(uniform_double, uniform double)

PREFETCHES(varying_int8, int8)
PREFETCHES(varying_int16, int16)
PREFETCHES(varying_int32, int32)
PREFETCHES(varying_int64, int64)
PREFETCHES(varying_float, float)
PREFETCHES(varying_double, double)

#undef PREFETCHES

///////////////////////////////////////////////////////////////////////////
// Horizontal ops / reductions

static inline uniform float reduce_add(float x) {
    // zero the lanes where the mask is off
    return __reduce_add_float(__mask ? x : 0.);
}


static inline uniform float reduce_min(float v) {
    // For the lanes where the mask is off, replace the given value with
    // infinity, so that it doesn't affect the result.
    int iflt_max = 0x7f800000; // infinity
    // Must use __floatbits_varying_int32, not floatbits(), since with the
    // latter the current mask enters into the returned result...
    return __reduce_min_float(__mask ? v : __floatbits_varying_int32(iflt_max));
}

static inline uniform float reduce_max(float v) {
    // For the lanes where the mask is off, replace the given value with
    // negative infinity, so that it doesn't affect the result.
    const uniform int iflt_neg_max = 0xff800000; // -infinity
    // Must use __floatbits_varying_int32, not floatbits(), since with the
    // latter the current mask enters into the returned result...
    return __reduce_max_float(__mask ? v : __floatbits_varying_int32(iflt_neg_max));
}

static inline uniform int reduce_add(int x) {
    // Zero out the values for lanes that aren't running
    return __reduce_add_int32(x & __mask);
}

static inline uniform int reduce_min(int v) {
    // Set values for non-running lanes to the maximum integer value so
    // they don't affect the result.
    int int_max = 0x7fffffff;
    return __reduce_min_int32(__mask ? v : int_max);
}

static inline uniform int reduce_max(int v) {
    // Set values for non-running lanes to the minimum integer value so
    // they don't affect the result.
    int int_min = 0x80000000;
    return __reduce_max_int32(__mask ? v : int_min);
}

static inline uniform unsigned int reduce_add(unsigned int x) {
    // Set values for non-running lanes to zero so they don't affect the
    // result.
    return __reduce_add_uint32(x & __mask);
}

static inline uniform unsigned int reduce_min(unsigned int v) {
    // Set values for non-running lanes to the maximum unsigned integer
    // value so they don't affect the result.
    unsigned int uint_max = 0xffffffff;
    return __reduce_min_uint32(__mask ? v : uint_max);
}

static inline uniform unsigned int reduce_max(unsigned int v) {
    // Set values for non-running lanes to zero so they don't affect the
    // result.
    return __reduce_max_uint32(__mask ? v : 0);
}


static inline uniform double reduce_add(double x) {
    // zero the lanes where the mask is off
    return __reduce_add_double(__mask ? x : 0.);
}

static inline uniform double reduce_min(double v) {
    int64 iflt_max = 0x7ff0000000000000; // infinity
    // Must use __doublebits_varying_int64, not doublebits(), since with the
    // latter the current mask enters into the returned result...
    return __reduce_min_double(__mask ? v : __doublebits_varying_int64(iflt_max));
}

static inline uniform double reduce_max(double v) {
    const uniform int64 iflt_neg_max = 0xfff0000000000000; // -infinity
    // Must use __doublebits_varying_int64, not doublebits(), since with the
    // latter the current mask enters into the returned result...
    return __reduce_max_double(__mask ? v : __doublebits_varying_int64(iflt_neg_max));
}

static inline uniform int64 reduce_add(int64 x) {
    // Zero out the values for lanes that aren't running
    return __reduce_add_int64(x & (int64)(__mask));
}

static inline uniform int64 reduce_min(int64 v) {
    // Set values for non-running lanes to the maximum integer value so
    // they don't affect the result.
    int64 int_max = 0x7fffffffffffffff;
    return __reduce_min_int64(__mask ? v : int_max);
}

static inline uniform int64 reduce_max(int64 v) {
    // Set values for non-running lanes to the minimum integer value so
    // they don't affect the result.
    int64 int_min = 0x8000000000000000;
    return __reduce_max_int64(__mask ? v : int_min);
}

static inline uniform unsigned int64 reduce_add(unsigned int64 x) {
    // Set values for non-running lanes to zero so they don't affect the
    // result.
    return __reduce_add_int64(x & (int64)(__mask));
}

static inline uniform unsigned int64 reduce_min(unsigned int64 v) {
    // Set values for non-running lanes to the maximum unsigned integer
    // value so they don't affect the result.
    unsigned int64 uint_max = 0xffffffffffffffff;
    return __reduce_min_uint64(__mask ? v : uint_max);
}

static inline uniform unsigned int64 reduce_max(unsigned int64 v) {
    // Set values for non-running lanes to zero so they don't affect the
    // result.
    return __reduce_max_uint64(__mask ? v : 0);
}

#define REDUCE_EQUAL(TYPE, FUNCTYPE)                               \
static inline uniform bool reduce_equal(TYPE v) {                  \
    uniform TYPE unusedValue;                                      \
    return __reduce_equal_##FUNCTYPE(v, unusedValue, (int32)__mask); \
}                                                                  \
static inline uniform bool reduce_equal(TYPE v, reference uniform TYPE value) { \
    return __reduce_equal_##FUNCTYPE(v, value, (int32)__mask);       \
}

REDUCE_EQUAL(int32, int32)
REDUCE_EQUAL(unsigned int32, int32)
REDUCE_EQUAL(float, float)
REDUCE_EQUAL(int64, int64)
REDUCE_EQUAL(unsigned int64, int64)
REDUCE_EQUAL(double, double)

static int32 exclusive_scan_add(int32 v) {
    return __exclusive_scan_add_i32(v, (int32)__mask);
}

static unsigned int32 exclusive_scan_add(unsigned int32 v) {
    return __exclusive_scan_add_i32(v, __mask);
}

static float exclusive_scan_add(float v) {
    return __exclusive_scan_add_float(v, __mask);
}

static int64 exclusive_scan_add(int64 v) {
    return __exclusive_scan_add_i64(v, (int32)__mask);
}

static unsigned int64 exclusive_scan_add(unsigned int64 v) {
    return __exclusive_scan_add_i64(v, __mask);
}

static double exclusive_scan_add(double v) {
    return __exclusive_scan_add_double(v, __mask);
}

static int32 exclusive_scan_and(int32 v) {
    return __exclusive_scan_and_i32(v, (int32)__mask);
}

static unsigned int32 exclusive_scan_and(unsigned int32 v) {
    return __exclusive_scan_and_i32(v, __mask);
}

static int64 exclusive_scan_and(int64 v) {
    return __exclusive_scan_and_i64(v, (int32)__mask);
}

static unsigned int64 exclusive_scan_and(unsigned int64 v) {
    return __exclusive_scan_and_i64(v, __mask);
}

static int32 exclusive_scan_or(int32 v) {
    return __exclusive_scan_or_i32(v, (int32)__mask);
}

static unsigned int32 exclusive_scan_or(unsigned int32 v) {
    return __exclusive_scan_or_i32(v, __mask);
}

static int64 exclusive_scan_or(int64 v) {
    return __exclusive_scan_or_i64(v, (int32)__mask);
}

static unsigned int64 exclusive_scan_or(unsigned int64 v) {
    return __exclusive_scan_or_i64(v, __mask);
}

///////////////////////////////////////////////////////////////////////////
// packed load, store

static inline uniform int 
packed_load_active(uniform unsigned int a[], uniform int start,
                   reference unsigned int vals) {
    return __packed_load_active(a, start, vals, __mask);
}

static inline uniform int
packed_store_active(uniform unsigned int a[], uniform int start,
                    unsigned int vals) {
    return __packed_store_active(a, start, vals, __mask);
}

static inline uniform int packed_load_active(uniform int a[], uniform int start,
                                             reference int vals) {
    return __packed_load_active(a, start, vals, __mask);
}

static inline uniform int packed_store_active(uniform int a[], uniform int start,
                                              int vals) {
    return __packed_store_active(a, start, vals, __mask);
}

///////////////////////////////////////////////////////////////////////////
// Atomics and memory barriers

static inline void memory_barrier() {
    __memory_barrier();
}

#define DEFINE_ATOMIC_OP(TA,TB,OPA,OPB)                                 \
static inline TA atomic_##OPA##_global(uniform reference TA ref, TA value) { \
    memory_barrier();                                                   \
    TA ret = __atomic_##OPB##_##TB##_global(ref, value, __mask);  \
    memory_barrier();                                                   \
    return ret;                                                         \
}

DEFINE_ATOMIC_OP(int32,int32,add,add)
DEFINE_ATOMIC_OP(int32,int32,subtract,sub)
DEFINE_ATOMIC_OP(int32,int32,min,min)
DEFINE_ATOMIC_OP(int32,int32,max,max)
DEFINE_ATOMIC_OP(int32,int32,and,and)
DEFINE_ATOMIC_OP(int32,int32,or,or)
DEFINE_ATOMIC_OP(int32,int32,xor,xor)
DEFINE_ATOMIC_OP(int32,int32,swap,swap)

// For everything but atomic min and max, we can use the same
// implementations for unsigned as for signed.
DEFINE_ATOMIC_OP(unsigned int32,int32,add,add)
DEFINE_ATOMIC_OP(unsigned int32,int32,subtract,sub)
DEFINE_ATOMIC_OP(unsigned int32,uint32,min,umin)
DEFINE_ATOMIC_OP(unsigned int32,uint32,max,umax)
DEFINE_ATOMIC_OP(unsigned int32,int32,and,and)
DEFINE_ATOMIC_OP(unsigned int32,int32,or,or)
DEFINE_ATOMIC_OP(unsigned int32,int32,xor,xor)
DEFINE_ATOMIC_OP(unsigned int32,int32,swap,swap)

DEFINE_ATOMIC_OP(float,float,swap,swap)

DEFINE_ATOMIC_OP(int64,int64,add,add)
DEFINE_ATOMIC_OP(int64,int64,subtract,sub)
DEFINE_ATOMIC_OP(int64,int64,min,min)
DEFINE_ATOMIC_OP(int64,int64,max,max)
DEFINE_ATOMIC_OP(int64,int64,and,and)
DEFINE_ATOMIC_OP(int64,int64,or,or)
DEFINE_ATOMIC_OP(int64,int64,xor,xor)
DEFINE_ATOMIC_OP(int64,int64,swap,swap)

// For everything but atomic min and max, we can use the same
// implementations for unsigned as for signed.
DEFINE_ATOMIC_OP(unsigned int64,int64,add,add)
DEFINE_ATOMIC_OP(unsigned int64,int64,subtract,sub)
DEFINE_ATOMIC_OP(unsigned int64,uint64,min,umin)
DEFINE_ATOMIC_OP(unsigned int64,uint64,max,umax)
DEFINE_ATOMIC_OP(unsigned int64,int64,and,and)
DEFINE_ATOMIC_OP(unsigned int64,int64,or,or)
DEFINE_ATOMIC_OP(unsigned int64,int64,xor,xor)
DEFINE_ATOMIC_OP(unsigned int64,int64,swap,swap)

DEFINE_ATOMIC_OP(double,double,swap,swap)

#undef DEFINE_ATOMIC_OP

#define ATOMIC_DECL_CMPXCHG(TA, TB)                                        \
static inline TA atomic_compare_exchange_global(                           \
         uniform reference TA ref, TA oldval, TA newval) {                 \
    memory_barrier();                                                      \
    TA ret = __atomic_compare_exchange_##TB##_global(ref, oldval, newval, __mask); \
    memory_barrier();                                                      \
    return ret;                                                            \
}

ATOMIC_DECL_CMPXCHG(int32, int32)
ATOMIC_DECL_CMPXCHG(unsigned int32, int32)
ATOMIC_DECL_CMPXCHG(float, float)
ATOMIC_DECL_CMPXCHG(int64, int64)
ATOMIC_DECL_CMPXCHG(unsigned int64, int64)
ATOMIC_DECL_CMPXCHG(double, double)

#undef ATOMIC_DECL_CMPXCHG

///////////////////////////////////////////////////////////////////////////
// Floating-Point Math

static inline float abs(float a) {
    // Floating-point hack: zeroing the high bit clears the sign
    unsigned int i = intbits(a);
    i &= 0x7fffffff;
    return floatbits(i);
}

static inline uniform float abs(uniform float a) {
    uniform unsigned int i = intbits(a);
    i &= 0x7fffffff;
    return floatbits(i);
}

static inline double abs(double a) {
    // zeroing the high bit clears the sign
    unsigned int64 i = intbits(a);
    i &= 0x7fffffffffffffff;
    return doublebits(i);
}

static inline uniform double abs(uniform double a) {
    uniform unsigned int64 i = intbits(a);
    i &= 0x7fffffffffffffff;
    return doublebits(i);
}

static inline unsigned int signbits(float x) {
    unsigned int i = intbits(x);
    return (i & 0x80000000);
}

static inline uniform unsigned int signbits(uniform float x) {
    uniform unsigned int i = intbits(x);
    return (i & 0x80000000);
}

static inline unsigned int64 signbits(double x) {
    unsigned int64 i = intbits(x);
    return (i & 0x8000000000000000);
}

static inline uniform unsigned int64 signbits(uniform double x) {
    uniform unsigned int64 i = intbits(x);
    return (i & 0x8000000000000000);
}

static inline float round(float x) {
    return __round_varying_float(x);
}

static inline uniform float round(uniform float x) {
    return __round_uniform_float(x);
}

static inline double round(double x) {
    return __round_varying_double(x);
}

static inline uniform double round(uniform double x) {
    return __round_uniform_double(x);
}

static inline float floor(float x) {
    return __floor_varying_float(x);
}

static inline uniform float floor(uniform float x) {
    return __floor_uniform_float(x);
}

static inline double floor(double x) {
    return __floor_varying_double(x);
}

static inline uniform double floor(uniform double x) {
    return __floor_uniform_double(x);
}

static inline float ceil(float x) {
    return __ceil_varying_float(x);
}

static inline uniform float ceil(uniform float x) {
    return __ceil_uniform_float(x);
}

static inline double ceil(double x) {
    return __ceil_varying_double(x);
}

static inline uniform double ceil(uniform double x) {
    return __ceil_uniform_double(x);
}

static inline float rcp(float v) {
    return __rcp_varying_float(v);
}

static inline uniform float rcp(uniform float v) {
    return __rcp_uniform_float(v);
}

///////////////////////////////////////////////////////////////////////////
// min/max

// float

static inline float min(float a, float b) {
    return __min_varying_float(a, b);
}

static inline uniform float min(uniform float a, uniform float b) {
    return __min_uniform_float(a, b);
}

static inline float max(float a, float b) {
    return __max_varying_float(a, b);
}

static inline uniform float max(uniform float a, uniform float b) {
    return __max_uniform_float(a, b);
}


// double

static inline double min(double a, double b) {
    return __min_varying_double(a, b);
}

static inline uniform double min(uniform double a, uniform double b) {
    return __min_uniform_double(a, b);
}

static inline double max(double a, double b) {
    return __max_varying_double(a, b);
}

static inline uniform double max(uniform double a, uniform double b) {
    return __max_uniform_double(a, b);
}

// int8

static inline uniform unsigned int8 min(uniform unsigned int8 a,
                                        uniform unsigned int8 b) {
    return (a < b) ? a : b;
}

static inline uniform unsigned int8 max(uniform unsigned int8 a, 
                                        uniform unsigned int8 b) {
    return (a > b) ? a : b;
}

static inline uniform int8 min(uniform int8 a, uniform int8 b) {
    return (a < b) ? a : b;
}

static inline uniform int8 max(uniform int8 a, uniform int8 b) {
    return (a > b) ? a : b;
}

static inline unsigned int8 min(unsigned int8 a, unsigned int8 b) {
    return (a < b) ? a : b;
}

static inline unsigned int8 max(unsigned int8 a, unsigned int8 b) {
    return (a > b) ? a : b;
}

static inline int8 min(int8 a, int8 b) {
    return (a < b) ? a : b;
}

static inline int8 max(int8 a, int8 b) {
    return (a > b) ? a : b;
}

// int16

static inline uniform unsigned int16 min(uniform unsigned int16 a, 
                                         uniform unsigned int16 b) {
    return (a < b) ? a : b;
}

static inline uniform unsigned int16 max(uniform unsigned int16 a, 
                                         uniform unsigned int16 b) {
    return (a > b) ? a : b;
}

static inline uniform int16 min(uniform int16 a, uniform int16 b) {
    return (a < b) ? a : b;
}

static inline uniform int16 max(uniform int16 a, uniform int16 b) {
    return (a > b) ? a : b;
}

static inline unsigned int16 min(unsigned int16 a, unsigned int16 b) {
    return (a < b) ? a : b;
}

static inline unsigned int16 max(unsigned int16 a, unsigned int16 b) {
    return (a > b) ? a : b;
}

static inline int16 min(int16 a, int16 b) {
    return (a < b) ? a : b;
}

static inline int16 max(int16 a, int16 b) {
    return (a > b) ? a : b;
}

// int32

static inline unsigned int min(unsigned int a, unsigned int b) {
    return __min_varying_uint32(a, b);
}

static inline uniform unsigned int min(uniform unsigned int a, uniform unsigned int b) {
    return __min_uniform_uint32(a, b);
}

static inline unsigned int max(unsigned int a, unsigned int b) {
    return __max_varying_uint32(a, b);
}

static inline uniform unsigned int max(uniform unsigned int a, uniform unsigned int b) {
    return __max_uniform_uint32(a, b);
}

static inline int min(int a, int b) {
    return __min_varying_int32(a, b);
}

static inline uniform int min(uniform int a, uniform int b) {
    return __min_uniform_int32(a, b);
}

static inline int max(int a, int b) {
    return __max_varying_int32(a, b);
}

static inline uniform int max(uniform int a, uniform int b) {
    return __max_uniform_int32(a, b);
}

// int64

static inline unsigned int64 min(unsigned int64 a, unsigned int64 b) {
    return __min_varying_uint64(a, b);
}

static inline uniform unsigned int64 min(uniform unsigned int64 a, uniform unsigned int64 b) {
    return __min_uniform_uint64(a, b);
}

static inline unsigned int64 max(unsigned int64 a, unsigned int64 b) {
    return __max_varying_uint64(a, b);
}

static inline uniform unsigned int64 max(uniform unsigned int64 a, uniform unsigned int64 b) {
    return __max_uniform_uint64(a, b);
}

static inline int64 min(int64 a, int64 b) {
    return __min_varying_int64(a, b);
}

static inline uniform int64 min(uniform int64 a, uniform int64 b) {
    return __min_uniform_int64(a, b);
}

static inline int64 max(int64 a, int64 b) {
    return __max_varying_int64(a, b);
}

static inline uniform int64 max(uniform int64 a, uniform int64 b) {
    return __max_uniform_int64(a, b);
}

///////////////////////////////////////////////////////////////////////////
// clamps

// float

static inline float clamp(float v, float low, float high) {
    return min(max(v, low), high);
}

static inline uniform float clamp(uniform float v, uniform float low, uniform float high) {
    return min(max(v, low), high);
}

// int8

static inline unsigned int8 clamp(unsigned int8 v, unsigned int8 low, 
                                   unsigned int8 high) {
    return min(max(v, low), high);
}

static inline uniform unsigned int8 clamp(uniform unsigned int8 v, 
                                           uniform unsigned int8 low, 
                                           uniform unsigned int8 high) {
    return min(max(v, low), high);
}

static inline int8 clamp(int8 v, int8 low, int8 high) {
    return min(max(v, low), high);
}

static inline uniform int8 clamp(uniform int8 v, uniform int8 low, 
                                  uniform int8 high) {
    return min(max(v, low), high);
}

// int16

static inline unsigned int16 clamp(unsigned int16 v, unsigned int16 low, 
                                   unsigned int16 high) {
    return min(max(v, low), high);
}

static inline uniform unsigned int16 clamp(uniform unsigned int16 v, 
                                           uniform unsigned int16 low, 
                                           uniform unsigned int16 high) {
    return min(max(v, low), high);
}

static inline int16 clamp(int16 v, int16 low, int16 high) {
    return min(max(v, low), high);
}

static inline uniform int16 clamp(uniform int16 v, uniform int16 low, 
                                  uniform int16 high) {
    return min(max(v, low), high);
}

// int32

static inline unsigned int clamp(unsigned int v, unsigned int low, unsigned int high) {
    return min(max(v, low), high);
}

static inline uniform unsigned int clamp(uniform unsigned int v, uniform unsigned int low, 
                                         uniform unsigned int high) {
    return min(max(v, low), high);
}

static inline int clamp(int v, int low, int high) {
    return min(max(v, low), high);
}

static inline uniform int clamp(uniform int v, uniform int low, uniform int high) {
    return min(max(v, low), high);
}

// int64

static inline unsigned int64 clamp(unsigned int64 v, unsigned int64 low, 
                                   unsigned int64 high) {
    return min(max(v, low), high);
}

static inline uniform unsigned int64 clamp(uniform unsigned int64 v, 
                                           uniform unsigned int64 low, 
                                           uniform unsigned int64 high) {
    return min(max(v, low), high);
}

static inline int64 clamp(int64 v, int64 low, int64 high) {
    return min(max(v, low), high);
}

static inline uniform int64 clamp(uniform int64 v, uniform int64 low, 
                                  uniform int64 high) {
    return min(max(v, low), high);
}

///////////////////////////////////////////////////////////////////////////
// Transcendentals (float precision)

static inline float sqrt(float v) {
    return __sqrt_varying_float(v);
}

static inline uniform float sqrt(uniform float v) {
    return __sqrt_uniform_float(v);
}

static inline float rsqrt(float v) {
    return __rsqrt_varying_float(v);
}

static inline uniform float rsqrt(uniform float v) {
    return __rsqrt_uniform_float(v);
}

static inline float ldexp(float x, int n) {
    unsigned int ex = 0x7F800000u;
    unsigned int ix = intbits(x);
    ex &= ix;              // extract old exponent;
    ix = ix & ~0x7F800000u;  // clear exponent
    n = (n << 23) + ex;
    ix |= n; // insert new exponent
    return floatbits(ix);
}

static inline uniform float ldexp(uniform float x, uniform int n) {
    uniform unsigned int ex = 0x7F800000u;
    uniform unsigned int ix = intbits(x);
    ex &= ix;              // extract old exponent;
    ix = ix & ~0x7F800000u;  // clear exponent
    n = (n << 23) + ex;
    ix |= n; // insert new exponent
    return floatbits(ix);
}

static inline float frexp(float x, reference int pw2) {
    unsigned int ex = 0x7F800000u;              // exponent mask
    unsigned int ix = intbits(x);
    ex &= ix;
    ix &= ~0x7F800000u;  // clear exponent
    pw2 = (int)(ex >> 23) - 126; // compute exponent
    ix |= 0x3F000000u;         // insert exponent +1 in x
    return floatbits(ix);
}

static inline uniform float frexp(uniform float x, reference uniform int pw2) {
    uniform unsigned int ex = 0x7F800000u;              // exponent mask
    uniform unsigned int ix = intbits(x);
    ex &= ix;
    ix &= ~0x7F800000u;  // clear exponent
    pw2 = (uniform int)(ex >> 23) - 126; // compute exponent
    ix |= 0x3F000000u;         // insert exponent +1 in x
    return floatbits(ix);
}

// Most of the transcendental implementations in ispc code here come from
// Solomon Boulos's "syrah": https://github.com/boulos/syrah/

static inline float sin(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_sin(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
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


static inline float cos(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_cos(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
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


static inline void sincos(float x_full, reference float sin_result, reference float cos_result) {
    if (__math_lib == __math_lib_svml) {
        __svml_sincos(x_full, sin_result, cos_result);
    }
    else if (__math_lib == __math_lib_system) {
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform float s, c;
            __stdlib_sincosf(extract(x_full, i), s, c);
            sin_result = insert(sin_result, i, s);
            cos_result = insert(cos_result, i, c);
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

        sin_result = sin_usecos ? cos_formula : sin_formula;
        cos_result = cos_usecos ? cos_formula : sin_formula;

        sin_result = sin_flipsign ? -sin_result : sin_result;
        cos_result = cos_flipsign ? -cos_result : cos_result;
    }
}


static inline void sincos(uniform float x_full, reference uniform float sin_result,
                          reference uniform float cos_result) {
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

        sin_result = sin_usecos ? cos_formula : sin_formula;
        cos_result = cos_usecos ? cos_formula : sin_formula;

        sin_result = sin_flipsign ? -sin_result : sin_result;
        cos_result = cos_flipsign ? -cos_result : cos_result;
    }
}


static inline float tan(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_tan(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
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


static inline float atan(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_atan(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
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


static inline float atan2(float y, float x) {
    if (__math_lib == __math_lib_svml) {
        return __svml_atan2(y, x);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
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


static inline float exp(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_exp(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
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
static inline void __range_reduce_log(float input, reference float reduced, reference int exponent) {
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
    exponent = offset_exponent - 127; // get the real value

    // Blend the offset_exponent with the original input (do this in
    // int for now, until I decide if float can have & and &not)
    int blended = (int_version & nonexponent_mask) | (exponent_neg1);
    reduced = floatbits(blended);
}



static inline void __range_reduce_log(uniform float input, reference uniform float reduced, 
                                      reference uniform int exponent) {
    uniform int int_version = intbits(input);
    static const uniform int nonexponent_mask = 0x807FFFFF;

    static const uniform int exponent_neg1 = (126 << 23);
    uniform int biased_exponent = int_version >> 23;
    uniform int offset_exponent = biased_exponent + 1;
    exponent = offset_exponent - 127; // get the real value

    uniform int blended = (int_version & nonexponent_mask) | (exponent_neg1);
    reduced = floatbits(blended);
}


static inline float log(float x_full) {
    if (__math_lib == __math_lib_svml) {
        return __svml_log(x_full);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform float r = __stdlib_logf(extract(x_full, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
    else if (__math_lib == __math_lib_ispc_fast) {
        int e;
        x_full = frexp(x_full, e);
    
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
        __range_reduce_log(patched, reduced, exponent);

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

static inline uniform float log(uniform float x_full) {
    if (__math_lib == __math_lib_system ||
        __math_lib == __math_lib_svml) {
        return __stdlib_logf(x_full);
    }
    else if (__math_lib == __math_lib_ispc_fast) {
        uniform int e;
        x_full = frexp(x_full, e);
    
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
        __range_reduce_log(patched, reduced, exponent);

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

static inline float pow(float a, float b) {
    if (__math_lib == __math_lib_svml) {
        return __svml_pow(a, b);
    }
    else if (__math_lib == __math_lib_system) {
        float ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
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

static inline double sqrt(double v) {
    return __sqrt_varying_double(v);
}

static inline uniform double sqrt(uniform double v) {
    return __sqrt_uniform_double(v);
}

static inline double ldexp(double x, int n) {
    unsigned int64 ex = 0x7ff0000000000000;
    unsigned int64 ix = intbits(x);
    ex &= ix;
    ix = ix & ~0x7ff0000000000000;  // clear exponent
    int64 n64 = ((int64)n << 52) + ex;
    ix |= n64; // insert new exponent
    return doublebits(ix);
}

static inline uniform double ldexp(uniform double x, uniform int n) {
    uniform unsigned int64 ex = 0x7ff0000000000000;
    uniform unsigned int64 ix = intbits(x);
    ex &= ix;
    ix = ix & ~0x7ff0000000000000;  // clear exponent
    uniform int64 n64 = ((int64)n << 52) + ex;
    ix |= n64; // insert new exponent
    return doublebits(ix);
}

static inline double frexp(double x, reference int pw2) {
    unsigned int64 ex = 0x7ff0000000000000;              // exponent mask
    unsigned int64 ix = intbits(x);
    ex &= ix;
    ix &= ~0x7ff0000000000000;  // clear exponent
    pw2 = (int)(ex >> 52) - 1022; // compute exponent
    ix |= 0x3fe0000000000000;         // insert exponent +1 in x
    return doublebits(ix);
}

static inline uniform double frexp(uniform double x, reference uniform int pw2) {
    uniform unsigned int64 ex = 0x7ff0000000000000;              // exponent mask
    uniform unsigned int64 ix = intbits(x);
    ex &= ix;
    ix &= ~0x7ff0000000000000;  // clear exponent
    pw2 = (int)(ex >> 52) - 1022; // compute exponent
    ix |= 0x3fe0000000000000;         // insert exponent +1 in x
    return doublebits(ix);
}

static inline double sin(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return sin((float)x);
    else {
        double ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform double r = __stdlib_sin(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

static inline uniform double sin(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return sin((float)x);
    else
        return __stdlib_sin(x);
}

static inline double cos(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return cos((float)x);
    else {
        double ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform double r = __stdlib_cos(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

static inline uniform double cos(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return cos((float)x);
    else
        return __stdlib_cos(x);
}

static inline void sincos(double x, reference double sin_result,
                          reference double cos_result) {
    if (__math_lib == __math_lib_ispc_fast) {
        float sr, cr;
        sincos((float)x, sr, cr);
        sin_result = sr;
        cos_result = cr;
    }
    else {
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            uniform double sr, cr;
            if ((mask & (1 << i)) == 0)
                continue;
            __stdlib_sincos(extract(x, i), sr, cr);
            sin_result = insert(sin_result, i, sr);
            cos_result = insert(cos_result, i, cr);
        }
    }
}

static inline void sincos(uniform double x, reference uniform double sin_result,
                          reference uniform double cos_result) {
    if (__math_lib == __math_lib_ispc_fast) {
        uniform float sr, cr;
        sincos((uniform float)x, sr, cr);
        sin_result = sr;
        cos_result = cr;
    }
    else
        __stdlib_sincos(x, sin_result, cos_result);
}

static inline double tan(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return tan((float)x);
    else {
        double ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform double r = __stdlib_tan(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

static inline uniform double tan(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return tan((float)x);
    else
        return __stdlib_tan(x);
}

static inline double atan(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return atan((float)x);
    else {
        double ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform double r = __stdlib_atan(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

static inline uniform double atan(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return atan((float)x);
    else
        return __stdlib_atan(x);
}

static inline double atan2(double y, double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return atan2((float)y, (float)x);
    else {
        double ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform double r = __stdlib_atan2(extract(y, i), extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

static inline uniform double atan2(uniform double y, uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return atan2((float)y, (float)x);
    else
        return __stdlib_atan2(y, x);
}

static inline double exp(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return exp((float)x);
    else {
        double ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform double r = __stdlib_exp(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

static inline uniform double exp(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return exp((float)x);
    else
        return __stdlib_exp(x);
}

static inline double log(double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return log((float)x);
    else {
        double ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform double r = __stdlib_log(extract(x, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

static inline uniform double log(uniform double x) {
    if (__math_lib == __math_lib_ispc_fast)
        return log((float)x);
    else
        return __stdlib_log(x);
}

static inline double pow(double a, double b) {
    if (__math_lib == __math_lib_ispc_fast)
        return pow((float)a, (float)b);
    else {
        double ret;
        uniform int mask = lanemask();
        for (uniform int i = 0; i < programCount; ++i) {
            if ((mask & (1 << i)) == 0)
                continue;
            uniform double r = __stdlib_pow(extract(a, i), extract(b, i));
            ret = insert(ret, i, r);
        }
        return ret;
    }
}

static inline uniform double pow(uniform double a, uniform double b) {
    if (__math_lib == __math_lib_ispc_fast)
        return pow((float)a, (float)b);
    else
        return __stdlib_pow(a, b);
}

///////////////////////////////////////////////////////////////////////////
// half-precision floats

static inline uniform float half_to_float(uniform unsigned int16 h) {
    if ((h & 0x7FFFu) == 0) 
        // Signed zero
        return floatbits(((unsigned int32) h) << 16);
    else {
        // Though these are int16 quantities, we get much better code 
        // with them stored as int32s...
        uniform unsigned int32 hs = h & (int32)0x8000u;  // Pick off sign bit
        uniform unsigned int32 he = h & (int32)0x7C00u;  // Pick off exponent bits
        uniform unsigned int32 hm = h & (int32)0x03FFu;  // Pick off mantissa bits
        if (he == 0) {  
            // Denormal will convert to normalized
            uniform int e = -1;
            // The following loop figures out how much extra to adjust the exponent
            // Shift until leading bit overflows into exponent bit
            do {
                e++;
                hm <<= 1;
            } while((hm & 0x0400u) == 0);

            // Sign bit
            uniform unsigned int32 xs = ((unsigned int32) hs) << 16; 
            // Exponent: unbias the halfp, then bias the single
            uniform int32 xes = ((int32)(he >> 10)) - 15 + 127 - e;
            // Exponent
            uniform unsigned int32 xe = (unsigned int32) (xes << 23); 
            // Mantissa
            uniform unsigned int32 xm = ((unsigned int32) (hm & 0x03FFu)) << 13; 
            return floatbits(xs | xe | xm);
        } 
        else {
            if (he == 0x7C00u) {  
                // Inf or NaN (all the exponent bits are set)
                if (hm == 0)
                    // Zero mantissa -> signed inf
                    return floatbits((((unsigned int32) hs) << 16) | 
                                     ((unsigned int32) 0x7F800000u));
                else
                    // NaN
                    return floatbits(0xFFC00000u);
            }
            else { 
                // Normalized number
                // sign
                uniform unsigned int32 xs = ((unsigned int32) hs) << 16; 
                // Exponent: unbias the halfp, then bias the single
                uniform int32 xes = ((int32) (he >> 10)) - 15 + 127; 
                // Exponent
                uniform unsigned int32 xe = (unsigned int32) (xes << 23);
                // Mantissa
                uniform unsigned int32 xm = ((unsigned int32) hm) << 13; 
                return floatbits(xs | xe | xm);
            }
        }
    }
}

static inline float half_to_float(unsigned int16 h) {
    if ((h & 0x7FFFu) == 0) 
        // Signed zero
        return floatbits(((unsigned int32) h) << 16);
    else {
        // Though these are int16 quantities, we get much better code 
        // with them stored as int32s...
        unsigned int32 hs = h & (int32)0x8000u;  // Pick off sign bit
        unsigned int32 he = h & (int32)0x7C00u;  // Pick off exponent bits
        unsigned int32 hm = h & (int32)0x03FFu;  // Pick off mantissa bits
        cif (he == 0) {  
            // Denormal will convert to normalized
            int e = -1;
            // The following loop figures out how much extra to adjust the exponent
            // Shift until leading bit overflows into exponent bit
            do {
                e++;
                hm <<= 1;
            } while((hm & 0x0400u) == 0);

            // Sign bit
            unsigned int32 xs = ((unsigned int32) hs) << 16; 
            // Exponent: unbias the halfp, then bias the single
            int32 xes = ((int32)(he >> 10)) - 15 + 127 - e;
            // Exponent
            unsigned int32 xe = (unsigned int32) (xes << 23); 
            // Mantissa
            unsigned int32 xm = ((unsigned int32) (hm & 0x03FFu)) << 13; 
            return floatbits(xs | xe | xm);
        } 
        else {
            if (he == 0x7C00u) {  
                // Inf or NaN (all the exponent bits are set)
                if (hm == 0)
                    // Zero mantissa -> signed inf
                    return floatbits((((unsigned int32) hs) << 16) | 
                                     ((unsigned int32) 0x7F800000u));
                else
                    // NaN
                    return floatbits(0xFFC00000u);
            }
            else { 
                // Normalized number
                // sign
                unsigned int32 xs = ((unsigned int32) hs) << 16; 
                // Exponent: unbias the halfp, then bias the single
                int32 xes = ((int32) (he >> 10)) - 15 + 127; 
                // Exponent
                unsigned int32 xe = (unsigned int32) (xes << 23);
                // Mantissa
                unsigned int32 xm = ((unsigned int32) hm) << 13; 
                return floatbits(xs | xe | xm);
            }
        }
    }
}


static inline uniform int16 float_to_half(uniform float f) {
    uniform int32 x = intbits(f);
    // Store the return value in an int32 until the very end; this ends up
    // generating better code...
    uniform int32 ret;
    if ((x & 0x7FFFFFFFu) == 0)
        // Signed zero
        ret = (x >> 16); 
    else {
        uniform unsigned int32 xs = x & 0x80000000u;  // Pick off sign bit
        uniform unsigned int32 xe = x & 0x7F800000u;  // Pick off exponent bits
        uniform unsigned int32 xm = x & 0x007FFFFFu;  // Pick off mantissa bits
        if (xe == 0) {  
            // Denormal will underflow, return a signed zero
            ret = (xs >> 16);
        } 
        else {
            if (xe == 0x7F800000u) {  
                // Inf or NaN (all the exponent bits are set)
                if (xm == 0)
                    // Zero mantissa -> signed infinity
                    ret = ((xs >> 16) | 0x7C00u);
                else
                    // NaN, only 1st mantissa bit set
                    ret = 0xFE00u;
            }
            else { 
                // Normalized number
                uniform unsigned int32 hs = (xs >> 16); // Sign bit
                uniform unsigned int32 hm;
                // Exponent unbias the single, then bias the halfp
                uniform int32 hes = ((int)(xe >> 23)) - 127 + 15; 
                if (hes >= 0x1F)  
                    // Overflow: return signed infinity
                    ret = ((xs >> 16) | 0x7C00u);
                else if (hes <= 0) {
                    // Underflow
                    if ((14 - hes) > 24) {  
                        // Mantissa shifted all the way off & no rounding possibility
                        hm = 0u;  // Set mantissa to zero
                    } 
                    else {
                        xm |= 0x00800000u;  // Add the hidden leading bit
                        hm = (xm >> (14 - hes)); // Mantissa
                        if ((xm >> (13 - hes)) & 0x00000001u) // Check for rounding
                            // Round, might overflow into exp bit, but this is OK
                            hm += 1u; 
                    }
                    ret = (hs | hm);
                } 
                else {
                    uniform unsigned int32 he = (hes << 10); // Exponent
                    hm = (xm >> 13); // Mantissa
                    if (xm & 0x00001000u) // Check for rounding
                        // Round, might overflow to inf, this is OK
                        ret = (hs | he | hm) + 1u; 
                    else
                        ret = (hs | he | hm);
                }
            }
        }
    }
    return (int16)ret;
}


static inline int16 float_to_half(float f) {
    int32 x = intbits(f);
    // Store the return value in an int32 until the very end; this ends up
    // generating better code...
    int32 ret;
    if ((x & 0x7FFFFFFFu) == 0)
        // Signed zero
        ret = (x >> 16); 
    else {
        unsigned int32 xs = x & 0x80000000u;  // Pick off sign bit
        unsigned int32 xe = x & 0x7F800000u;  // Pick off exponent bits
        unsigned int32 xm = x & 0x007FFFFFu;  // Pick off mantissa bits
        if (xe == 0) {  
            // Denormal will underflow, return a signed zero
            ret = (xs >> 16);
        } 
        else {
            cif (xe == 0x7F800000u) {  
                // Inf or NaN (all the exponent bits are set)
                if (xm == 0)
                    // Zero mantissa -> signed infinity
                    ret = ((xs >> 16) | 0x7C00u);
                else
                    // NaN, only 1st mantissa bit set
                    ret = 0xFE00u;
            }
            else { 
                // Normalized number
                unsigned int32 hs = (xs >> 16); // Sign bit
                unsigned int32 hm;
                // Exponent unbias the single, then bias the halfp
                int32 hes = ((int)(xe >> 23)) - 127 + 15; 
                if (hes >= 0x1F)  
                    // Overflow: return signed infinity
                    ret = ((xs >> 16) | 0x7C00u);
                else if (hes <= 0) {
                    // Underflow
                    if ((14 - hes) > 24) {  
                        // Mantissa shifted all the way off & no rounding possibility
                        hm = 0u;  // Set mantissa to zero
                    } 
                    else {
                        xm |= 0x00800000u;  // Add the hidden leading bit
                        hm = (xm >> (14 - hes)); // Mantissa
                        if ((xm >> (13 - hes)) & 0x00000001u) // Check for rounding
                            // Round, might overflow into exp bit, but this is OK
                            hm += 1u; 
                    }
                    ret = (hs | hm);
                } 
                else {
                    unsigned int32 he = (hes << 10); // Exponent
                    hm = (xm >> 13); // Mantissa
                    if (xm & 0x00001000u) // Check for rounding
                        // Round, might overflow to inf, this is OK
                        ret = (hs | he | hm) + 1u; 
                    else
                        ret = (hs | he | hm);
                }
            }
        }
    }
    return (int16)ret;
}


static inline uniform float half_to_float_fast(uniform unsigned int16 h) {
    uniform unsigned int32 hs = h & (int32)0x8000u;  // Pick off sign bit
    uniform unsigned int32 he = h & (int32)0x7C00u;  // Pick off exponent bits
    uniform unsigned int32 hm = h & (int32)0x03FFu;  // Pick off mantissa bits

    // sign
    uniform unsigned int32 xs = ((unsigned int32) hs) << 16; 
    // Exponent: unbias the halfp, then bias the single
    uniform int32 xes = ((int32) (he >> 10)) - 15 + 127; 
    // Exponent
    uniform unsigned int32 xe = (unsigned int32) (xes << 23);
    // Mantissa
    uniform unsigned int32 xm = ((unsigned int32) hm) << 13; 
    return floatbits(xs | xe | xm);

}

static inline float half_to_float_fast(unsigned int16 h) {
    unsigned int32 hs = h & (int32)0x8000u;  // Pick off sign bit
    unsigned int32 he = h & (int32)0x7C00u;  // Pick off exponent bits
    unsigned int32 hm = h & (int32)0x03FFu;  // Pick off mantissa bits

    // sign
    unsigned int32 xs = ((unsigned int32) hs) << 16; 
    // Exponent: unbias the halfp, then bias the single
    int32 xes = ((int32) (he >> 10)) - 15 + 127; 
    // Exponent
    unsigned int32 xe = (unsigned int32) (xes << 23);
    // Mantissa
    unsigned int32 xm = ((unsigned int32) hm) << 13; 
    return floatbits(xs | xe | xm);

}

static inline uniform int16 float_to_half_fast(uniform float f) {
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

static inline int16 float_to_half_fast(float f) {
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

///////////////////////////////////////////////////////////////////////////
// RNG stuff

struct RNGState {
    unsigned int z1, z2, z3, z4;
};

static inline unsigned int random(reference RNGState state)
{
    unsigned int b;

    b  = ((state.z1 << 6) ^ state.z1) >> 13;
    state.z1 = ((state.z1 & 4294967294U) << 18) ^ b;
    b  = ((state.z2 << 2) ^ state.z2) >> 27; 
    state.z2 = ((state.z2 & 4294967288U) << 2) ^ b;
    b  = ((state.z3 << 13) ^ state.z3) >> 21;
    state.z3 = ((state.z3 & 4294967280U) << 7) ^ b;
    b  = ((state.z4 << 3) ^ state.z4) >> 12;
    state.z4 = ((state.z4 & 4294967168U) << 13) ^ b;
    return (state.z1 ^ state.z2 ^ state.z3 ^ state.z4);
}

static inline float frandom(reference RNGState state)
{
    unsigned int irand = random(state);
    irand &= (1<<23)-1;
    return floatbits(0x3F800000 | irand)-1.0f;
}

static inline uniform unsigned int __seed4(reference RNGState state, 
                                           uniform int start,
                                           uniform unsigned int seed) {
    uniform unsigned int c1 = 0xf0f0f0f0;
    uniform unsigned int c2 = 0x0f0f0f0f;

    state.z1 = insert(state.z1, start + 0, seed);
    state.z1 = insert(state.z1, start + 1, seed ^ c1);
    state.z1 = insert(state.z1, start + 2, (seed << 3) ^ c1);
    state.z1 = insert(state.z1, start + 3, (seed << 2) ^ c2);

    seed += 131;
    state.z2 = insert(state.z2, start + 0, seed);
    state.z2 = insert(state.z2, start + 1, seed ^ c1);
    state.z2 = insert(state.z2, start + 2, (seed << 3) ^ c1);
    state.z2 = insert(state.z2, start + 3, (seed << 2) ^ c2);

    seed ^= extract(state.z2, 2);
    state.z3 = insert(state.z3, start + 0, seed);
    state.z3 = insert(state.z3, start + 1, seed ^ c1);
    state.z3 = insert(state.z3, start + 2, (seed << 3) ^ c1);
    state.z3 = insert(state.z3, start + 3, (seed << 2) ^ c2);

    seed <<= 4;
    seed += 3;
    seed ^= extract(state.z1, 3);
    state.z4 = insert(state.z4, start + 0, seed);
    state.z4 = insert(state.z4, start + 1, seed ^ c1);
    state.z4 = insert(state.z4, start + 2, (seed << 3) ^ c1);
    state.z4 = insert(state.z4, start + 3, (seed << 2) ^ c2);

    return seed;
}

static inline void seed_rng(reference uniform RNGState state, uniform unsigned int seed) {
    seed = __seed4(state, 0, seed);
    if (programCount == 8)
        __seed4(state, 4, seed ^ 0xbeeff00d);
}

static inline void fastmath() {
    __fastmath();
}
