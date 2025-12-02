// Copyright (c) 2025, Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// This file provides the target-independent definitions for builtin functions.

#define ISPC_INTERNAL_STDLIB_COMPILATION
#include "core.isph"
#undef ISPC_INTERNAL_STDLIB_COMPILATION

#define ISPC_INTERNAL_STDLIB_COMPILATION
#include "builtins.isph"
#undef ISPC_INTERNAL_STDLIB_COMPILATION

#if TARGET_WIDTH == 2
#define DOUBLE_WIDTH 4
#elif TARGET_WIDTH == 4
#define DOUBLE_WIDTH 8
#elif TARGET_WIDTH == 8
#define DOUBLE_WIDTH 16
#elif TARGET_WIDTH == 16
#define DOUBLE_WIDTH 32
#elif TARGET_WIDTH == 32
#define DOUBLE_WIDTH 64
#elif TARGET_WIDTH == 64
#define DOUBLE_WIDTH 128
#else
#error Unknown value of TARGET_WIDTH
#endif

// Macros for attributes
#define NOESCAPE __attribute__((noescape))
#define ADDRSPACE(N) __attribute__((address_space(N)))
#define READONLY __attribute__((memory("read")))
#define READNONE __attribute__((memory("none")))
#define UNMANGLED __attribute__((unmangled))
#define CDECL __attribute__((cdecl))
#define EXT __attribute__((unmangled)) __attribute__((cdecl)) unmasked
// all ISPC pointer arguments of functions are noalias so we don't need to
// support it until it changed.
// #define NOALIAS __attribute__((noalias))
#define NOALIAS

// Declared function to trap unsupported functions during compilation (linking) time.
// In sane cases, this function call should always be removed.
extern "C" void __not_supported();

// Overloaded functions that are wrappers around insertelement and extractelement LLVM operations.
unmasked bool __insert(bool v, uniform int32 i, uniform bool val) {
#if (ISPC_MASK_BITS == 1)
    return @llvm.ispc.insert(v, i, val);
#elif (ISPC_MASK_BITS == 64)
    return @llvm.ispc.insert(v, i, val ? -1ll : 0ll);
#else
    // For mask != 1 bit, varying bool and uniform bool have different representations (<N x i32> and i1).
    return @llvm.ispc.insert(v, i, val ? -1 : 0);
#endif
}
unmasked int8 __insert(int8 v, uniform int32 i, uniform int8 val) { return @llvm.ispc.insert(v, i, val); }
unmasked int16 __insert(int16 v, uniform int32 i, uniform int16 val) { return @llvm.ispc.insert(v, i, val); }
unmasked int32 __insert(int32 v, uniform int32 i, uniform int32 val) { return @llvm.ispc.insert(v, i, val); }
unmasked int64 __insert(int64 v, uniform int32 i, uniform int64 val) { return @llvm.ispc.insert(v, i, val); }
unmasked uint8 __insert(uint8 v, uniform int32 i, uniform uint8 val) { return @llvm.ispc.insert(v, i, val); }
unmasked uint16 __insert(uint16 v, uniform int32 i, uniform uint16 val) { return @llvm.ispc.insert(v, i, val); }
unmasked uint32 __insert(uint32 v, uniform int32 i, uniform uint32 val) { return @llvm.ispc.insert(v, i, val); }
unmasked uint64 __insert(uint64 v, uniform int32 i, uniform uint64 val) { return @llvm.ispc.insert(v, i, val); }
unmasked float16 __insert(float16 v, uniform int32 i, uniform float16 val) { return @llvm.ispc.insert(v, i, val); }
unmasked float __insert(float v, uniform int32 i, uniform float val) { return @llvm.ispc.insert(v, i, val); }
unmasked double __insert(double v, uniform int32 i, uniform double val) { return @llvm.ispc.insert(v, i, val); }

unmasked uniform bool __extract(bool v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform int8 __extract(int8 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform int16 __extract(int16 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform int32 __extract(int32 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform int64 __extract(int64 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform uint8 __extract(uint8 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform uint16 __extract(uint16 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform uint32 __extract(uint32 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform uint64 __extract(uint64 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform float16 __extract(float16 v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform float __extract(float v, uniform int32 i) { return @llvm.ispc.extract(v, i); }
unmasked uniform double __extract(double v, uniform int32 i) { return @llvm.ispc.extract(v, i); }

// overloaded select functions
unmasked int8 __select(bool mask, int8 v0, int8 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked int16 __select(bool mask, int16 v0, int16 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked int32 __select(bool mask, int32 v0, int32 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked int64 __select(bool mask, int64 v0, int64 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked uint8 __select(bool mask, uint8 v0, uint8 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked uint16 __select(bool mask, uint16 v0, uint16 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked uint32 __select(bool mask, uint32 v0, uint32 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked uint64 __select(bool mask, uint64 v0, uint64 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked float16 __select(bool mask, float16 v0, float16 v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked float __select(bool mask, float v0, float v1) { return @llvm.ispc.select(mask, v0, v1); }
unmasked double __select(bool mask, double v0, double v1) { return @llvm.ispc.select(mask, v0, v1); }

// concat: concatenate varying vectors to short vector of DOUBLE_WIDTH
unmasked uniform int8<DOUBLE_WIDTH> __concat(int8 v0, int8 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform int16<DOUBLE_WIDTH> __concat(int16 v0, int16 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform int32<DOUBLE_WIDTH> __concat(int32 v0, int32 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform int64<DOUBLE_WIDTH> __concat(int64 v0, int64 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform uint8<DOUBLE_WIDTH> __concat(uint8 v0, uint8 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform uint16<DOUBLE_WIDTH> __concat(uint16 v0, uint16 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform uint32<DOUBLE_WIDTH> __concat(uint32 v0, uint32 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform uint64<DOUBLE_WIDTH> __concat(uint64 v0, uint64 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform float16<DOUBLE_WIDTH> __concat(float16 v0, float16 v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform float<DOUBLE_WIDTH> __concat(float v0, float v1) { return @llvm.ispc.concat(v0, v1); }
unmasked uniform double<DOUBLE_WIDTH> __concat(double v0, double v1) { return @llvm.ispc.concat(v0, v1); }

// overloaded __masked_expandload functions
unmasked varying int8 __masked_expandload(uniform int8 *uniform from, UIntMaskType mask, varying int8 passthru) {
    return @llvm.masked.expandload(from, mask, passthru);
}
unmasked varying int16 __masked_expandload(uniform int8 *uniform from, UIntMaskType mask, varying int16 passthru) {
    return @llvm.masked.expandload(from, mask, passthru);
}
unmasked varying int32 __masked_expandload(uniform int8 *uniform from, UIntMaskType mask, varying int32 passthru) {
    return @llvm.masked.expandload(from, mask, passthru);
}
unmasked varying int64 __masked_expandload(uniform int8 *uniform from, UIntMaskType mask, varying int64 passthru) {
    return @llvm.masked.expandload(from, mask, passthru);
}
unmasked varying float16 __masked_expandload(uniform int8 *uniform from, UIntMaskType mask, varying float16 passthru) {
    return @llvm.masked.expandload(from, mask, passthru);
}
unmasked varying float __masked_expandload(uniform int8 *uniform from, UIntMaskType mask, varying float passthru) {
    return @llvm.masked.expandload(from, mask, passthru);
}
unmasked varying double __masked_expandload(uniform int8 *uniform from, UIntMaskType mask, varying double passthru) {
    return @llvm.masked.expandload(from, mask, passthru);
}

// bitcast
template <typename L, typename R> unmasked L __bitcast(R x) {
    // This function should never be called, instead specialized versions should be used.
    __not_supported();
}
// uniform bitcasts
template <> unmasked uniform int8 __bitcast<uniform int8, uniform int8>(uniform int8 x) { return x; }
template <> unmasked uniform uint8 __bitcast<uniform uint8, uniform uint8>(uniform uint8 x) { return x; }
template <> unmasked uniform uint8 __bitcast<uniform uint8, uniform int8>(uniform int8 x) {
    return @llvm.ispc.bitcast(x, (uniform uint8)0);
}
template <> unmasked uniform int8 __bitcast<uniform int8, uniform uint8>(uniform uint8 x) {
    return @llvm.ispc.bitcast(x, (uniform int8)0);
}
template <> unmasked uniform int16 __bitcast<uniform int16, uniform int16>(uniform int16 x) { return x; }
template <> unmasked uniform uint16 __bitcast<uniform uint16, uniform uint16>(uniform uint16 x) { return x; }
template <> unmasked uniform float16 __bitcast<uniform float16, uniform float16>(uniform float16 x) { return x; }
template <> unmasked uniform uint16 __bitcast<uniform uint16, uniform int16>(uniform int16 x) {
    return @llvm.ispc.bitcast(x, (uniform uint16)0);
}
template <> unmasked uniform int16 __bitcast<uniform int16, uniform uint16>(uniform uint16 x) {
    return @llvm.ispc.bitcast(x, (uniform int16)0);
}
template <> unmasked uniform int16 __bitcast<uniform int16, uniform float16>(uniform float16 x) {
    return @llvm.ispc.bitcast(x, (uniform int16)0);
}
template <> unmasked uniform uint16 __bitcast<uniform uint16, uniform float16>(uniform float16 x) {
    return @llvm.ispc.bitcast(x, (uniform uint16)0);
}
template <> unmasked uniform float16 __bitcast<uniform float16, uniform uint16>(uniform uint16 x) {
    return @llvm.ispc.bitcast(x, (uniform float16)0);
}
template <> unmasked uniform float16 __bitcast<uniform float16, uniform int16>(uniform int16 x) {
    return @llvm.ispc.bitcast(x, (uniform float16)0);
}
template <> unmasked uniform int32 __bitcast<uniform int32, uniform int32>(uniform int32 x) { return x; }
template <> unmasked uniform uint32 __bitcast<uniform uint32, uniform uint32>(uniform uint32 x) { return x; }
template <> unmasked uniform float __bitcast<uniform float, uniform float>(uniform float x) { return x; }
template <> unmasked uniform uint32 __bitcast<uniform uint32, uniform int32>(uniform int32 x) {
    return @llvm.ispc.bitcast(x, (uniform uint32)0);
}
template <> unmasked uniform int32 __bitcast<uniform int32, uniform uint32>(uniform uint32 x) {
    return @llvm.ispc.bitcast(x, (uniform int32)0);
}
template <> unmasked uniform int32 __bitcast<uniform int32, uniform float>(uniform float x) {
    return @llvm.ispc.bitcast(x, (uniform int32)0);
}
template <> unmasked uniform uint32 __bitcast<uniform uint32, uniform float>(uniform float x) {
    return @llvm.ispc.bitcast(x, (uniform uint32)0);
}
template <> unmasked uniform float __bitcast<uniform float, uniform uint32>(uniform uint32 x) {
    return @llvm.ispc.bitcast(x, (uniform float)0);
}
template <> unmasked uniform float __bitcast<uniform float, uniform int32>(uniform int32 x) {
    return @llvm.ispc.bitcast(x, (uniform float)0);
}
template <> unmasked uniform int64 __bitcast<uniform int64, uniform int64>(uniform int64 x) { return x; }
template <> unmasked uniform uint64 __bitcast<uniform uint64, uniform uint64>(uniform uint64 x) { return x; }
template <> unmasked uniform double __bitcast<uniform double, uniform double>(uniform double x) { return x; }
template <> unmasked uniform uint64 __bitcast<uniform uint64, uniform int64>(uniform int64 x) {
    return @llvm.ispc.bitcast(x, (uniform uint64)0);
}
template <> unmasked uniform int64 __bitcast<uniform int64, uniform uint64>(uniform uint64 x) {
    return @llvm.ispc.bitcast(x, (uniform int64)0);
}
template <> unmasked uniform int64 __bitcast<uniform int64, uniform double>(uniform double x) {
    return @llvm.ispc.bitcast(x, (uniform int64)0);
}
template <> unmasked uniform uint64 __bitcast<uniform uint64, uniform double>(uniform double x) {
    return @llvm.ispc.bitcast(x, (uniform uint64)0);
}
template <> unmasked uniform double __bitcast<uniform double, uniform uint64>(uniform uint64 x) {
    return @llvm.ispc.bitcast(x, (uniform double)0);
}
template <> unmasked uniform double __bitcast<uniform double, uniform int64>(uniform int64 x) {
    return @llvm.ispc.bitcast(x, (uniform double)0);
}
// varying bitcasts
template <> unmasked varying int8 __bitcast<varying int8, varying int8>(varying int8 x) { return x; }
template <> unmasked varying uint8 __bitcast<varying uint8, varying uint8>(varying uint8 x) { return x; }
template <> unmasked varying uint8 __bitcast<varying uint8, varying int8>(varying int8 x) {
    return @llvm.ispc.bitcast(x, (varying uint8)0);
}
template <> unmasked varying int8 __bitcast<varying int8, varying uint8>(varying uint8 x) {
    return @llvm.ispc.bitcast(x, (varying int8)0);
}
template <> unmasked varying int16 __bitcast<varying int16, varying int16>(varying int16 x) { return x; }
template <> unmasked varying uint16 __bitcast<varying uint16, varying uint16>(varying uint16 x) { return x; }
template <> unmasked varying float16 __bitcast<varying float16, varying float16>(varying float16 x) { return x; }
template <> unmasked varying uint16 __bitcast<varying uint16, varying int16>(varying int16 x) {
    return @llvm.ispc.bitcast(x, (varying uint16)0);
}
template <> unmasked varying int16 __bitcast<varying int16, varying uint16>(varying uint16 x) {
    return @llvm.ispc.bitcast(x, (varying int16)0);
}
template <> unmasked varying int16 __bitcast<varying int16, varying float16>(varying float16 x) {
    return @llvm.ispc.bitcast(x, (varying int16)0);
}
template <> unmasked varying uint16 __bitcast<varying uint16, varying float16>(varying float16 x) {
    return @llvm.ispc.bitcast(x, (varying uint16)0);
}
template <> unmasked varying float16 __bitcast<varying float16, varying uint16>(varying uint16 x) {
    return @llvm.ispc.bitcast(x, (varying float16)0);
}
template <> unmasked varying float16 __bitcast<varying float16, varying int16>(varying int16 x) {
    return @llvm.ispc.bitcast(x, (varying float16)0);
}
template <> unmasked varying int32 __bitcast<varying int32, varying int32>(varying int32 x) { return x; }
template <> unmasked varying uint32 __bitcast<varying uint32, varying uint32>(varying uint32 x) { return x; }
template <> unmasked varying float __bitcast<varying float, varying float>(varying float x) { return x; }
template <> unmasked varying uint32 __bitcast<varying uint32, varying int32>(varying int32 x) {
    return @llvm.ispc.bitcast(x, (varying uint32)0);
}
template <> unmasked varying int32 __bitcast<varying int32, varying uint32>(varying uint32 x) {
    return @llvm.ispc.bitcast(x, (varying int32)0);
}
template <> unmasked varying int32 __bitcast<varying int32, varying float>(varying float x) {
    return @llvm.ispc.bitcast(x, (varying int32)0);
}
template <> unmasked varying uint32 __bitcast<varying uint32, varying float>(varying float x) {
    return @llvm.ispc.bitcast(x, (varying uint32)0);
}
template <> unmasked varying float __bitcast<varying float, varying uint32>(varying uint32 x) {
    return @llvm.ispc.bitcast(x, (varying float)0);
}
template <> unmasked varying float __bitcast<varying float, varying int32>(varying int32 x) {
    return @llvm.ispc.bitcast(x, (varying float)0);
}
template <> unmasked varying int64 __bitcast<varying int64, varying int64>(varying int64 x) { return x; }
template <> unmasked varying uint64 __bitcast<varying uint64, varying uint64>(varying uint64 x) { return x; }
template <> unmasked varying double __bitcast<varying double, varying double>(varying double x) { return x; }
template <> unmasked varying uint64 __bitcast<varying uint64, varying int64>(varying int64 x) {
    return @llvm.ispc.bitcast(x, (varying uint64)0);
}
template <> unmasked varying int64 __bitcast<varying int64, varying uint64>(varying uint64 x) {
    return @llvm.ispc.bitcast(x, (varying int64)0);
}
template <> unmasked varying int64 __bitcast<varying int64, varying double>(varying double x) {
    return @llvm.ispc.bitcast(x, (varying int64)0);
}
template <> unmasked varying uint64 __bitcast<varying uint64, varying double>(varying double x) {
    return @llvm.ispc.bitcast(x, (varying uint64)0);
}
template <> unmasked varying double __bitcast<varying double, varying uint64>(varying uint64 x) {
    return @llvm.ispc.bitcast(x, (varying double)0);
}
template <> unmasked varying double __bitcast<varying double, varying int64>(varying int64 x) {
    return @llvm.ispc.bitcast(x, (varying double)0);
}

template <typename T> varying T __idiv_t(varying T a, varying T b) {
    varying T res = 0;
    foreach_active(i) {
        uniform T x = __extract(a, i) / __extract(b, i);
        res = __insert(res, i, x);
    }
    return res;
}

template <typename T> varying T __irem_t(varying T a, varying T b) {
    varying T res = 0;
    foreach_active(i) {
        uniform T x = __extract(a, i) % __extract(b, i);
        res = __insert(res, i, x);
    }
    return res;
}

UNMANGLED CDECL int8 __sdiv_i8(int8 a, int8 b) { return __idiv_t(a, b); }
UNMANGLED CDECL int16 __sdiv_i16(int16 a, int16 b) { return __idiv_t(a, b); }
UNMANGLED CDECL int32 __sdiv_i32(int32 a, int32 b) { return __idiv_t(a, b); }
UNMANGLED CDECL int64 __sdiv_i64(int64 a, int64 b) { return __idiv_t(a, b); }
UNMANGLED CDECL uint8 __udiv_i8(uint8 a, uint8 b) { return __idiv_t(a, b); }
UNMANGLED CDECL uint16 __udiv_i16(uint16 a, uint16 b) { return __idiv_t(a, b); }
UNMANGLED CDECL uint32 __udiv_i32(uint32 a, uint32 b) { return __idiv_t(a, b); }
UNMANGLED CDECL uint64 __udiv_i64(uint64 a, uint64 b) { return __idiv_t(a, b); }

UNMANGLED CDECL int8 __srem_i8(int8 a, int8 b) { return __irem_t(a, b); }
UNMANGLED CDECL int16 __srem_i16(int16 a, int16 b) { return __irem_t(a, b); }
UNMANGLED CDECL int32 __srem_i32(int32 a, int32 b) { return __irem_t(a, b); }
UNMANGLED CDECL int64 __srem_i64(int64 a, int64 b) { return __irem_t(a, b); }
UNMANGLED CDECL uint8 __urem_i8(uint8 a, uint8 b) { return __irem_t(a, b); }
UNMANGLED CDECL uint16 __urem_i16(uint16 a, uint16 b) { return __irem_t(a, b); }
UNMANGLED CDECL uint32 __urem_i32(uint32 a, uint32 b) { return __irem_t(a, b); }
UNMANGLED CDECL uint64 __urem_i64(uint64 a, uint64 b) { return __irem_t(a, b); }

// vector ops
EXT uniform bool __extract_bool(IntMaskType vec, uniform int32 lane) { return __extract(vec, lane); }
EXT uniform int16 __extract_int16(int16 vec, uniform int32 lane) { return __extract(vec, lane); }
EXT uniform int32 __extract_int32(int32 vec, uniform int32 lane) { return __extract(vec, lane); }
EXT uniform int64 __extract_int64(int64 vec, uniform int32 lane) { return __extract(vec, lane); }
EXT uniform int8 __extract_int8(int8 vec, uniform int32 lane) { return __extract(vec, lane); }
EXT READNONE IntMaskType __insert_bool(IntMaskType vec, uniform int32 lane, uniform bool val) {
    return __insert(vec, lane, (uniform IntMaskType)val);
}
EXT READNONE int16 __insert_int16(int16 vec, uniform int32 lane, uniform int16 val) { return __insert(vec, lane, val); }
EXT READNONE int32 __insert_int32(int32 vec, uniform int32 lane, uniform int32 val) { return __insert(vec, lane, val); }
EXT READNONE int64 __insert_int64(int64 vec, uniform int32 lane, uniform int64 val) { return __insert(vec, lane, val); }
EXT READNONE int8 __insert_int8(int8 vec, uniform int32 lane, uniform int8 val) { return __insert(vec, lane, val); }

// // EXT uniform bool __is_compile_time_constant_mask(UIntMaskType);
// // EXT uniform bool __is_compile_time_constant_uniform_int32(uniform int32);
// // EXT uniform bool __is_compile_time_constant_varying_int32(varying int32);

// trigonometric functions
// __have_native_trigonometry
EXT READNONE uniform double __acos_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform float __acos_uniform_float(uniform float) { __not_supported(); }
EXT READNONE uniform float16 __acos_uniform_half(uniform float16) { __not_supported(); }
EXT READNONE varying double __acos_varying_double(varying double) { __not_supported(); }
EXT READNONE varying float __acos_varying_float(varying float) { __not_supported(); }
EXT READNONE varying float16 __acos_varying_half(varying float16) { __not_supported(); }
EXT READNONE uniform double __asin_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform float __asin_uniform_float(uniform float) { __not_supported(); }
EXT READNONE uniform float16 __asin_uniform_half(uniform float16) { __not_supported(); }
EXT READNONE varying double __asin_varying_double(varying double) { __not_supported(); }
EXT READNONE varying float __asin_varying_float(varying float) { __not_supported(); }
EXT READNONE varying float16 __asin_varying_half(varying float16) { __not_supported(); }
EXT READNONE uniform double __atan2_uniform_double(uniform double, uniform double) { __not_supported(); }
EXT READNONE uniform float __atan2_uniform_float(uniform float, uniform float) { __not_supported(); }
EXT READNONE uniform float16 __atan2_uniform_half(uniform float16, uniform float16) { __not_supported(); }
EXT READNONE varying double __atan2_varying_double(varying double, varying double) { __not_supported(); }
EXT READNONE varying float __atan2_varying_float(varying float, varying float) { __not_supported(); }
EXT READNONE varying float16 __atan2_varying_half(varying float16, varying float16) { __not_supported(); }
EXT READNONE uniform double __atan_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform float __atan_uniform_float(uniform float) { __not_supported(); }
EXT READNONE uniform float16 __atan_uniform_half(uniform float16) { __not_supported(); }
EXT READNONE varying double __atan_varying_double(varying double) { __not_supported(); }
EXT READNONE varying float __atan_varying_float(varying float) { __not_supported(); }
EXT READNONE varying float16 __atan_varying_half(varying float16) { __not_supported(); }
EXT READNONE uniform double __cos_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform float __cos_uniform_float(uniform float) { __not_supported(); }
EXT READNONE uniform float16 __cos_uniform_half(uniform float16) { __not_supported(); }
EXT READNONE varying double __cos_varying_double(varying double) { __not_supported(); }
EXT READNONE varying float __cos_varying_float(varying float) { __not_supported(); }
EXT READNONE varying float16 __cos_varying_half(varying float16) { __not_supported(); }
EXT void __sincos_uniform_double(uniform double, uniform int8 *uniform, uniform int8 *uniform) { __not_supported(); }
EXT void __sincos_uniform_float(uniform float, uniform int8 *uniform, uniform int8 *uniform) { __not_supported(); }
EXT void __sincos_uniform_half(uniform float16, uniform int8 *uniform, uniform int8 *uniform) { __not_supported(); }
EXT void __sincos_varying_double(varying double, uniform int8 *uniform, uniform int8 *uniform) { __not_supported(); }
EXT void __sincos_varying_float(varying float, uniform int8 *uniform, uniform int8 *uniform) { __not_supported(); }
EXT void __sincos_varying_half(varying float16, uniform int8 *uniform, uniform int8 *uniform) { __not_supported(); }
EXT READNONE uniform double __sin_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform float __sin_uniform_float(uniform float) { __not_supported(); }
EXT READNONE uniform float16 __sin_uniform_half(uniform float16) { __not_supported(); }
EXT READNONE varying double __sin_varying_double(varying double) { __not_supported(); }
EXT READNONE varying float __sin_varying_float(varying float) { __not_supported(); }
EXT READNONE varying float16 __sin_varying_half(varying float16) { __not_supported(); }
EXT READNONE uniform double __tan_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform float __tan_uniform_float(uniform float) { __not_supported(); }
EXT READNONE uniform float16 __tan_uniform_half(uniform float16) { __not_supported(); }
EXT READNONE varying double __tan_varying_double(varying double) { __not_supported(); }
EXT READNONE varying float __tan_varying_float(varying float) { __not_supported(); }
EXT READNONE varying float16 __tan_varying_half(varying float16) { __not_supported(); }

// saturating math functions
// m_hasSaturatingArithmetic
EXT uniform int64 __abs_ui64(uniform int64) { __not_supported(); }
EXT varying int64 __abs_vi64(varying int64) { __not_supported(); }
EXT uniform int8 __padds_ui8(uniform int8 x, uniform int8 y) { return @llvm.sadd.sat(x, y); }
EXT uniform int16 __padds_ui16(uniform int16 x, uniform int16 y) { return @llvm.sadd.sat(x, y); }
EXT uniform int32 __padds_ui32(uniform int32 x, uniform int32 y) { return @llvm.sadd.sat(x, y); }
EXT uniform int64 __padds_ui64(uniform int64 x, uniform int64 y) { return @llvm.sadd.sat(x, y); }
EXT varying int8 __padds_vi8(varying int8 x, varying int8 y) { return @llvm.sadd.sat(x, y); }
EXT varying int16 __padds_vi16(varying int16 x, varying int16 y) { return @llvm.sadd.sat(x, y); }
EXT varying int32 __padds_vi32(varying int32 x, varying int32 y) { return @llvm.sadd.sat(x, y); }
EXT varying int64 __padds_vi64(varying int64 x, varying int64 y) { return @llvm.sadd.sat(x, y); }
EXT uniform uint8 __paddus_ui8(uniform uint8 x, uniform uint8 y) { return @llvm.uadd.sat(x, y); }
EXT uniform uint16 __paddus_ui16(uniform uint16 x, uniform uint16 y) { return @llvm.uadd.sat(x, y); }
EXT uniform uint32 __paddus_ui32(uniform uint32 x, uniform uint32 y) { return @llvm.uadd.sat(x, y); }
EXT uniform uint64 __paddus_ui64(uniform uint64 x, uniform uint64 y) { return @llvm.uadd.sat(x, y); }
EXT varying uint8 __paddus_vi8(varying uint8 x, varying uint8 y) { return @llvm.uadd.sat(x, y); }
EXT varying uint16 __paddus_vi16(varying uint16 x, varying uint16 y) { return @llvm.uadd.sat(x, y); }
EXT varying uint32 __paddus_vi32(varying uint32 x, varying uint32 y) { return @llvm.uadd.sat(x, y); }
EXT varying uint64 __paddus_vi64(varying uint64 x, varying uint64 y) { return @llvm.uadd.sat(x, y); }
EXT uniform int8 __psubs_ui8(uniform int8 x, uniform int8 y) { return @llvm.ssub.sat(x, y); }
EXT uniform int16 __psubs_ui16(uniform int16 x, uniform int16 y) { return @llvm.ssub.sat(x, y); }
EXT uniform int32 __psubs_ui32(uniform int32 x, uniform int32 y) { return @llvm.ssub.sat(x, y); }
EXT uniform int64 __psubs_ui64(uniform int64 x, uniform int64 y) { return @llvm.ssub.sat(x, y); }
EXT varying int8 __psubs_vi8(varying int8 x, varying int8 y) { return @llvm.ssub.sat(x, y); }
EXT varying int16 __psubs_vi16(varying int16 x, varying int16 y) { return @llvm.ssub.sat(x, y); }
EXT varying int32 __psubs_vi32(varying int32 x, varying int32 y) { return @llvm.ssub.sat(x, y); }
EXT varying int64 __psubs_vi64(varying int64 x, varying int64 y) { return @llvm.ssub.sat(x, y); }
EXT uniform uint16 __psubus_ui16(uniform uint16 x, uniform uint16 y) { return @llvm.usub.sat(x, y); }
EXT uniform uint32 __psubus_ui32(uniform uint32 x, uniform uint32 y) { return @llvm.usub.sat(x, y); }
EXT uniform uint64 __psubus_ui64(uniform uint64 x, uniform uint64 y) { return @llvm.usub.sat(x, y); }
EXT uniform uint8 __psubus_ui8(uniform uint8 x, uniform uint8 y) { return @llvm.usub.sat(x, y); }
EXT varying uint16 __psubus_vi16(varying uint16 x, varying uint16 y) { return @llvm.usub.sat(x, y); }
EXT varying uint32 __psubus_vi32(varying uint32 x, varying uint32 y) { return @llvm.usub.sat(x, y); }
EXT varying uint64 __psubus_vi64(varying uint64 x, varying uint64 y) { return @llvm.usub.sat(x, y); }
EXT varying uint8 __psubus_vi8(varying uint8 x, varying uint8 y) { return @llvm.usub.sat(x, y); }
EXT uniform int8 __pmuls_ui8(uniform int8 x, uniform int8 y) { __not_supported(); }
EXT uniform int16 __pmuls_ui16(uniform int16 x, uniform int16 y) { __not_supported(); }
EXT uniform int32 __pmuls_ui32(uniform int32 x, uniform int32 y) { __not_supported(); }
EXT uniform int64 __pmuls_ui64(uniform int64 x, uniform int64 y) { __not_supported(); }
EXT varying int8 __pmuls_vi8(varying int8 x, varying int8 y) { __not_supported(); }
EXT varying int16 __pmuls_vi16(varying int16 x, varying int16 y) { __not_supported(); }
EXT varying int32 __pmuls_vi32(varying int32 x, varying int32 y) { __not_supported(); }
EXT varying int64 __pmuls_vi64(varying int64 x, varying int64 y) { __not_supported(); }
EXT uniform int8 __pmulus_ui8(uniform int8 x, uniform int8 y) { __not_supported(); }
EXT uniform int16 __pmulus_ui16(uniform int16 x, uniform int16 y) { __not_supported(); }
EXT uniform int32 __pmulus_ui32(uniform int32 x, uniform int32 y) { __not_supported(); }
EXT uniform int64 __pmulus_ui64(uniform int64 x, uniform int64 y) { __not_supported(); }
EXT varying int8 __pmulus_vi8(varying int8 x, varying int8 y) { __not_supported(); }
EXT varying int16 __pmulus_vi16(varying int16 x, varying int16 y) { __not_supported(); }
EXT varying int32 __pmulus_vi32(varying int32 x, varying int32 y) { __not_supported(); }
EXT varying int64 __pmulus_vi64(varying int64 x, varying int64 y) { __not_supported(); }

// rounding floats and doubles
EXT READNONE uniform double __ceil_uniform_double(uniform double x) { return @llvm.ceil(x); }
EXT READNONE uniform float __ceil_uniform_float(uniform float x) { return @llvm.ceil(x); }
EXT READNONE varying double __ceil_varying_double(varying double x) { return @llvm.ceil(x); }
EXT READNONE varying float __ceil_varying_float(varying float x) { return @llvm.ceil(x); }
EXT READNONE uniform double __floor_uniform_double(uniform double x) { return @llvm.floor(x); }
EXT READNONE uniform float __floor_uniform_float(uniform float x) { return @llvm.floor(x); }
EXT READNONE varying double __floor_varying_double(varying double x) { return @llvm.floor(x); }
EXT READNONE varying float __floor_varying_float(varying float x) { return @llvm.floor(x); }
EXT READNONE uniform double __round_uniform_double(uniform double x) { return @llvm.roundeven(x); }
EXT READNONE uniform float __round_uniform_float(uniform float x) { return @llvm.roundeven(x); }
EXT READNONE varying double __round_varying_double(varying double x) { return @llvm.roundeven(x); }
EXT READNONE varying float __round_varying_float(varying float x) { return @llvm.roundeven(x); }

// rcp
EXT READNONE uniform float __rcp_fast_uniform_float(uniform float x) { return 1 / x; }
EXT READNONE varying float __rcp_fast_varying_float(varying float x) { return 1 / x; }
EXT READNONE uniform float __rcp_uniform_float(uniform float x) { return 1 / x; }
EXT READNONE varying float __rcp_varying_float(varying float x) { return 1 / x; }

// rsqrt
EXT READNONE varying float __rsqrt_fast_varying_float(varying float x) { return 1 / @llvm.sqrt(x); }
EXT READNONE uniform float __rsqrt_fast_uniform_float(uniform float x) { return 1 / @llvm.sqrt(x); }
EXT READNONE varying float __rsqrt_varying_float(varying float x) { return 1 / @llvm.sqrt(x); }
EXT READNONE uniform float __rsqrt_uniform_float(uniform float x) { return 1 / @llvm.sqrt(x); }

// sqrt for floats and doubles
EXT READNONE uniform double __sqrt_uniform_double(uniform double x) { return @llvm.sqrt(x); }
EXT READNONE uniform float __sqrt_uniform_float(uniform float x) { return @llvm.sqrt(x); }
EXT READNONE varying double __sqrt_varying_double(varying double x) { return @llvm.sqrt(x); }
EXT READNONE varying float __sqrt_varying_float(varying float x) { return @llvm.sqrt(x); }

// Implements uniform and varying trunc() for float and double types.
EXT READNONE uniform double __trunc_uniform_double(uniform double x) { return @llvm.trunc(x); }
EXT READNONE uniform float __trunc_uniform_float(uniform float x) { return @llvm.trunc(x); }
EXT READNONE varying double __trunc_varying_double(varying double x) { return @llvm.trunc(x); }
EXT READNONE varying float __trunc_varying_float(varying float x) { return @llvm.trunc(x); }

// transcendental float math functions
// __have_native_transcendentals
EXT READNONE uniform double __exp_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform float __exp_uniform_float(uniform float) { __not_supported(); }
EXT READNONE varying double __exp_varying_double(varying double) { __not_supported(); }
EXT READNONE varying float __exp_varying_float(varying float) { __not_supported(); }
EXT READNONE uniform double __log_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform float __log_uniform_float(uniform float) { __not_supported(); }
EXT READNONE varying double __log_varying_double(varying double) { __not_supported(); }
EXT READNONE varying float __log_varying_float(varying float) { __not_supported(); }
EXT READNONE uniform double __pow_uniform_double(uniform double, uniform double) { __not_supported(); }
EXT READNONE uniform float __pow_uniform_float(uniform float, uniform float) { __not_supported(); }
EXT READNONE varying double __pow_varying_double(varying double, varying double) { __not_supported(); }
EXT READNONE varying float __pow_varying_float(varying float, varying float) { __not_supported(); }

// __have_native_rcpd
EXT READNONE uniform double __rcp_fast_uniform_double(uniform double) { __not_supported(); }
EXT READNONE uniform double __rcp_uniform_double(uniform double) { __not_supported(); }
EXT READNONE varying double __rcp_fast_varying_double(varying double) { __not_supported(); }
EXT READNONE varying double __rcp_varying_double(varying double) { __not_supported(); }

// __have_native_half_full_support
EXT READNONE uniform float16 __rcp_uniform_half(uniform float16) { __not_supported(); }
EXT READNONE varying float16 __rcp_varying_half(varying float16) { __not_supported(); }
EXT READNONE uniform float16 __rsqrt_uniform_half(uniform float16) { __not_supported(); }
EXT READNONE varying float16 __rsqrt_varying_half(varying float16) { __not_supported(); }

// __have_native_rsqrtd
EXT READNONE uniform double __rsqrt_fast_uniform_double(uniform double) { __not_supported(); }
EXT READNONE varying double __rsqrt_fast_varying_double(varying double) { __not_supported(); }
EXT READNONE uniform double __rsqrt_uniform_double(uniform double) { __not_supported(); }
EXT READNONE varying double __rsqrt_varying_double(varying double) { __not_supported(); }

// dot product
// __have_intel_vnni
EXT READNONE varying int32 __dot2add_i16i16packed_sat(varying uint32, varying uint32, varying int32) {
    __not_supported();
}
EXT READNONE varying int32 __dot2add_i16i16packed(varying uint32, varying uint32, varying int32) { __not_supported(); }
EXT READNONE varying int32 __dot4add_u8i8packed_sat(varying uint32, varying uint32, varying int32) {
    __not_supported();
}
// __have_intel_vnni || __have_arm_i8mm
EXT READNONE varying int32 __dot4add_u8i8packed(varying uint32, varying uint32, varying int32) { __not_supported(); }

// __have_arm_dot_product || __have_intel_vnni_int8
EXT READNONE varying uint32 __dot4add_u8u8packed(varying uint32, varying uint32, varying uint32) { __not_supported(); }
EXT READNONE varying int32 __dot4add_i8i8packed(varying uint32, varying uint32, varying int32) { __not_supported(); }

// __have_intel_vnni_int8
EXT READNONE varying uint32 __dot4add_u8u8packed_sat(varying uint32, varying uint32, varying uint32) {
    __not_supported();
}
EXT READNONE varying int32 __dot4add_i8i8packed_sat(varying uint32, varying uint32, varying int32) {
    __not_supported();
}

// __have_intel_vnni_int16
EXT READNONE varying uint32 __dot2add_u16u16packed_sat(varying uint32, varying uint32, varying uint32) {
    __not_supported();
}
EXT READNONE varying uint32 __dot2add_u16u16packed(varying uint32, varying uint32, varying uint32) {
    __not_supported();
}

EXT READNONE varying int32 __dot2add_u16i16packed_sat(varying uint32, varying uint32, varying int32) {
    __not_supported();
}
EXT READNONE varying int32 __dot2add_u16i16packed(varying uint32, varying uint32, varying int32) { __not_supported(); }

// __have_native_half_converts
EXT uniform int16 __float_to_half_uniform(uniform float x) { __not_supported(); }
EXT varying int16 __float_to_half_varying(varying float x) { __not_supported(); }
EXT uniform float __half_to_float_uniform(uniform int16 x) { __not_supported(); }
EXT varying float __half_to_float_varying(varying int16 x) { __not_supported(); }

EXT void __prefetch_read_sized_uniform_1(uniform int8 *uniform, uniform int8) { __not_supported(); }
EXT void __prefetch_read_sized_uniform_2(uniform int8 *uniform, uniform int8) { __not_supported(); }
EXT void __prefetch_read_sized_uniform_3(uniform int8 *uniform, uniform int8) { __not_supported(); }
EXT void __prefetch_read_sized_uniform_nt(uniform int8 *uniform, uniform int8) { __not_supported(); }
EXT void __prefetch_read_sized_varying_1(varying int64, uniform int8, UIntMaskType) { __not_supported(); }
EXT void __prefetch_read_sized_varying_2(varying int64, uniform int8, UIntMaskType) { __not_supported(); }
EXT void __prefetch_read_sized_varying_3(varying int64, uniform int8, UIntMaskType) { __not_supported(); }
EXT void __prefetch_read_sized_varying_nt(varying int64, uniform int8, UIntMaskType) { __not_supported(); }

// memcpy/memmove/memset
EXT void __memcpy32(uniform int8 *uniform dst, uniform int8 *uniform src, uniform int32 len) {
    @llvm.memcpy(dst, src, len, false);
}
EXT void __memcpy64(uniform int8 *uniform dst, uniform int8 *uniform src, uniform int64 len) {
    @llvm.memcpy(dst, src, len, false);
}
EXT void __memmove32(uniform int8 *uniform dst, uniform int8 *uniform src, uniform int32 len) {
    @llvm.memmove(dst, src, len, false);
}
EXT void __memmove64(uniform int8 *uniform dst, uniform int8 *uniform src, uniform int64 len) {
    @llvm.memmove(dst, src, len, false);
}
EXT void __memset32(uniform int8 *uniform dst, uniform int8 val, uniform int32 len) {
    @llvm.memset(dst, val, len, false);
}
EXT void __memset64(uniform int8 *uniform dst, uniform int8 val, uniform int64 len) {
    @llvm.memset(dst, val, len, false);
}

// assume and assert
EXT uniform int32 puts(uniform int8 *uniform str);
EXT void abort();
EXT void __do_assume_uniform(uniform bool test) { @llvm.assume(test); }

EXT void __do_assert_uniform(uniform int8 *uniform str, uniform bool test, UIntMaskType) {
    if (!test) {
        puts(str);
        abort();
    }
}
EXT void __do_assert_varying(uniform int8 *uniform str, UIntMaskType test, UIntMaskType mask) {
    if (__any((test ^ 1) & mask)) {
        puts(str);
        abort();
    }
}

EXT void __fastmath() { __not_supported(); }
// FTZ/DAZ functions
#if ISPC_TARGET_NEON
EXT uniform SizeType __set_ftz_daz_flags() { __not_supported(); }
EXT void __restore_ftz_daz_flags(uniform SizeType) { __not_supported(); }
#else  // ISPC_TARGET_NEON
EXT uniform int32 __set_ftz_daz_flags() { __not_supported(); }
EXT void __restore_ftz_daz_flags(uniform int32) { __not_supported(); }
#endif // ISPC_TARGET_NEON

// rdrand
EXT uniform bool __rdrand_i16(uniform int8 *uniform) { __not_supported(); }
EXT uniform bool __rdrand_i32(uniform int8 *uniform) { __not_supported(); }
EXT uniform bool __rdrand_i64(uniform int8 *uniform) { __not_supported(); }

// sign extension
EXT READNONE uniform int32 __sext_uniform_bool(uniform bool x) { return x ? -1 : 0; }
EXT READNONE varying int32 __sext_varying_bool(bool x) { return x ? -1 : 0; }

// read hw clock
EXT uniform int64 __clock() { return @llvm.readcyclecounter(); }

EXT READNONE void __memory_barrier() { @llvm.ispc.fence.seq_cst(); }

// stdlib transcendental functions
// These functions provide entrypoints that call out to the libm
// implementations of the transcendental functions
EXT READNONE uniform float sinf(uniform float);
EXT READNONE uniform float cosf(uniform float);
EXT void sincosf(uniform float, uniform float *uniform, uniform float *uniform);
EXT READNONE uniform float asinf(uniform float);
EXT READNONE uniform float acosf(uniform float);
EXT READNONE uniform float tanf(uniform float);
EXT READNONE uniform float atanf(uniform float);
EXT READNONE uniform float atan2f(uniform float, uniform float);
EXT READNONE uniform float expf(uniform float);
EXT READNONE uniform float logf(uniform float);
EXT READNONE uniform float powf(uniform float, uniform float);
EXT READNONE uniform float cbrtf(uniform float);

EXT READNONE uniform float __stdlib_sinf(uniform float x) { return sinf(x); }
EXT READNONE uniform float __stdlib_cosf(uniform float x) { return cosf(x); }
EXT void __stdlib_sincosf(uniform float x, uniform float *uniform sin_v, uniform float *uniform cos_v) {
    sincosf(x, sin_v, cos_v);
}
EXT READNONE uniform float __stdlib_asinf(uniform float x) { return asinf(x); }
EXT READNONE uniform float __stdlib_acosf(uniform float x) { return acosf(x); }
EXT READNONE uniform float __stdlib_tanf(uniform float x) { return tanf(x); }
EXT READNONE uniform float __stdlib_atanf(uniform float x) { return atanf(x); }
EXT READNONE uniform float __stdlib_atan2f(uniform float x, uniform float y) { return atan2f(x, y); }
EXT READNONE uniform float __stdlib_logf(uniform float x) { return logf(x); }
EXT READNONE uniform float __stdlib_expf(uniform float x) { return expf(x); }
EXT READNONE uniform float __stdlib_powf(uniform float x, uniform float y) { return powf(x, y); }
EXT READNONE uniform float __stdlib_cbrtf(uniform float x) { return cbrtf(x); }

EXT READNONE uniform double sin(uniform double);
EXT READNONE uniform double cos(uniform double);
EXT void sincos(uniform double, uniform double *uniform, uniform double *uniform);
EXT READNONE uniform double asin(uniform double);
EXT READNONE uniform double acos(uniform double);
EXT READNONE uniform double tan(uniform double);
EXT READNONE uniform double atan(uniform double);
EXT READNONE uniform double atan2(uniform double, uniform double);
EXT READNONE uniform double exp(uniform double);
EXT READNONE uniform double log(uniform double);
EXT READNONE uniform double pow(uniform double, uniform double);
EXT READNONE uniform double cbrt(uniform double);

EXT READNONE uniform double __stdlib_sin(uniform double x) { return sin(x); }
EXT READNONE uniform double __stdlib_cos(uniform double x) { return cos(x); }
EXT void __stdlib_sincos(uniform double x, uniform double *uniform sin_v, uniform double *uniform cos_v) {
    sincos(x, sin_v, cos_v);
}
EXT READNONE uniform double __stdlib_asin(uniform double x) { return asin(x); }
EXT READNONE uniform double __stdlib_acos(uniform double x) { return acos(x); }
EXT READNONE uniform double __stdlib_tan(uniform double x) { return tan(x); }
EXT READNONE uniform double __stdlib_atan(uniform double x) { return atan(x); }
EXT READNONE uniform double __stdlib_atan2(uniform double x, uniform double y) { return atan2(x, y); }
EXT READNONE uniform double __stdlib_log(uniform double x) { return log(x); }
EXT READNONE uniform double __stdlib_exp(uniform double x) { return exp(x); }
EXT READNONE uniform double __stdlib_pow(uniform double x, uniform double y) { return pow(x, y); }
EXT READNONE uniform double __stdlib_cbrt(uniform double x) { return cbrt(x); }

EXT READNONE uniform float16 __round_uniform_half(uniform float16 x) { return @llvm.roundeven(x); }
EXT READNONE varying float16 __round_varying_half(varying float16 x) { return @llvm.roundeven(x); }
EXT READNONE uniform float16 __floor_uniform_half(uniform float16 x) { return @llvm.floor(x); }
EXT READNONE varying float16 __floor_varying_half(varying float16 x) { return @llvm.floor(x); }
EXT READNONE uniform float16 __ceil_uniform_half(uniform float16 x) { return @llvm.ceil(x); }
EXT READNONE varying float16 __ceil_varying_half(varying float16 x) { return @llvm.ceil(x); }
EXT READNONE uniform float16 __trunc_uniform_half(uniform float16 x) { return @llvm.trunc(x); }
EXT READNONE varying float16 __trunc_varying_half(varying float16 x) { return @llvm.trunc(x); }
EXT READNONE uniform float16 __log_uniform_half(uniform float16 x) { return @llvm.log(x); }
EXT READNONE varying float16 __log_varying_half(varying float16 x) { return @llvm.log(x); }
EXT READNONE uniform float16 __exp_uniform_half(uniform float16 x) { return @llvm.exp(x); }
EXT READNONE varying float16 __exp_varying_half(varying float16 x) { return @llvm.exp(x); }
EXT READNONE uniform float16 __pow_uniform_half(uniform float16 x, uniform float16 y) { return @llvm.pow(x, y); }
EXT READNONE varying float16 __pow_varying_half(varying float16 x, varying float16 y) { return @llvm.pow(x, y); }
EXT READNONE uniform float16 __sqrt_uniform_half(uniform float16 x) { return @llvm.sqrt(x); }
EXT READNONE varying float16 __sqrt_varying_half(varying float16 x) { return @llvm.sqrt(x); }

// 16-bit float reduction
EXT READNONE uniform float16 __reduce_add_half(varying float16 x) { return @llvm.vector.reduce.fadd(-0.0f16, x); }
EXT READNONE uniform float16 __reduce_max_half(varying float16 x) { return @llvm.vector.reduce.fmax(x); }
EXT READNONE uniform float16 __reduce_min_half(varying float16 x) { return @llvm.vector.reduce.fmin(x); }

// 16-bit float min and max functions
EXT READNONE uniform float16 __min_uniform_half(uniform float16 x, uniform float16 y) { return x < y ? x : y; }
EXT READNONE varying float16 __min_varying_half(varying float16 x, varying float16 y) { return x < y ? x : y; }
EXT READNONE uniform float16 __max_uniform_half(uniform float16 x, uniform float16 y) { return x > y ? x : y; }
EXT READNONE varying float16 __max_varying_half(varying float16 x, varying float16 y) { return x > y ? x : y; }

// horizontal int8 ops
EXT READNONE uniform int16 __reduce_add_int8(varying int8 x) { return @llvm.vector.reduce.add((int16)x); }
EXT READNONE uniform int8 __reduce_min_int8(varying int8 x) { return @llvm.vector.reduce.smin(x); }
EXT READNONE uniform int8 __reduce_max_int8(varying int8 x) { return @llvm.vector.reduce.smax(x); }

// horizontal int16 ops
EXT READNONE uniform int16 __reduce_add_int16(varying int16 x) { return @llvm.vector.reduce.add((int32)x); }
EXT READNONE uniform int16 __reduce_min_int16(varying int16 x) { return @llvm.vector.reduce.smin(x); }
EXT READNONE uniform int16 __reduce_max_int16(varying int16 x) { return @llvm.vector.reduce.smax(x); }

// horizontal float ops
EXT READNONE uniform float __reduce_add_float(varying float x) { return @llvm.vector.reduce.fadd(-0.0f, x); }
EXT READNONE uniform float __reduce_max_float(varying float x) { return @llvm.vector.reduce.fmax(x); }
EXT READNONE uniform float __reduce_min_float(varying float x) { return @llvm.vector.reduce.fmin(x); }

// horizontal int32 ops
// TODO!: don't we lose result with reduce_add having int32 return type?
EXT READNONE uniform int32 __reduce_add_int32(varying int32 x) { return @llvm.vector.reduce.add((int64)x); }
EXT READNONE uniform int32 __reduce_min_int32(varying int32 x) { return @llvm.vector.reduce.smin(x); }
EXT READNONE uniform int32 __reduce_max_int32(varying int32 x) { return @llvm.vector.reduce.smax(x); }

// horizontal uint8 ops
EXT READNONE uniform uint8 __reduce_min_uint8(varying uint8 x) { return @llvm.vector.reduce.umin(x); }
EXT READNONE uniform uint8 __reduce_max_uint8(varying uint8 x) { return @llvm.vector.reduce.umax(x); }

// horizontal uint16 ops
EXT READNONE uniform uint16 __reduce_min_uint16(varying uint16 x) { return @llvm.vector.reduce.umin(x); }
EXT READNONE uniform uint16 __reduce_max_uint16(varying uint16 x) { return @llvm.vector.reduce.umax(x); }

// horizontal uint32 ops
EXT READNONE uniform uint32 __reduce_min_uint32(varying uint32 x) { return @llvm.vector.reduce.umin(x); }
EXT READNONE uniform uint32 __reduce_max_uint32(varying uint32 x) { return @llvm.vector.reduce.umax(x); }

// horizontal double ops
EXT READNONE uniform double __reduce_add_double(varying double x) { return @llvm.vector.reduce.fadd(-0.0d, x); }
EXT READNONE uniform double __reduce_min_double(varying double x) { return @llvm.vector.reduce.fmin(x); }
EXT READNONE uniform double __reduce_max_double(varying double x) { return @llvm.vector.reduce.fmax(x); }

// horizontal int64 ops
EXT READNONE uniform int64 __reduce_add_int64(varying int64 x) { return @llvm.vector.reduce.add(x); }
EXT READNONE uniform int64 __reduce_min_int64(varying int64 x) { return @llvm.vector.reduce.smin(x); }
EXT READNONE uniform int64 __reduce_max_int64(varying int64 x) { return @llvm.vector.reduce.smax(x); }

// horizontal uint64 ops
EXT READNONE uniform uint64 __reduce_min_uint64(varying uint64 x) { return @llvm.vector.reduce.umin(x); }
EXT READNONE uniform uint64 __reduce_max_uint64(varying uint64 x) { return @llvm.vector.reduce.umax(x); }

// horizontal equality ops
template <typename T> unmasked uniform bool __reduce_equal(varying T x, uniform T *uniform ret, UIntMaskType mask) {
    uniform int first_active = __count_trailing_zeros_uniform_i64(__movmsk(mask));
    uniform T first_val = __extract(x, first_active);
    varying T all_equal = first_val;
    // Only compare active lanes
    varying bool matches = (x == all_equal);
    // True if matches OR lane is inactive
    if (__all(matches || !mask)) {
        *ret = first_val;
        return true;
    }
    return false;
}
EXT uniform bool __reduce_equal_int8(varying int8 val, uniform int8 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<int8>(val, ret, mask);
}
EXT uniform bool __reduce_equal_uint8(varying uint8 val, uniform uint8 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<uint8>(val, ret, mask);
}
EXT uniform bool __reduce_equal_int16(varying int16 val, uniform int16 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<int16>(val, ret, mask);
}
EXT uniform bool __reduce_equal_uint16(varying uint16 val, uniform uint16 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<uint16>(val, ret, mask);
}
EXT uniform bool __reduce_equal_int32(varying int32 val, uniform int32 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<int32>(val, ret, mask);
}
EXT uniform bool __reduce_equal_uint32(varying uint32 val, uniform uint32 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<uint32>(val, ret, mask);
}
EXT uniform bool __reduce_equal_int64(varying int64 val, uniform int64 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<int64>(val, ret, mask);
}
EXT uniform bool __reduce_equal_uint64(varying uint64 val, uniform uint64 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<uint64>(val, ret, mask);
}
EXT uniform bool __reduce_equal_half(varying float16 val, uniform float16 *uniform ret, UIntMaskType mask) {
    return __reduce_equal<float16>(val, ret, mask);
}
EXT uniform bool __reduce_equal_float(varying float val, uniform float *uniform ret, UIntMaskType mask) {
    return __reduce_equal<float>(val, ret, mask);
}
EXT uniform bool __reduce_equal_double(varying double val, uniform double *uniform ret, UIntMaskType mask) {
    return __reduce_equal<double>(val, ret, mask);
}

// 64-bit integer min and max functions
EXT READNONE uniform int64 __min_uniform_int64(uniform int64 x, uniform int64 y) { return x < y ? x : y; }
EXT READNONE uniform int64 __max_uniform_int64(uniform int64 x, uniform int64 y) { return x > y ? x : y; }
EXT READNONE varying int64 __min_varying_int64(varying int64 x, varying int64 y) { return x < y ? x : y; }
EXT READNONE varying int64 __max_varying_int64(varying int64 x, varying int64 y) { return x > y ? x : y; }
EXT READNONE uniform uint64 __min_uniform_uint64(uniform uint64 x, uniform uint64 y) { return x < y ? x : y; }
EXT READNONE uniform uint64 __max_uniform_uint64(uniform uint64 x, uniform uint64 y) { return x > y ? x : y; }
EXT READNONE varying uint64 __min_varying_uint64(varying uint64 x, varying uint64 y) { return x < y ? x : y; }
EXT READNONE varying uint64 __max_varying_uint64(varying uint64 x, varying uint64 y) { return x > y ? x : y; }

// float min/max
EXT READNONE uniform float __min_uniform_float(uniform float x, uniform float y) { return @llvm.minnum(x, y); }
EXT READNONE uniform float __max_uniform_float(uniform float x, uniform float y) { return @llvm.maxnum(x, y); }
EXT READNONE varying float __min_varying_float(varying float x, varying float y) { return @llvm.minnum(x, y); }
EXT READNONE varying float __max_varying_float(varying float x, varying float y) { return @llvm.maxnum(x, y); }

// int8 min/max
EXT READNONE uniform int8 __min_uniform_int8(uniform int8 x, uniform int8 y) { return x < y ? x : y; }
EXT READNONE uniform int8 __max_uniform_int8(uniform int8 x, uniform int8 y) { return x > y ? x : y; }
EXT READNONE uniform uint8 __min_uniform_uint8(uniform uint8 x, uniform uint8 y) { return x < y ? x : y; }
EXT READNONE uniform uint8 __max_uniform_uint8(uniform uint8 x, uniform uint8 y) { return x > y ? x : y; }
EXT READNONE varying int8 __min_varying_int8(varying int8 x, varying int8 y) { return x < y ? x : y; }
EXT READNONE varying int8 __max_varying_int8(varying int8 x, varying int8 y) { return x > y ? x : y; }
EXT READNONE varying uint8 __min_varying_uint8(varying uint8 x, varying uint8 y) { return x < y ? x : y; }
EXT READNONE varying uint8 __max_varying_uint8(varying uint8 x, varying uint8 y) { return x > y ? x : y; }

// int16 min/max
EXT READNONE uniform int16 __min_uniform_int16(uniform int16 x, uniform int16 y) { return x < y ? x : y; }
EXT READNONE uniform int16 __max_uniform_int16(uniform int16 x, uniform int16 y) { return x > y ? x : y; }
EXT READNONE uniform uint16 __min_uniform_uint16(uniform uint16 x, uniform uint16 y) { return x < y ? x : y; }
EXT READNONE uniform uint16 __max_uniform_uint16(uniform uint16 x, uniform uint16 y) { return x > y ? x : y; }
EXT READNONE varying int16 __min_varying_int16(varying int16 x, varying int16 y) { return x < y ? x : y; }
EXT READNONE varying int16 __max_varying_int16(varying int16 x, varying int16 y) { return x > y ? x : y; }
EXT READNONE varying uint16 __min_varying_uint16(varying uint16 x, varying uint16 y) { return x < y ? x : y; }
EXT READNONE varying uint16 __max_varying_uint16(varying uint16 x, varying uint16 y) { return x > y ? x : y; }

// unsigned int min/max
EXT READNONE uniform int32 __min_uniform_int32(uniform int32 x, uniform int32 y) { return x < y ? x : y; }
EXT READNONE uniform int32 __max_uniform_int32(uniform int32 x, uniform int32 y) { return x > y ? x : y; }
EXT READNONE uniform uint32 __min_uniform_uint32(uniform uint32 x, uniform uint32 y) { return x < y ? x : y; }
EXT READNONE uniform uint32 __max_uniform_uint32(uniform uint32 x, uniform uint32 y) { return x > y ? x : y; }
EXT READNONE varying int32 __min_varying_int32(varying int32 x, varying int32 y) { return x < y ? x : y; }
EXT READNONE varying int32 __max_varying_int32(varying int32 x, varying int32 y) { return x > y ? x : y; }
EXT READNONE varying uint32 __min_varying_uint32(varying uint32 x, varying uint32 y) { return x < y ? x : y; }
EXT READNONE varying uint32 __max_varying_uint32(varying uint32 x, varying uint32 y) { return x > y ? x : y; }

// double precision min/max
EXT READNONE uniform double __min_uniform_double(uniform double x, uniform double y) { return @llvm.minnum(x, y); }
EXT READNONE uniform double __max_uniform_double(uniform double x, uniform double y) { return @llvm.maxnum(x, y); }
EXT READNONE varying double __min_varying_double(varying double x, varying double y) { return @llvm.minnum(x, y); }
EXT READNONE varying double __max_varying_double(varying double x, varying double y) { return @llvm.maxnum(x, y); }

// int8/int16 avg builtins
EXT varying uint8 __avg_up_uint8(varying uint8 a, varying uint8 b) { return ((uint16)a + (uint16)b + 1) >> 1; }
EXT varying uint16 __avg_up_uint16(varying uint16 a, varying uint16 b) { return ((uint32)a + (uint32)b + 1) >> 1; }
EXT varying uint8 __avg_down_uint8(varying uint8 a, varying uint8 b) { return ((uint16)a + (uint16)b) >> 1; }
EXT varying uint16 __avg_down_uint16(varying uint16 a, varying uint16 b) { return ((uint32)a + (uint32)b) >> 1; }
// It’s fine to use a logical right shift for signed types here because we
// perform the shift in a larger data type and then truncate back to the
// original size. It doesn’t matter whether the most significant bit is set to
// 0 or 1, because it will be discarded during truncation.
EXT varying int8 __avg_up_int8(varying int8 a, varying int8 b) { return ((int16)a + (int16)b + 1) >> 1; }
EXT varying int16 __avg_up_int16(varying int16 a, varying int16 b) { return ((int32)a + (int32)b + 1) >> 1; }
EXT varying int8 __avg_down_int8(varying int8 a, varying int8 b) { return ((int16)a + (int16)b) >> 1; }
EXT varying int16 __avg_down_int16(varying int16 a, varying int16 b) { return ((int32)a + (int32)b) >> 1; }

// count trailing zeros
EXT READNONE uniform int32 __count_trailing_zeros_uniform_i32(uniform int32 x) { return @llvm.cttz(x, false); }
EXT READNONE uniform int64 __count_trailing_zeros_uniform_i64(uniform int64 x) { return @llvm.cttz(x, false); }
EXT READNONE uniform int32 __count_leading_zeros_uniform_i32(uniform int32 x) { return @llvm.ctlz(x, false); }
EXT READNONE uniform int64 __count_leading_zeros_uniform_i64(uniform int64 x) { return @llvm.ctlz(x, false); }

EXT READNONE varying int32 __count_trailing_zeros_varying_i32(varying int32 x) {
    varying int32 res = @llvm.cttz(x, false);
    return res;
}
EXT READNONE varying int64 __count_trailing_zeros_varying_i64(varying int64 x) {
    varying int64 res = @llvm.cttz(x, false);
    return res;
}
EXT READNONE varying int32 __count_leading_zeros_varying_i32(varying int32 x) {
    varying int32 res = @llvm.ctlz(x, false);
    return res;
}
EXT READNONE varying int64 __count_leading_zeros_varying_i64(varying int64 x) {
    varying int32 res = @llvm.ctlz(x, false);
    return res;
}

// population count
EXT uniform int32 __popcnt_int32_uniform(uniform int32 x) { return @llvm.ctpop(x); }
EXT uniform int32 __popcnt_int64_uniform(uniform int64 x) { return (int32)@llvm.ctpop(x); }
EXT varying int32 __popcnt_int32_varying(varying int32 x, UIntMaskType mask) {
    int r;
    for (uniform int i = 0; i < programCount; ++i)
        r = __insert(r, i, __popcnt_int32_uniform(__extract(x, i)));
    return mask ? r : 0;
}
EXT varying int32 __popcnt_int64_varying(varying int64 x, UIntMaskType mask) {
    int r;
    for (uniform int i = 0; i < programCount; ++i)
        r = __insert(r, i, __popcnt_int64_uniform(__extract(x, i)));
    return mask ? r : 0;
}

// various bitcasts from one type to another
EXT READNONE uniform double __doublebits_uniform_int64(uniform int64 x) {
    return @llvm.ispc.bitcast(x, (uniform double)0);
}
EXT READNONE varying double __doublebits_varying_int64(varying int64 x) {
    return @llvm.ispc.bitcast(x, (varying double)0);
}
EXT READNONE uniform float __floatbits_uniform_int32(uniform int32 x) {
    return @llvm.ispc.bitcast(x, (uniform float)0);
}
EXT READNONE varying float __floatbits_varying_int32(varying int32 x) {
    return @llvm.ispc.bitcast(x, (varying float)0);
}
EXT READNONE uniform float16 __halfbits_uniform_int16(uniform int16 x) {
    return @llvm.ispc.bitcast(x, (uniform float16)0);
}
EXT READNONE varying float16 __halfbits_varying_int16(varying int16 x) {
    return @llvm.ispc.bitcast(x, (varying float16)0);
}
EXT READNONE uniform int64 __intbits_uniform_double(uniform double x) {
    return @llvm.ispc.bitcast(x, (uniform int64)0);
}
EXT READNONE uniform int32 __intbits_uniform_float(uniform float x) { return @llvm.ispc.bitcast(x, (uniform int32)0); }
EXT READNONE uniform int16 __intbits_uniform_half(uniform float16 x) { return @llvm.ispc.bitcast(x, (uniform int16)0); }
EXT READNONE varying int64 __intbits_varying_double(varying double x) {
    return @llvm.ispc.bitcast(x, (varying int64)0);
}
EXT READNONE varying int32 __intbits_varying_float(varying float x) { return @llvm.ispc.bitcast(x, (varying int32)0); }
EXT READNONE varying int16 __intbits_varying_half(varying float16 x) { return @llvm.ispc.bitcast(x, (varying int16)0); }

// broadcast
template <typename T> unmasked varying T __broadcast(varying T v, uniform int32 i) { return __extract(v, i); }
EXT READNONE double __broadcast_double(double v, uniform int32 i) { return __broadcast<double>(v, i); }
EXT READNONE float __broadcast_float(float v, uniform int32 i) { return __broadcast<float>(v, i); }
EXT READNONE float16 __broadcast_half(float16 v, uniform int32 i) { return __broadcast<float16>(v, i); }
EXT READNONE int16 __broadcast_i16(int16 v, uniform int32 i) { return __broadcast<int16>(v, i); }
EXT READNONE int32 __broadcast_i32(int32 v, uniform int32 i) { return __broadcast<int32>(v, i); }
EXT READNONE int64 __broadcast_i64(int64 v, uniform int32 i) { return __broadcast<int64>(v, i); }
EXT READNONE int8 __broadcast_i8(int8 v, uniform int32 i) { return __broadcast<int8>(v, i); }

// exclusive scan
template <typename T> unmasked varying T __exclusive_scan_add(varying T v, UIntMaskType mask) {
    varying T res = 0;
    varying T val = mask ? v : 0;
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        res = __insert(res, i, __extract(val, i - 1) + __extract(res, i - 1));
    }
    return res;
}
EXT double __exclusive_scan_add_double(double v, UIntMaskType mask) { return __exclusive_scan_add<double>(v, mask); }
EXT float __exclusive_scan_add_float(float v, UIntMaskType mask) { return __exclusive_scan_add<float>(v, mask); }
EXT float16 __exclusive_scan_add_half(float16 v, UIntMaskType mask) { return __exclusive_scan_add<float16>(v, mask); }
EXT int8 __exclusive_scan_add_i8(int8 v, UIntMaskType mask) { return __exclusive_scan_add<int8>(v, mask); }
EXT int16 __exclusive_scan_add_i16(int16 v, UIntMaskType mask) { return __exclusive_scan_add<int16>(v, mask); }
EXT int32 __exclusive_scan_add_i32(int32 v, UIntMaskType mask) { return __exclusive_scan_add<int32>(v, mask); }
EXT int64 __exclusive_scan_add_i64(int64 v, UIntMaskType mask) { return __exclusive_scan_add<int64>(v, mask); }
template <typename T> unmasked varying T __exclusive_scan_and(varying T v, UIntMaskType mask) {
    varying T res = -1;
    varying T val = mask ? v : -1;
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        res = __insert(res, i, __extract(val, i - 1) & __extract(res, i - 1));
    }
    return res;
}
EXT int8 __exclusive_scan_and_i8(int8 v, UIntMaskType mask) { return __exclusive_scan_and<int8>(v, mask); }
EXT int16 __exclusive_scan_and_i16(int16 v, UIntMaskType mask) { return __exclusive_scan_and<int16>(v, mask); }
EXT int32 __exclusive_scan_and_i32(int32 v, UIntMaskType mask) { return __exclusive_scan_and<int32>(v, mask); }
EXT int64 __exclusive_scan_and_i64(int64 v, UIntMaskType mask) { return __exclusive_scan_and<int64>(v, mask); }
template <typename T> unmasked varying T __exclusive_scan_or(varying T v, UIntMaskType mask) {
    varying T res = 0;
    varying T val = mask ? v : 0;
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        res = __insert(res, i, __extract(val, i - 1) | __extract(res, i - 1));
    }
    return res;
}
EXT int8 __exclusive_scan_or_i8(int8 v, UIntMaskType mask) { return __exclusive_scan_or<int8>(v, mask); }
EXT int16 __exclusive_scan_or_i16(int16 v, UIntMaskType mask) { return __exclusive_scan_or<int16>(v, mask); }
EXT int32 __exclusive_scan_or_i32(int32 v, UIntMaskType mask) { return __exclusive_scan_or<int32>(v, mask); }
EXT int64 __exclusive_scan_or_i64(int64 v, UIntMaskType mask) { return __exclusive_scan_or<int64>(v, mask); }

// rotate
template <typename T> unmasked varying T __rotate(varying T v, uniform int32 i) {
    uniform T<DOUBLE_WIDTH> dbl = __concat(v, v);
    uniform int32 start_pos = i & (TARGET_WIDTH - 1);
    varying T *uniform ptr = (varying T * uniform) & dbl[start_pos];
    return *ptr;
}
EXT READNONE varying int8 __rotate_i8(varying int8 v, uniform int32 i) { return __rotate<int8>(v, i); }
EXT READNONE varying int16 __rotate_i16(varying int16 v, uniform int32 i) { return __rotate<int16>(v, i); }
EXT READNONE varying int32 __rotate_i32(varying int32 v, uniform int32 i) { return __rotate<int32>(v, i); }
EXT READNONE varying int64 __rotate_i64(varying int64 v, uniform int32 i) { return __rotate<int64>(v, i); }
EXT READNONE varying float16 __rotate_half(varying float16 v, uniform int32 i) { return __rotate<float16>(v, i); }
EXT READNONE varying float __rotate_float(varying float v, uniform int32 i) { return __rotate<float>(v, i); }
EXT READNONE varying double __rotate_double(varying double v, uniform int32 i) { return __rotate<double>(v, i); }

// shift
template <typename T> unmasked varying T __shift(varying T v, uniform int32 i) {
    varying T<3> trp = {0, v, 0};
    uniform int32 start_pos = i + TARGET_WIDTH;
    uniform T *uniform ptr = (uniform T * uniform) & trp;
    varying T *uniform res_ptr = (varying T * uniform) & ptr[start_pos];
    return *res_ptr;
}
EXT READNONE varying int8 __shift_i8(varying int8 v, uniform int32 i) { return __shift<int8>(v, i); }
EXT READNONE varying int16 __shift_i16(varying int16 v, uniform int32 i) { return __shift<int16>(v, i); }
EXT READNONE varying int32 __shift_i32(varying int32 v, uniform int32 i) { return __shift<int32>(v, i); }
EXT READNONE varying int64 __shift_i64(varying int64 v, uniform int32 i) { return __shift<int64>(v, i); }
EXT READNONE varying float16 __shift_half(varying float16 v, uniform int32 i) { return __shift<float16>(v, i); }
EXT READNONE varying float __shift_float(varying float v, uniform int32 i) { return __shift<float>(v, i); }
EXT READNONE varying double __shift_double(varying double v, uniform int32 i) { return __shift<double>(v, i); }

// prefetching
// Constants have to be with suffixes l to avoid cast as prefetch arguments are immediate values
EXT void __prefetch_read_uniform_1(uniform int8 *uniform ptr) { @llvm.prefetch(ptr, 0l, 3l, 1l); }
EXT void __prefetch_read_uniform_2(uniform int8 *uniform ptr) { @llvm.prefetch(ptr, 0l, 2l, 1l); }
EXT void __prefetch_read_uniform_3(uniform int8 *uniform ptr) { @llvm.prefetch(ptr, 0l, 1l, 1l); }
EXT void __prefetch_read_uniform_nt(uniform int8 *uniform ptr) { @llvm.prefetch(ptr, 0l, 0l, 1l); }
EXT void __prefetch_write_uniform_1(uniform int8 *uniform ptr) { @llvm.prefetch(ptr, 1l, 3l, 1l); }
EXT void __prefetch_write_uniform_2(uniform int8 *uniform ptr) { @llvm.prefetch(ptr, 1l, 2l, 1l); }
EXT void __prefetch_write_uniform_3(uniform int8 *uniform ptr) { @llvm.prefetch(ptr, 1l, 1l, 1l); }
UNMANGLED CDECL void __prefetch_read_varying_1(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        __prefetch_read_uniform_1((uniform int8 * uniform) p_lane);
    }
}
UNMANGLED CDECL void __prefetch_read_varying_2(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        __prefetch_read_uniform_2((uniform int8 * uniform) p_lane);
    }
}
UNMANGLED CDECL void __prefetch_read_varying_3(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        __prefetch_read_uniform_3((uniform int8 * uniform) p_lane);
    }
}
UNMANGLED CDECL void __prefetch_read_varying_nt(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        __prefetch_read_uniform_nt((uniform int8 * uniform) p_lane);
    }
}
UNMANGLED CDECL void __prefetch_write_varying_1(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        __prefetch_write_uniform_1((uniform int8 * uniform) p_lane);
    }
}
UNMANGLED CDECL void __prefetch_write_varying_2(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        __prefetch_write_uniform_2((uniform int8 * uniform) p_lane);
    }
}
UNMANGLED CDECL void __prefetch_write_varying_3(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        __prefetch_write_uniform_3((uniform int8 * uniform) p_lane);
    }
}
UNMANGLED CDECL void __prefetch_write_varying_3_native(uniform int8 *uniform, uniform int32, varying int32) {
    __not_supported();
}

// shuffle
template <typename T> unmasked varying T __shuffle(varying T v, varying int32 perm) {
    varying T result = 0;
    for (uniform int32 i = 0; i < TARGET_WIDTH; i++) {
        uniform T lane_v = __extract(v, __extract(perm, i));
        result = __insert(result, i, lane_v);
    }
    return result;
}
EXT READNONE varying int8 __shuffle_i8(varying int8 v, varying int32 perm) { return __shuffle<int8>(v, perm); }
EXT READNONE varying int16 __shuffle_i16(varying int16 v, varying int32 perm) { return __shuffle<int16>(v, perm); }
EXT READNONE varying int32 __shuffle_i32(varying int32 v, varying int32 perm) { return __shuffle<int32>(v, perm); }
EXT READNONE varying int64 __shuffle_i64(varying int64 v, varying int32 perm) { return __shuffle<int64>(v, perm); }
EXT READNONE varying float16 __shuffle_half(varying float16 v, varying int32 perm) {
    return __shuffle<float16>(v, perm);
}
EXT READNONE varying float __shuffle_float(varying float v, varying int32 perm) { return __shuffle<float>(v, perm); }
EXT READNONE varying double __shuffle_double(varying double v, varying int32 perm) {
    return __shuffle<double>(v, perm);
}

// shuffle2
template <typename T> unmasked varying T __shuffle2(varying T v0, varying T v1, varying int32 perm) {
    uniform T<DOUBLE_WIDTH> v = __concat(v0, v1);
    varying T result = 0;
    for (uniform int32 i = 0; i < TARGET_WIDTH; i++) {
        uniform T lane_v = v[__extract(perm, i)];
        result = __insert(result, i, lane_v);
    }
    return result;
}
EXT READNONE varying int8 __shuffle2_i8(varying int8 v0, varying int8 v1, varying int32 perm) {
    return __shuffle2<int8>(v0, v1, perm);
}
EXT READNONE varying int16 __shuffle2_i16(varying int16 v0, varying int16 v1, varying int32 perm) {
    return __shuffle2<int16>(v0, v1, perm);
}
EXT READNONE varying int32 __shuffle2_i32(varying int32 v0, varying int32 v1, varying int32 perm) {
    return __shuffle2<int32>(v0, v1, perm);
}
EXT READNONE varying int64 __shuffle2_i64(varying int64 v0, varying int64 v1, varying int32 perm) {
    return __shuffle2<int64>(v0, v1, perm);
}
EXT READNONE varying float16 __shuffle2_half(varying float16 v0, varying float16 v1, varying int32 perm) {
    return __shuffle2<float16>(v0, v1, perm);
}
EXT READNONE varying float __shuffle2_float(varying float v0, varying float v1, varying int32 perm) {
    return __shuffle2<float>(v0, v1, perm);
}
EXT READNONE varying double __shuffle2_double(varying double v0, varying double v1, varying int32 perm) {
    return __shuffle2<double>(v0, v1, perm);
}

#define UNIX 1
#define WINDOWS 2

#ifndef BUILD_OS
#error "BUILD_OS not defined"
#endif

#ifndef RUNTIME
#error "RUNTIME not defined"
#endif

// new/delete
//
// Set of functions for 32 bit and 64 bit runtime.
// They are different for Windows and Unix (Linux/MacOS),
// on Windows we have to use _aligned_malloc/_aligned_free,
// while on Unix we use posix_memalign/free

#if BUILD_OS == UNIX
EXT uniform int32 posix_memalign(uniform int8 *uniform *uniform, uniform int32, uniform int32);
EXT void free(uniform int8 *uniform);
#endif // BUILD_OS == UNIX

#if BUILD_OS == WINDOWS
EXT uniform int8 *uniform _aligned_malloc(uniform int32, uniform int32);
EXT void _aligned_free(uniform int8 *uniform);
#endif // BUILD_OS == WINDOWS

// new/delete
#if BUILD_OS == UNIX && RUNTIME == 32
EXT uniform int8 *uniform __new_uniform_32rt(uniform int64 size) {
    uniform int8 *uniform ptr = 0;
    posix_memalign(&ptr, __memory_alignment, size);
    return ptr;
}
UNMANGLED CDECL varying int64 __new_varying32_32rt(varying int32 size) {
    varying int64 addr = 0;
    foreach_active(i) {
        uniform int64 s = __extract(size, i);
        uniform int8 *uniform ptr = 0;
        posix_memalign(&ptr, __memory_alignment, s);
        addr = __insert(addr, i, (uniform int64)ptr);
    }
    return addr;
}
EXT void __delete_uniform_32rt(uniform int8 *uniform ptr) { free(ptr); }
UNMANGLED CDECL void __delete_varying_32rt(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        free((uniform int8 * uniform) p_lane);
    }
}
#endif // BUILD_OS == UNIX && RUNTIME == 32

#if BUILD_OS == UNIX && RUNTIME == 64
EXT uniform int8 *uniform __new_uniform_64rt(uniform int64 size) {
    uniform int8 *uniform ptr = 0;
    posix_memalign(&ptr, __memory_alignment, size);
    return ptr;
}
UNMANGLED CDECL varying int64 __new_varying32_64rt(varying int32 size) {
    varying int64 addr = 0;
    foreach_active(i) {
        uniform int32 s = __extract(size, i);
        uniform int8 *uniform ptr = 0;
        posix_memalign(&ptr, __memory_alignment, s);
        addr = __insert(addr, i, (uniform int64)ptr);
    }
    return addr;
}
UNMANGLED CDECL varying int64 __new_varying64_64rt(varying int64 size) {
    varying int64 addr = 0;
    foreach_active(i) {
        uniform int64 s = __extract(size, i);
        uniform int8 *uniform ptr = 0;
        posix_memalign(&ptr, __memory_alignment, s);
        addr = __insert(addr, i, (uniform int64)ptr);
    }
    return addr;
}
EXT void __delete_uniform_64rt(uniform int8 *uniform ptr) { free(ptr); }
UNMANGLED CDECL void __delete_varying_64rt(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        free((uniform int8 * uniform) p_lane);
    }
}
#endif // BUILD_OS == UNIX && RUNTIME == 64

#if BUILD_OS == WINDOWS && RUNTIME == 32
EXT uniform int8 *uniform __new_uniform_32rt(uniform int64 size) { return _aligned_malloc(size, __memory_alignment); }
UNMANGLED CDECL varying int64 __new_varying32_32rt(varying int32 size) {
    varying int64 addr = 0;
    foreach_active(i) {
        uniform int64 s = __extract(size, i);
        uniform int8 *uniform ptr = _aligned_malloc(s, __memory_alignment);
        addr = __insert(addr, i, (uniform int64)ptr);
    }
    return addr;
}
EXT void __delete_uniform_32rt(uniform int8 *uniform ptr) { _aligned_free(ptr); }
UNMANGLED CDECL void __delete_varying_32rt(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        _aligned_free((uniform int8 * uniform) p_lane);
    }
}
#endif // BUILD_OS == WINDOWS && RUNTIME == 32

#if BUILD_OS == WINDOWS && RUNTIME == 64
EXT uniform int8 *uniform __new_uniform_64rt(uniform int64 size) { return _aligned_malloc(size, __memory_alignment); }
UNMANGLED CDECL varying int64 __new_varying32_64rt(varying int32 size) {
    varying int64 addr = 0;
    foreach_active(i) {
        uniform int32 s = __extract(size, i);
        uniform int8 *uniform ptr = _aligned_malloc(s, __memory_alignment);
        addr = __insert(addr, i, (uniform int64)ptr);
    }
    return addr;
}
UNMANGLED CDECL varying int64 __new_varying64_64rt(varying int64 size) {
    varying int64 addr = 0;
    foreach_active(i) {
        uniform int64 s = __extract(size, i);
        uniform int8 *uniform ptr = _aligned_malloc(s, __memory_alignment);
        addr = __insert(addr, i, (uniform int64)ptr);
    }
    return addr;
}
EXT void __delete_uniform_64rt(uniform int8 *uniform ptr) { _aligned_free(ptr); }
UNMANGLED CDECL void __delete_varying_64rt(varying int64 addr) {
    foreach_active(i) {
        uniform int64 p_lane = __extract(addr, i);
        _aligned_free((uniform int8 * uniform) p_lane);
    }
}
#endif // BUILD_OS == WINDOWS && RUNTIME == 64

// streaming stores
EXT void __streaming_store_uniform_i8(NOESCAPE uniform int8 *uniform ptr, uniform int8 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_uniform_i16(NOESCAPE uniform int8 *uniform ptr, uniform int16 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_uniform_i32(NOESCAPE uniform int8 *uniform ptr, uniform int32 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_uniform_i64(NOESCAPE uniform int8 *uniform ptr, uniform int64 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_uniform_half(NOESCAPE uniform int8 *uniform ptr, uniform float16 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_uniform_float(NOESCAPE uniform int8 *uniform ptr, uniform float val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_uniform_double(NOESCAPE uniform int8 *uniform ptr, uniform double val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_varying_i8(NOESCAPE uniform int8 *uniform ptr, varying int8 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_varying_i16(NOESCAPE uniform int8 *uniform ptr, varying int16 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_varying_i32(NOESCAPE uniform int8 *uniform ptr, varying int32 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_varying_i64(NOESCAPE uniform int8 *uniform ptr, varying int64 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_varying_half(NOESCAPE uniform int8 *uniform ptr, varying float16 val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_varying_float(NOESCAPE uniform int8 *uniform ptr, varying float val) {
    @llvm.ispc.stream_store(ptr, val);
}
EXT void __streaming_store_varying_double(NOESCAPE uniform int8 *uniform ptr, varying double val) {
    @llvm.ispc.stream_store(ptr, val);
}

// streaming loads
EXT uniform int8 __streaming_load_uniform_i8(NOESCAPE uniform int8 *uniform ptr) {
    uniform int8 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT uniform int16 __streaming_load_uniform_i16(NOESCAPE uniform int8 *uniform ptr) {
    uniform int16 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT uniform int32 __streaming_load_uniform_i32(NOESCAPE uniform int8 *uniform ptr) {
    uniform int32 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT uniform int64 __streaming_load_uniform_i64(NOESCAPE uniform int8 *uniform ptr) {
    uniform int64 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT uniform float16 __streaming_load_uniform_half(NOESCAPE uniform int8 *uniform ptr) {
    uniform float16 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT uniform float __streaming_load_uniform_float(NOESCAPE uniform int8 *uniform ptr) {
    uniform float dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT uniform double __streaming_load_uniform_double(NOESCAPE uniform int8 *uniform ptr) {
    uniform double dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT varying int8 __streaming_load_varying_i8(NOESCAPE uniform int8 *uniform ptr) {
    varying int8 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT varying int16 __streaming_load_varying_i16(NOESCAPE uniform int8 *uniform ptr) {
    varying int16 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT varying int32 __streaming_load_varying_i32(NOESCAPE uniform int8 *uniform ptr) {
    varying int32 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT varying int64 __streaming_load_varying_i64(NOESCAPE uniform int8 *uniform ptr) {
    varying int64 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT varying float16 __streaming_load_varying_half(NOESCAPE uniform int8 *uniform ptr) {
    varying float16 dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT varying float __streaming_load_varying_float(NOESCAPE uniform int8 *uniform ptr) {
    varying float dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}
EXT varying double __streaming_load_varying_double(NOESCAPE uniform int8 *uniform ptr) {
    varying double dummy;
    return @llvm.ispc.stream_load(ptr, dummy);
}

template <typename T> unmasked varying T __masked_load(varying T *uniform ptr, UIntMaskType mask) {
    // This function should never be called, instead specialized versions should be used.
    __not_supported();
}
template <> unmasked varying int8 __masked_load<int8>(varying int8 *uniform ptr, UIntMaskType mask) {
    varying int8 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load((uniform int8 * uniform) ptr, mask, passthru);
#else
    return @llvm.masked.load((uniform int8 * uniform) ptr, 1l, mask, passthru);
#endif
}
template <> unmasked varying int16 __masked_load<int16>(varying int16 *uniform ptr, UIntMaskType mask) {
    varying int16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load((uniform int8 * uniform) ptr, mask, passthru);
#else
    return @llvm.masked.load((uniform int8 * uniform) ptr, 1l, mask, passthru);
#endif
}
template <> unmasked varying int32 __masked_load<int32>(varying int32 *uniform ptr, UIntMaskType mask) {
    varying int32 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load((uniform int8 * uniform) ptr, mask, passthru);
#else
    return @llvm.masked.load((uniform int8 * uniform) ptr, 1l, mask, passthru);
#endif
}
template <> unmasked varying int64 __masked_load<int64>(varying int64 *uniform ptr, UIntMaskType mask) {
    varying int64 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load((uniform int8 * uniform) ptr, mask, passthru);
#else
    return @llvm.masked.load((uniform int8 * uniform) ptr, 1l, mask, passthru);
#endif
}
template <> unmasked varying float16 __masked_load<float16>(varying float16 *uniform ptr, UIntMaskType mask) {
    varying float16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load((uniform int8 * uniform) ptr, mask, passthru);
#else
    return @llvm.masked.load((uniform int8 * uniform) ptr, 1l, mask, passthru);
#endif
}
template <> unmasked varying float __masked_load<float>(varying float *uniform ptr, UIntMaskType mask) {
    varying float passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load((uniform int8 * uniform) ptr, mask, passthru);
#else
    return @llvm.masked.load((uniform int8 * uniform) ptr, 1l, mask, passthru);
#endif
}
template <> unmasked varying double __masked_load<double>(varying double *uniform ptr, UIntMaskType mask) {
    varying double passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.load((uniform int8 * uniform) ptr, mask, passthru);
#else
    return @llvm.masked.load((uniform int8 * uniform) ptr, 1l, mask, passthru);
#endif
}

#if TARGET_WIDTH == 64
#define MASK_HIGH_BIT_ON -9223372036854775808
#elif TARGET_WIDTH == 32
#define MASK_HIGH_BIT_ON 2147483648
#else
#define MASK_HIGH_BIT_ON (1ull << (TARGET_WIDTH - 1))
#endif

template <typename T> unmasked varying T __masked_load_fast(varying T *uniform ptr, UIntMaskType mask) {
    uniform uint64 mm = __movmsk(mask);
    uniform uint64 mm_and_low = mm & 1;
    uniform uint64 mm_and_high = mm & MASK_HIGH_BIT_ON;
    uniform uint64 mm_and_high_shift = mm_and_high >> (TARGET_WIDTH - 1);
    uniform bool can_vload = mm_and_low & mm_and_high_shift;
    uniform bool can_vload_maybe_fast = can_vload || (__fast_masked_vload != 0);
    if (can_vload_maybe_fast) {
        return *ptr;
    } else {
        return __masked_load<T>(ptr, mask);
    }
}

EXT varying int8 __masked_load_i8(varying int8 *uniform ptr, UIntMaskType mask) {
    return __masked_load_fast<int8>(ptr, mask);
}
EXT varying int16 __masked_load_i16(varying int16 *uniform ptr, UIntMaskType mask) {
    return __masked_load_fast<int16>(ptr, mask);
}
EXT varying int32 __masked_load_i32(varying int32 *uniform ptr, UIntMaskType mask) {
    return __masked_load_fast<int32>(ptr, mask);
}
EXT varying int64 __masked_load_i64(varying int64 *uniform ptr, UIntMaskType mask) {
    return __masked_load_fast<int64>(ptr, mask);
}
EXT varying float16 __masked_load_half(varying float16 *uniform ptr, UIntMaskType mask) {
    return __masked_load_fast<float16>(ptr, mask);
}
EXT varying float __masked_load_float(varying float *uniform ptr, UIntMaskType mask) {
    return __masked_load_fast<float>(ptr, mask);
}
EXT varying double __masked_load_double(varying double *uniform ptr, UIntMaskType mask) {
    return __masked_load_fast<double>(ptr, mask);
}

EXT void __masked_store_i8(NOESCAPE varying int8 *uniform ptr, varying int8 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, mask);
#else
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, 1l, mask);
#endif
}

EXT void __masked_store_i16(NOESCAPE varying int16 *uniform ptr, varying int16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, mask);
#else
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, 1l, mask);
#endif
}
EXT void __masked_store_i32(NOESCAPE varying int32 *uniform ptr, varying int32 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, mask);
#else
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, 1l, mask);
#endif
}
EXT void __masked_store_i64(NOESCAPE varying int64 *uniform ptr, varying int64 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, mask);
#else
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, 1l, mask);
#endif
}
EXT void __masked_store_half(NOESCAPE varying float16 *uniform ptr, varying float16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, mask);
#else
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, 1l, mask);
#endif
}
EXT void __masked_store_float(NOESCAPE varying float *uniform ptr, varying float val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, mask);
#else
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, 1l, mask);
#endif
}
EXT void __masked_store_double(NOESCAPE varying double *uniform ptr, varying double val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, mask);
#else
    @llvm.masked.store(val, (uniform int8 * uniform) ptr, 1l, mask);
#endif
}

EXT void __masked_store_blend_i8(NOESCAPE varying int8 *uniform ptr, varying int8 val, UIntMaskType mask) {
    @llvm.ispc.blend_store(val, (uniform int8 * uniform) ptr, mask);
}
EXT void __masked_store_blend_i16(NOESCAPE varying int16 *uniform ptr, varying int16 val, UIntMaskType mask) {
    @llvm.ispc.blend_store(val, (uniform int8 * uniform) ptr, mask);
}
EXT void __masked_store_blend_i32(NOESCAPE varying int32 *uniform ptr, varying int32 val, UIntMaskType mask) {
    @llvm.ispc.blend_store(val, (uniform int8 * uniform) ptr, mask);
}
EXT void __masked_store_blend_i64(NOESCAPE varying int64 *uniform ptr, varying int64 val, UIntMaskType mask) {
    @llvm.ispc.blend_store(val, (uniform int8 * uniform) ptr, mask);
}
EXT void __masked_store_blend_half(NOESCAPE varying float16 *uniform ptr, varying float16 val, UIntMaskType mask) {
    @llvm.ispc.blend_store(val, (uniform int8 * uniform) ptr, mask);
}
EXT void __masked_store_blend_float(NOESCAPE varying float *uniform ptr, varying float val, UIntMaskType mask) {
    @llvm.ispc.blend_store(val, (uniform int8 * uniform) ptr, mask);
}
EXT void __masked_store_blend_double(NOESCAPE varying double *uniform ptr, varying double val, UIntMaskType mask) {
    @llvm.ispc.blend_store(val, (uniform int8 * uniform) ptr, mask);
}

EXT uniform uint64 __cast_mask_to_i64(UIntMaskType mask) { return @llvm.ispc.packmask(mask); }

EXT uniform bool __wasm_cmp_msk_eq(UIntMaskType v1, UIntMaskType v2) {
    return __cast_mask_to_i64(v1) == __cast_mask_to_i64(v2);
}

// mask operations
EXT uniform int64 __movmsk(UIntMaskType mask) { return __cast_mask_to_i64(mask); }
EXT uniform bool __any(UIntMaskType mask) { return __cast_mask_to_i64(mask) != 0; }
EXT uniform bool __all(UIntMaskType mask) { return __cast_mask_to_i64(mask) == ~0ULL >> (64 - TARGET_WIDTH); }
EXT uniform bool __none(UIntMaskType mask) { return __cast_mask_to_i64(mask) == 0; }

// general gather functions
// 1l is needed for immarg of i32
EXT READONLY varying int8 __gather32_i8(varying uint32 addrs, UIntMaskType mask) {
    varying int8 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((varying uint64)addrs, mask, passthru);
#else
    return @llvm.masked.gather((varying uint64)addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying int8 __gather64_i8(varying uint64 addrs, UIntMaskType mask) {
    varying int8 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather(addrs, mask, passthru);
#else
    return @llvm.masked.gather(addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying int16 __gather32_i16(varying uint32 addrs, UIntMaskType mask) {
    varying int16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((varying uint64)addrs, mask, passthru);
#else
    return @llvm.masked.gather((varying uint64)addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying int16 __gather64_i16(varying uint64 addrs, UIntMaskType mask) {
    varying int16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather(addrs, mask, passthru);
#else
    return @llvm.masked.gather(addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying float16 __gather32_half(varying uint32 addrs, UIntMaskType mask) {
    varying float16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((varying uint64)addrs, mask, passthru);
#else
    return @llvm.masked.gather((varying uint64)addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying float16 __gather64_half(varying uint64 addrs, UIntMaskType mask) {
    varying float16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather(addrs, mask, passthru);
#else
    return @llvm.masked.gather(addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying int32 __gather32_i32(varying uint32 addrs, UIntMaskType mask) {
    varying int32 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((varying uint64)addrs, mask, passthru);
#else
    return @llvm.masked.gather((varying uint64)addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying int32 __gather64_i32(varying uint64 addrs, UIntMaskType mask) {
    varying int32 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather(addrs, mask, passthru);
#else
    return @llvm.masked.gather(addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying float __gather32_float(varying uint32 addrs, UIntMaskType mask) {
    varying float passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((varying uint64)addrs, mask, passthru);
#else
    return @llvm.masked.gather((varying uint64)addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying float __gather64_float(varying uint64 addrs, UIntMaskType mask) {
    varying float passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather(addrs, mask, passthru);
#else
    return @llvm.masked.gather(addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying int64 __gather32_i64(varying uint32 addrs, UIntMaskType mask) {
    varying int64 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((varying uint64)addrs, mask, passthru);
#else
    return @llvm.masked.gather((varying uint64)addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying int64 __gather64_i64(varying uint64 addrs, UIntMaskType mask) {
    varying int64 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather(addrs, mask, passthru);
#else
    return @llvm.masked.gather(addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying double __gather32_double(varying uint32 addrs, UIntMaskType mask) {
    varying double passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((varying uint64)addrs, mask, passthru);
#else
    return @llvm.masked.gather((varying uint64)addrs, 1l, mask, passthru);
#endif
}
EXT READONLY varying double __gather64_double(varying uint64 addrs, UIntMaskType mask) {
    varying double passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather(addrs, mask, passthru);
#else
    return @llvm.masked.gather(addrs, 1l, mask, passthru);
#endif
}

// generic gather functions
// for the targets which have gathers but --disable-gathers is requested
template <typename T, typename ADDRi, typename ADDRu> varying T __gather_generic(varying ADDRu addrs) {
    varying T res;
    foreach_active(i) {
        uniform ADDRi addr = (uniform ADDRi)__extract(addrs, i);
        uniform T *uniform ptr = (uniform T * uniform) addr;
        uniform T val = *ptr;
        res = __insert(res, i, val);
    }
    return res;
}
UNMANGLED CDECL READONLY varying int8 __gather32_generic_i8(varying uint32 addrs) {
    return __gather_generic<int8, int32, uint32>(addrs);
}
UNMANGLED CDECL READONLY varying int8 __gather64_generic_i8(varying uint64 addrs) {
    return __gather_generic<int8, int64, uint64>(addrs);
}
UNMANGLED CDECL READONLY varying int16 __gather32_generic_i16(varying uint32 addrs) {
    return __gather_generic<int16, int32, uint32>(addrs);
}
UNMANGLED CDECL READONLY varying int16 __gather64_generic_i16(varying uint64 addrs) {
    return __gather_generic<int16, int64, uint64>(addrs);
}
UNMANGLED CDECL READONLY varying float16 __gather32_generic_half(varying uint32 addrs) {
    return __gather_generic<float16, int32, uint32>(addrs);
}
UNMANGLED CDECL READONLY varying float16 __gather64_generic_half(varying uint64 addrs) {
    return __gather_generic<float16, int64, uint64>(addrs);
}
UNMANGLED CDECL READONLY varying int32 __gather32_generic_i32(varying uint32 addrs) {
    return __gather_generic<int32, int32, uint32>(addrs);
}
UNMANGLED CDECL READONLY varying int32 __gather64_generic_i32(varying uint64 addrs) {
    return __gather_generic<int32, int64, uint64>(addrs);
}
UNMANGLED CDECL READONLY varying float __gather32_generic_float(varying uint32 addrs) {
    return __gather_generic<float, int32, uint32>(addrs);
}
UNMANGLED CDECL READONLY varying float __gather64_generic_float(varying uint64 addrs) {
    return __gather_generic<float, int64, uint64>(addrs);
}
UNMANGLED CDECL READONLY varying int64 __gather32_generic_i64(varying uint32 addrs) {
    return __gather_generic<int64, int32, uint32>(addrs);
}
UNMANGLED CDECL READONLY varying int64 __gather64_generic_i64(varying uint64 addrs) {
    return __gather_generic<int64, int64, uint64>(addrs);
}
UNMANGLED CDECL READONLY varying double __gather32_generic_double(varying uint32 addrs) {
    return __gather_generic<double, int32, uint32>(addrs);
}
UNMANGLED CDECL READONLY varying double __gather64_generic_double(varying uint64 addrs) {
    return __gather_generic<double, int64, uint64>(addrs);
}

// gather functions with base offsets
EXT READONLY varying int8 __gather_base_offsets32_i8(uniform int8 *uniform ptr, uniform int32 scale,
                                                     varying int32 offsets, UIntMaskType mask) {
    varying int8 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying int8 __gather_base_offsets64_i8(uniform int8 *uniform ptr, uniform int32 scale,
                                                     varying int64 offsets, UIntMaskType mask) {
    varying int8 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying int16 __gather_base_offsets32_i16(uniform int8 *uniform ptr, uniform int32 scale,
                                                       varying int32 offsets, UIntMaskType mask) {
    varying int16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying int16 __gather_base_offsets64_i16(uniform int8 *uniform ptr, uniform int32 scale,
                                                       varying int64 offsets, UIntMaskType mask) {
    varying int16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying float16 __gather_base_offsets32_half(uniform int8 *uniform ptr, uniform int32 scale,
                                                          varying int32 offsets, UIntMaskType mask) {
    varying float16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying float16 __gather_base_offsets64_half(uniform int8 *uniform ptr, uniform int32 scale,
                                                          varying int64 offsets, UIntMaskType mask) {
    varying float16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying int32 __gather_base_offsets32_i32(uniform int8 *uniform ptr, uniform int32 scale,
                                                       varying int32 offsets, UIntMaskType mask) {
    varying int32 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying int32 __gather_base_offsets64_i32(uniform int8 *uniform ptr, uniform int32 scale,
                                                       varying int64 offsets, UIntMaskType mask) {
    varying int32 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying float __gather_base_offsets32_float(uniform int8 *uniform ptr, uniform int32 scale,
                                                         varying int32 offsets, UIntMaskType mask) {
    varying float passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying float __gather_base_offsets64_float(uniform int8 *uniform ptr, uniform int32 scale,
                                                         varying int64 offsets, UIntMaskType mask) {
    varying float passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying int64 __gather_base_offsets32_i64(uniform int8 *uniform ptr, uniform int32 scale,
                                                       varying int32 offsets, UIntMaskType mask) {
    varying int64 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying int64 __gather_base_offsets64_i64(uniform int8 *uniform ptr, uniform int32 scale,
                                                       varying int64 offsets, UIntMaskType mask) {
    varying int64 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying double __gather_base_offsets32_double(uniform int8 *uniform ptr, uniform int32 scale,
                                                           varying int32 offsets, UIntMaskType mask) {
    varying double passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}
EXT READONLY varying double __gather_base_offsets64_double(uniform int8 *uniform ptr, uniform int32 scale,
                                                           varying int64 offsets, UIntMaskType mask) {
    varying double passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)ptr + offsets * scale, 1l, mask, passthru);
#endif
}

// gather functions with factored base offsets
EXT READONLY varying int8 __gather_factored_base_offsets32_i8(uniform int8 *uniform base, varying int32 offsets,
                                                              uniform int32 scale, varying int32 delta,
                                                              UIntMaskType mask) {
    varying int8 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying int8 __gather_factored_base_offsets64_i8(uniform int8 *uniform base, varying int64 offsets,
                                                              uniform int32 scale, varying int64 delta,
                                                              UIntMaskType mask) {
    varying int8 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying int16 __gather_factored_base_offsets32_i16(uniform int8 *uniform base, varying int32 offsets,
                                                                uniform int32 scale, varying int32 delta,
                                                                UIntMaskType mask) {
    varying int16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying int16 __gather_factored_base_offsets64_i16(uniform int8 *uniform base, varying int64 offsets,
                                                                uniform int32 scale, varying int64 delta,
                                                                UIntMaskType mask) {
    varying int16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying float16 __gather_factored_base_offsets32_half(uniform int8 *uniform base, varying int32 offsets,
                                                                   uniform int32 scale, varying int32 delta,
                                                                   UIntMaskType mask) {
    varying float16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying float16 __gather_factored_base_offsets64_half(uniform int8 *uniform base, varying int64 offsets,
                                                                   uniform int32 scale, varying int64 delta,
                                                                   UIntMaskType mask) {
    varying float16 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying int32 __gather_factored_base_offsets32_i32(uniform int8 *uniform base, varying int32 offsets,
                                                                uniform int32 scale, varying int32 delta,
                                                                UIntMaskType mask) {
    varying int32 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying int32 __gather_factored_base_offsets64_i32(uniform int8 *uniform base, varying int64 offsets,
                                                                uniform int32 scale, varying int64 delta,
                                                                UIntMaskType mask) {
    varying int32 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying float __gather_factored_base_offsets32_float(uniform int8 *uniform base, varying int32 offsets,
                                                                  uniform int32 scale, varying int32 delta,
                                                                  UIntMaskType mask) {
    varying float passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying float __gather_factored_base_offsets64_float(uniform int8 *uniform base, varying int64 offsets,
                                                                  uniform int32 scale, varying int64 delta,
                                                                  UIntMaskType mask) {
    varying float passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying int64 __gather_factored_base_offsets32_i64(uniform int8 *uniform base, varying int32 offsets,
                                                                uniform int32 scale, varying int32 delta,
                                                                UIntMaskType mask) {
    varying int64 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying int64 __gather_factored_base_offsets64_i64(uniform int8 *uniform base, varying int64 offsets,
                                                                uniform int32 scale, varying int64 delta,
                                                                UIntMaskType mask) {
    varying int64 passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying double __gather_factored_base_offsets32_double(uniform int8 *uniform base, varying int32 offsets,
                                                                    uniform int32 scale, varying int32 delta,
                                                                    UIntMaskType mask) {
    varying double passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}
EXT READONLY varying double __gather_factored_base_offsets64_double(uniform int8 *uniform base, varying int64 offsets,
                                                                    uniform int32 scale, varying int64 delta,
                                                                    UIntMaskType mask) {
    varying double passthru;
#if LLVM_VERSION_MAJOR >= 22
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), mask, passthru);
#else
    return @llvm.masked.gather((uniform uint64)base + (offsets * scale + delta), 1l, mask, passthru);
#endif
}

// scatter generic functions
// For the targets which have scatters but --disable-scatters is requested
template <typename T, typename ADDRi, typename ADDRu> void __scatter_generic(varying ADDRu addrs, varying T val) {
    foreach_active(i) {
        uniform ADDRi addr = (uniform ADDRi)__extract(addrs, i);
        uniform T *uniform ptr = (uniform T * uniform) addr;
        *ptr = __extract(val, i);
    }
}
UNMANGLED CDECL void __scatter32_generic_i8(varying uint32 addr, varying int8 val) {
    __scatter_generic<int8, int32, uint32>(addr, val);
}
UNMANGLED CDECL void __scatter64_generic_i8(varying uint64 addr, varying int8 val) {
    __scatter_generic<int8, int64, uint64>(addr, val);
}
UNMANGLED CDECL void __scatter32_generic_i16(varying uint32 addr, varying int16 val) {
    __scatter_generic<int16, int32, uint32>(addr, val);
}
UNMANGLED CDECL void __scatter64_generic_i16(varying uint64 addr, varying int16 val) {
    __scatter_generic<int16, int64, uint64>(addr, val);
}
UNMANGLED CDECL void __scatter32_generic_half(varying uint32 addr, varying float16 val) {
    __scatter_generic<float16, int32, uint32>(addr, val);
}
UNMANGLED CDECL void __scatter64_generic_half(varying uint64 addr, varying float16 val) {
    __scatter_generic<float16, int64, uint64>(addr, val);
}
UNMANGLED CDECL void __scatter32_generic_i32(varying uint32 addr, varying int32 val) {
    __scatter_generic<int32, int32, uint32>(addr, val);
}
UNMANGLED CDECL void __scatter64_generic_i32(varying uint64 addr, varying int32 val) {
    __scatter_generic<int32, int64, uint64>(addr, val);
}
UNMANGLED CDECL void __scatter32_generic_float(varying uint32 addr, varying float val) {
    __scatter_generic<float, int32, uint32>(addr, val);
}
UNMANGLED CDECL void __scatter64_generic_float(varying uint64 addr, varying float val) {
    __scatter_generic<float, int64, uint64>(addr, val);
}
UNMANGLED CDECL void __scatter32_generic_i64(varying uint32 addr, varying int64 val) {
    __scatter_generic<int64, int32, uint32>(addr, val);
}
UNMANGLED CDECL void __scatter64_generic_i64(varying uint64 addr, varying int64 val) {
    __scatter_generic<int64, int64, uint64>(addr, val);
}
UNMANGLED CDECL void __scatter32_generic_double(varying uint32 addr, varying double val) {
    __scatter_generic<double, int32, uint32>(addr, val);
}
UNMANGLED CDECL void __scatter64_generic_double(varying uint64 addr, varying double val) {
    __scatter_generic<double, int64, uint64>(addr, val);
}

// general scatters
// 1l is needed because argument is immarg of i32
EXT void __scatter32_i8(varying uint32 addr, varying int8 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (varying uint64)addr, mask);
#else
    @llvm.masked.scatter(val, (varying uint64)addr, 1l, mask);
#endif
}
EXT void __scatter64_i8(varying uint64 addr, varying int8 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, addr, mask);
#else
    @llvm.masked.scatter(val, addr, 1l, mask);
#endif
}
EXT void __scatter32_i16(varying uint32 addr, varying int16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (varying uint64)addr, mask);
#else
    @llvm.masked.scatter(val, (varying uint64)addr, 1l, mask);
#endif
}
EXT void __scatter64_i16(varying uint64 addr, varying int16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, addr, mask);
#else
    @llvm.masked.scatter(val, addr, 1l, mask);
#endif
}
EXT void __scatter32_half(varying uint32 addr, varying float16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (varying uint64)addr, mask);
#else
    @llvm.masked.scatter(val, (varying uint64)addr, 1l, mask);
#endif
}
EXT void __scatter64_half(varying uint64 addr, varying float16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, addr, mask);
#else
    @llvm.masked.scatter(val, addr, 1l, mask);
#endif
}
EXT void __scatter32_i32(varying uint32 addr, varying int32 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (varying uint64)addr, mask);
#else
    @llvm.masked.scatter(val, (varying uint64)addr, 1l, mask);
#endif
}
EXT void __scatter64_i32(varying uint64 addr, varying int32 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, addr, mask);
#else
    @llvm.masked.scatter(val, addr, 1l, mask);
#endif
}
EXT void __scatter32_float(varying uint32 addr, varying float val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (varying uint64)addr, mask);
#else
    @llvm.masked.scatter(val, (varying uint64)addr, 1l, mask);
#endif
}
EXT void __scatter64_float(varying uint64 addr, varying float val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, addr, mask);
#else
    @llvm.masked.scatter(val, addr, 1l, mask);
#endif
}
EXT void __scatter32_i64(varying uint32 addr, varying int64 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (varying uint64)addr, mask);
#else
    @llvm.masked.scatter(val, (varying uint64)addr, 1l, mask);
#endif
}
EXT void __scatter64_i64(varying uint64 addr, varying int64 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, addr, mask);
#else
    @llvm.masked.scatter(val, addr, 1l, mask);
#endif
}
EXT void __scatter32_double(varying uint32 addr, varying double val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (varying uint64)addr, mask);
#else
    @llvm.masked.scatter(val, (varying uint64)addr, 1l, mask);
#endif
}
EXT void __scatter64_double(varying uint64 addr, varying double val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, addr, mask);
#else
    @llvm.masked.scatter(val, addr, 1l, mask);
#endif
}

// scatter functions with base offsets
EXT void __scatter_base_offsets32_i8(uniform int8 *uniform base, uniform int32 scale, varying int32 offset,
                                     varying int8 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets64_i8(uniform int8 *uniform base, uniform int32 scale, varying int64 offset,
                                     varying int8 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets32_i16(uniform int8 *uniform base, uniform int32 scale, varying int32 offset,
                                      varying int16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets64_i16(uniform int8 *uniform base, uniform int32 scale, varying int64 offset,
                                      varying int16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets32_half(uniform int8 *uniform base, uniform int32 scale, varying int32 offset,
                                       varying float16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets64_half(uniform int8 *uniform base, uniform int32 scale, varying int64 offset,
                                       varying float16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets32_i32(uniform int8 *uniform base, uniform int32 scale, varying int32 offset,
                                      varying int32 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets64_i32(uniform int8 *uniform base, uniform int32 scale, varying int64 offset,
                                      varying int32 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets32_float(uniform int8 *uniform base, uniform int32 scale, varying int32 offset,
                                        varying float val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets64_float(uniform int8 *uniform base, uniform int32 scale, varying int64 offset,
                                        varying float val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets32_i64(uniform int8 *uniform base, uniform int32 scale, varying int32 offset,
                                      varying int64 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets64_i64(uniform int8 *uniform base, uniform int32 scale, varying int64 offset,
                                      varying int64 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets32_double(uniform int8 *uniform base, uniform int32 scale, varying int32 offset,
                                         varying double val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}
EXT void __scatter_base_offsets64_double(uniform int8 *uniform base, uniform int32 scale, varying int64 offset,
                                         varying double val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + offset * scale, 1l, mask);
#endif
}

// scatters with factored base offsets
EXT void __scatter_factored_base_offsets32_i8(uniform int8 *uniform base, varying int32 offset, uniform int32 scale,
                                              varying int32 delta, varying int8 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets64_i8(uniform int8 *uniform base, varying int64 offset, uniform int32 scale,
                                              varying int64 delta, varying int8 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets32_i16(uniform int8 *uniform base, varying int32 offset, uniform int32 scale,
                                               varying int32 delta, varying int16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets64_i16(uniform int8 *uniform base, varying int64 offset, uniform int32 scale,
                                               varying int64 delta, varying int16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets32_half(uniform int8 *uniform base, varying int32 offset, uniform int32 scale,
                                                varying int32 delta, varying float16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets64_half(uniform int8 *uniform base, varying int64 offset, uniform int32 scale,
                                                varying int64 delta, varying float16 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets32_i32(uniform int8 *uniform base, varying int32 offset, uniform int32 scale,
                                               varying int32 delta, varying int32 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets64_i32(uniform int8 *uniform base, varying int64 offset, uniform int32 scale,
                                               varying int64 delta, varying int32 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets32_float(uniform int8 *uniform base, varying int32 offset, uniform int32 scale,
                                                 varying int32 delta, varying float val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets64_float(uniform int8 *uniform base, varying int64 offset, uniform int32 scale,
                                                 varying int64 delta, varying float val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets32_i64(uniform int8 *uniform base, varying int32 offset, uniform int32 scale,
                                               varying int32 delta, varying int64 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets64_i64(uniform int8 *uniform base, varying int64 offset, uniform int32 scale,
                                               varying int64 delta, varying int64 val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets32_double(uniform int8 *uniform base, varying int32 offset, uniform int32 scale,
                                                  varying int32 delta, varying double val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}
EXT void __scatter_factored_base_offsets64_double(uniform int8 *uniform base, varying int64 offset, uniform int32 scale,
                                                  varying int64 delta, varying double val, UIntMaskType mask) {
#if LLVM_VERSION_MAJOR >= 22
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), mask);
#else
    @llvm.masked.scatter(val, (uniform uint64)base + (offset * scale + delta), 1l, mask);
#endif
}

// packed active stores
EXT PackedStoreResultType __packed_store_activei8(uniform int8 *uniform ptr, varying int8 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_activei16(uniform int8 *uniform ptr, varying int16 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_activei32(uniform int8 *uniform ptr, varying int32 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_activei64(uniform int8 *uniform ptr, varying int64 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_activef16(uniform int8 *uniform ptr, varying float16 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_activef32(uniform int8 *uniform ptr, varying float val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_activef64(uniform int8 *uniform ptr, varying double val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_active2i8(uniform int8 *uniform ptr, varying int8 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_active2i16(uniform int8 *uniform ptr, varying int16 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_active2i32(uniform int8 *uniform ptr, varying int32 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_active2i64(uniform int8 *uniform ptr, varying int64 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_active2f16(uniform int8 *uniform ptr, varying float16 val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_active2f32(uniform int8 *uniform ptr, varying float val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT PackedStoreResultType __packed_store_active2f64(uniform int8 *uniform ptr, varying double val, UIntMaskType mask) {
    @llvm.masked.compressstore(val, ptr, mask);
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
// packed active loads
template <typename T>
uniform int32 __packed_load_active(uniform int8 *uniform from, uniform int8 *uniform to, UIntMaskType mask) {
    varying T *uniform ptr = (varying T * uniform) to;
    varying T passthru = *ptr;
    varying T vec = __masked_expandload(from, mask, passthru);
    *ptr = vec;
    return __popcnt_int64_uniform(__cast_mask_to_i64(mask));
}
EXT uniform int32 __packed_load_activei8(uniform int8 *uniform from, uniform int8 *uniform to, UIntMaskType mask) {
    return __packed_load_active<int8>(from, to, mask);
}
EXT uniform int32 __packed_load_activei16(uniform int8 *uniform from, uniform int8 *uniform to, UIntMaskType mask) {
    return __packed_load_active<int16>(from, to, mask);
}
EXT uniform int32 __packed_load_activei32(uniform int8 *uniform from, uniform int8 *uniform to, UIntMaskType mask) {
    return __packed_load_active<int32>(from, to, mask);
}
EXT uniform int32 __packed_load_activei64(uniform int8 *uniform from, uniform int8 *uniform to, UIntMaskType mask) {
    return __packed_load_active<int64>(from, to, mask);
}
EXT uniform int32 __packed_load_activef16(uniform int8 *uniform from, uniform int8 *uniform to, UIntMaskType mask) {
    return __packed_load_active<float16>(from, to, mask);
}
EXT uniform int32 __packed_load_activef32(uniform int8 *uniform from, uniform int8 *uniform to, UIntMaskType mask) {
    return __packed_load_active<float>(from, to, mask);
}
EXT uniform int32 __packed_load_activef64(uniform int8 *uniform from, uniform int8 *uniform to, UIntMaskType mask) {
    return __packed_load_active<double>(from, to, mask);
}

template <typename T>
unmasked void __aos_to_soa2(NOALIAS uniform int8 *uniform a, uniform int8 *uniform v0, uniform int8 *uniform v1) {
    // E.g., given 2 4-vectors. Convert 2-wide AOS values to SOA--specifically
    // <x0 y0 x1 y1> <x2 y2 x3 y3>, transpose to
    // <x0 x1 x2 x3> <y0 y1 y2 y3>.
    uniform T *uniform ptr = (uniform T * uniform) a;
    varying T x = 0, y = 0;
    for (uniform int i = 0; i < TARGET_WIDTH; i++) {
        x = __insert(x, i, ptr[0 + i * 2]);
        y = __insert(y, i, ptr[1 + i * 2]);
    }
    varying T *uniform x_ptr = (varying T * uniform) v0;
    varying T *uniform y_ptr = (varying T * uniform) v1;
    *x_ptr = x;
    *y_ptr = y;
}

EXT void __aos_to_soa2_float(NOALIAS uniform int8 *uniform a, uniform int8 *uniform v0, uniform int8 *uniform v1) {
    __aos_to_soa2<float>(a, v0, v1);
}
EXT void __aos_to_soa2_double(NOALIAS uniform int8 *uniform a, uniform int8 *uniform v0, uniform int8 *uniform v1) {
    __aos_to_soa2<double>(a, v0, v1);
}

template <typename T> unmasked void __soa_to_aos2(varying T v0, varying T v1, NOALIAS uniform int8 *uniform a) {
    // E.g., given 2 4-vectors. The inverse of __aos_to_soa3_float4: convert 2 4-vectors
    // <x0 x1 x2 x3> <y0 y1 y2 y3>, to
    // <x0 y0 x1 y1 x2 y2 x3 y3>.
    uniform T *uniform x = (uniform T * uniform) a;
    for (uniform int j = 0; j < TARGET_WIDTH; j++) {
        for (uniform int i = 0; i < 2; i++) {
            x[2 * j + 0] = __extract(v0, j);
            x[2 * j + 1] = __extract(v1, j);
        }
    }
}

EXT void __soa_to_aos2_float(varying float v0, varying float v1, NOALIAS uniform int8 *uniform a) {
    __soa_to_aos2<float>(v0, v1, a);
}

EXT void __soa_to_aos2_double(varying double v0, varying double v1, NOALIAS uniform int8 *uniform a) {
    __soa_to_aos2<double>(v0, v1, a);
}

template <typename T>
unmasked void __aos_to_soa4(NOALIAS uniform int8 *uniform p, NOALIAS uniform int8 *uniform v0,
                            NOALIAS uniform int8 *uniform v1, NOALIAS uniform int8 *uniform v2,
                            NOALIAS uniform int8 *uniform v3) {
    // For example, for 4-wide vectors:
    // take 4 4-wide vectors laid out like <r0 g0 b0 a0> <r1 g1 b1 a1> <r2 g2 b2 a2> <r3 g3 b3 a3>
    // and reorder them to <r0 r1 r2 r3> <g0 g1 g2 g3> ...
    // Deinterleave the 4 4-wide vectors into 2 8-wide vectors
    // { <r0 b0 r1 b1 r2 b2 r3 b3>, <g0 a0 g1 a1 g2 a2 g3 a3> }
    uniform T *uniform ptr = (uniform T * uniform) p;
    varying T r = 0, g = 0, b = 0, a = 0;
    for (uniform int i = 0; i < TARGET_WIDTH; i++) {
        r = __insert(r, i, ptr[0 + i * 4]);
        g = __insert(g, i, ptr[1 + i * 4]);
        b = __insert(b, i, ptr[2 + i * 4]);
        a = __insert(a, i, ptr[3 + i * 4]);
    }
    varying T *uniform r_ptr = (varying T * uniform) v0;
    varying T *uniform g_ptr = (varying T * uniform) v1;
    varying T *uniform b_ptr = (varying T * uniform) v2;
    varying T *uniform a_ptr = (varying T * uniform) v3;
    *r_ptr = r;
    *g_ptr = g;
    *b_ptr = b;
    *a_ptr = a;
}

EXT void __aos_to_soa4_float(NOALIAS uniform int8 *uniform a, NOALIAS uniform int8 *uniform v0,
                             NOALIAS uniform int8 *uniform v1, NOALIAS uniform int8 *uniform v2,
                             NOALIAS uniform int8 *uniform v3) {
    __aos_to_soa4<float>(a, v0, v1, v2, v3);
}

EXT void __aos_to_soa4_double(NOALIAS uniform int8 *uniform a, NOALIAS uniform int8 *uniform v0,
                              NOALIAS uniform int8 *uniform v1, NOALIAS uniform int8 *uniform v2,
                              NOALIAS uniform int8 *uniform v3) {
    __aos_to_soa4<double>(a, v0, v1, v2, v3);
}

template <typename T>
unmasked void __soa_to_aos4(varying T v0, varying T v1, varying T v2, varying T v3, NOALIAS uniform int8 *uniform a) {
    // For example, for 4-wide vectors:
    // take 4 4-wide vectors laid out like <r0 r1 r2 r3> <g0 g1 g2 g3> <b0 b1 b2 b3> <a0 a1 a2 a3>
    // and reorder them to 4 4-wide vectors laid out like <r0 g0 b0 a0> <r1 g1 b1 a1> <r2 g2 b2 a2> <r3 g3 b3 a3>
    uniform T *uniform x = (uniform T * uniform) a;
    for (uniform int j = 0; j < TARGET_WIDTH; j++) {
        for (uniform int i = 0; i < 4; i++) {
            x[4 * j + 0] = __extract(v0, j);
            x[4 * j + 1] = __extract(v1, j);
            x[4 * j + 2] = __extract(v2, j);
            x[4 * j + 3] = __extract(v3, j);
        }
    }
}

EXT void __soa_to_aos4_float(varying float v0, varying float v1, varying float v2, varying float v3,
                             NOALIAS uniform int8 *uniform a) {
    __soa_to_aos4<float>(v0, v1, v2, v3, a);
}

EXT void __soa_to_aos4_double(varying double v0, varying double v1, varying double v2, varying double v3,
                              NOALIAS uniform int8 *uniform a) {
    __soa_to_aos4<double>(v0, v1, v2, v3, a);
}

template <typename T>
unmasked void __aos_to_soa3(NOALIAS uniform int8 *uniform a, uniform int8 *uniform v0, uniform int8 *uniform v1,
                            uniform int8 *uniform v2) {
    // For example, for 4-wide vectors:
    // Convert 3-wide AOS values to SOA--specifically, given 3 4-vectors
    // <x0 y0 z0 x1> <y1 z1 x2 y2> <z2 x3 y3 z3>, transpose to
    // <x0 x1 x2 x3> <y0 y1 y2 y3> <z0 z1 z2 z3>.
    uniform T *uniform ptr = (uniform T * uniform) a;
    varying T x = 0, y = 0, z = 0;
    for (uniform int i = 0; i < TARGET_WIDTH; i++) {
        x = __insert(x, i, ptr[0 + i * 3]);
        y = __insert(y, i, ptr[1 + i * 3]);
        z = __insert(z, i, ptr[2 + i * 3]);
    }
    varying T *uniform x_ptr = (varying T * uniform) v0;
    varying T *uniform y_ptr = (varying T * uniform) v1;
    varying T *uniform z_ptr = (varying T * uniform) v2;
    *x_ptr = x;
    *y_ptr = y;
    *z_ptr = z;
}

EXT void __aos_to_soa3_float(NOALIAS uniform int8 *uniform a, uniform int8 *uniform v0, uniform int8 *uniform v1,
                             uniform int8 *uniform v2) {
    __aos_to_soa3<float>(a, v0, v1, v2);
}

EXT void __aos_to_soa3_double(NOALIAS uniform int8 *uniform a, uniform int8 *uniform v0, uniform int8 *uniform v1,
                              uniform int8 *uniform v2) {
    __aos_to_soa3<double>(a, v0, v1, v2);
}

template <typename T>
unmasked void __soa_to_aos3(varying T v0, varying T v1, varying T v2, NOALIAS uniform int8 *uniform a) {
    // For example, for 4-wide vectors:
    // The inverse of __aos_to_soa3_T4: convert 3 4-vectors
    // <x0 x1 x2 x3> <y0 y1 y2 y3> <z0 z1 z2 z3> to
    // <x0 y0 z0 x1> <y1 z1 x2 y2> <z2 x3 y3 z3>.
    uniform T *uniform x = (uniform T * uniform) a;
    for (uniform int j = 0; j < TARGET_WIDTH; j++) {
        for (uniform int i = 0; i < 3; i++) {
            x[3 * j + 0] = __extract(v0, j);
            x[3 * j + 1] = __extract(v1, j);
            x[3 * j + 2] = __extract(v2, j);
        }
    }
}

EXT void __soa_to_aos3_float(float v0, float v1, float v2, NOALIAS uniform int8 *uniform a) {
    __soa_to_aos3<float>(v0, v1, v2, a);
}
EXT void __soa_to_aos3_double(double v0, double v1, double v2, NOALIAS uniform int8 *uniform a) {
    __soa_to_aos3<double>(v0, v1, v2, a);
}

EXT uniform double __atomic_swap_uniform_double_global(uniform int8 *uniform ptr, uniform double val) {
    return @llvm.ispc.atomicrmw.xchg.seq_cst(ptr, val);
}
EXT uniform float __atomic_swap_uniform_float_global(uniform int8 *uniform ptr, uniform float val) {
    return @llvm.ispc.atomicrmw.xchg.seq_cst(ptr, val);
}
EXT uniform int32 __atomic_swap_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 val) {
    return @llvm.ispc.atomicrmw.xchg.seq_cst(ptr, val);
}
EXT uniform int64 __atomic_swap_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 val) {
    return @llvm.ispc.atomicrmw.xchg.seq_cst(ptr, val);
}
EXT uniform int32 __atomic_add_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 val) {
    return @llvm.ispc.atomicrmw.add.seq_cst(ptr, val);
}
EXT uniform int64 __atomic_add_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 val) {
    return @llvm.ispc.atomicrmw.add.seq_cst(ptr, val);
}
EXT uniform int32 __atomic_sub_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 val) {
    return @llvm.ispc.atomicrmw.sub.seq_cst(ptr, val);
}
EXT uniform int64 __atomic_sub_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 val) {
    return @llvm.ispc.atomicrmw.sub.seq_cst(ptr, val);
}
EXT uniform int32 __atomic_and_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 val) {
    return @llvm.ispc.atomicrmw.and.seq_cst(ptr, val);
}
EXT uniform int64 __atomic_and_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 val) {
    return @llvm.ispc.atomicrmw.and.seq_cst(ptr, val);
}
EXT uniform int32 __atomic_or_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 val) {
    return @llvm.ispc.atomicrmw.or.seq_cst(ptr, val);
}
EXT uniform int64 __atomic_or_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 val) {
    return @llvm.ispc.atomicrmw.or.seq_cst(ptr, val);
}
EXT uniform int32 __atomic_xor_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 val) {
    return @llvm.ispc.atomicrmw.xor.seq_cst(ptr, val);
}
EXT uniform int64 __atomic_xor_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 val) {
    return @llvm.ispc.atomicrmw.xor.seq_cst(ptr, val);
}
EXT uniform int32 __atomic_min_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 val) {
    return @llvm.ispc.atomicrmw.min.seq_cst(ptr, val);
}
EXT uniform int64 __atomic_min_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 val) {
    return @llvm.ispc.atomicrmw.min.seq_cst(ptr, val);
}
EXT uniform int32 __atomic_max_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 val) {
    return @llvm.ispc.atomicrmw.max.seq_cst(ptr, val);
}
EXT uniform int64 __atomic_max_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 val) {
    return @llvm.ispc.atomicrmw.max.seq_cst(ptr, val);
}
EXT uniform uint32 __atomic_umin_uniform_uint32_global(uniform int8 *uniform ptr, uniform uint32 val) {
    return @llvm.ispc.atomicrmw.umin.seq_cst(ptr, val);
}
EXT uniform uint64 __atomic_umin_uniform_uint64_global(uniform int8 *uniform ptr, uniform uint64 val) {
    return @llvm.ispc.atomicrmw.umin.seq_cst(ptr, val);
}
EXT uniform uint32 __atomic_umax_uniform_uint32_global(uniform int8 *uniform ptr, uniform uint32 val) {
    return @llvm.ispc.atomicrmw.umax.seq_cst(ptr, val);
}
EXT uniform uint64 __atomic_umax_uniform_uint64_global(uniform int8 *uniform ptr, uniform uint64 val) {
    return @llvm.ispc.atomicrmw.umax.seq_cst(ptr, val);
}
EXT uniform double __atomic_fadd_uniform_double_global(uniform int8 *uniform ptr, uniform double val) {
    return @llvm.ispc.atomicrmw.fadd.seq_cst(ptr, val);
}
EXT uniform float __atomic_fadd_uniform_float_global(uniform int8 *uniform ptr, uniform float val) {
    return @llvm.ispc.atomicrmw.fadd.seq_cst(ptr, val);
}
EXT uniform double __atomic_fsub_uniform_double_global(uniform int8 *uniform ptr, uniform double val) {
    return @llvm.ispc.atomicrmw.fsub.seq_cst(ptr, val);
}
EXT uniform float __atomic_fsub_uniform_float_global(uniform int8 *uniform ptr, uniform float val) {
    return @llvm.ispc.atomicrmw.fsub.seq_cst(ptr, val);
}
EXT uniform double __atomic_fmin_uniform_double_global(uniform int8 *uniform ptr, uniform double val) {
    return @llvm.ispc.atomicrmw.fmin.seq_cst(ptr, val);
}
EXT uniform float __atomic_fmin_uniform_float_global(uniform int8 *uniform ptr, uniform float val) {
    return @llvm.ispc.atomicrmw.fmin.seq_cst(ptr, val);
}
EXT uniform double __atomic_fmax_uniform_double_global(uniform int8 *uniform ptr, uniform double val) {
    return @llvm.ispc.atomicrmw.fmax.seq_cst(ptr, val);
}
EXT uniform float __atomic_fmax_uniform_float_global(uniform int8 *uniform ptr, uniform float val) {
    return @llvm.ispc.atomicrmw.fmax.seq_cst(ptr, val);
}

UNMANGLED CDECL varying double __atomic_fadd_double_global(uniform int8 *uniform ptr, varying double val) {
    varying double res;
    foreach_active(i) {
        uniform double r = __atomic_fadd_uniform_double_global(ptr, __extract(val, i));
        res = __insert(res, i, r);
    }
    return res;
}
UNMANGLED CDECL varying float __atomic_fadd_float_global(uniform int8 *uniform ptr, varying float val) {
    varying float res;
    foreach_active(i) {
        uniform float r = __atomic_fadd_uniform_float_global(ptr, __extract(val, i));
        res = __insert(res, i, r);
    }
    return res;
}
UNMANGLED CDECL varying double __atomic_fsub_double_global(uniform int8 *uniform ptr, varying double val) {
    varying double res;
    foreach_active(i) {
        uniform double r = __atomic_fsub_uniform_double_global(ptr, __extract(val, i));
        res = __insert(res, i, r);
    }
    return res;
}
UNMANGLED CDECL varying float __atomic_fsub_float_global(uniform int8 *uniform ptr, varying float val) {
    varying float res;
    foreach_active(i) {
        uniform float r = __atomic_fsub_uniform_float_global(ptr, __extract(val, i));
        res = __insert(res, i, r);
    }
    return res;
}

// More efficient implementation for atomics that are associative (e.g.,
// add, and, ...).  If a basic implementation would do sometihng like:
// result0 = atomic_op(ptr, val0)
// result1 = atomic_op(ptr, val1)
// ..
// Then instead we can do:
// tmp = (val0 op val1 op ...)
// result0 = atomic_op(ptr, tmp)
// result1 = (result0 op val0)
// ..
// And more efficiently compute the same result

EXT varying int32 __atomic_add_int32_global(uniform int8 *uniform ptr, varying int32 val, UIntMaskType mask) {
    // v has identity in the off lanes and the value in the on lanes
    varying int32 elt = 0;
    varying int32 v = @llvm.ispc.select((bool)mask, val, elt);
    // now compute the local reduction (v0 op v1 op ... )--initialize
    // elt so that the 0th element is the identity, the first is v0,
    // the second is (v0 op v1), ..
    uniform int32 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int32 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red + ei;
    }
    uniform int32 final = @llvm.ispc.atomicrmw.add.seq_cst(ptr, red);
    // now go back and compute the values to be returned for each program
    // instance--this just involves smearing the old value returned from the
    // actual atomic call across the vector and applying the vector op to the
    // elt vector computed above..
    return elt + final;
}
EXT varying int64 __atomic_add_int64_global(uniform int8 *uniform ptr, varying int64 val, UIntMaskType mask) {
    varying int64 elt = 0;
    varying int64 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int64 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int64 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red + ei;
    }
    uniform int64 final = @llvm.ispc.atomicrmw.add.seq_cst(ptr, red);
    return elt + final;
}
EXT varying int32 __atomic_sub_int32_global(uniform int8 *uniform ptr, varying int32 val, UIntMaskType mask) {
    varying int32 elt = 0;
    varying int32 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int32 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int32 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red - ei;
    }
    uniform int32 final = @llvm.ispc.atomicrmw.sub.seq_cst(ptr, red);
    return elt - final;
}
EXT varying int64 __atomic_sub_int64_global(uniform int8 *uniform ptr, varying int64 val, UIntMaskType mask) {
    varying int64 elt = 0;
    varying int64 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int64 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int64 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red - ei;
    }
    uniform int64 final = @llvm.ispc.atomicrmw.sub.seq_cst(ptr, red);
    return elt + final;
}
EXT varying int32 __atomic_and_int32_global(uniform int8 *uniform ptr, varying int32 val, UIntMaskType mask) {
    varying int32 elt = -1;
    varying int32 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int32 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int32 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red & ei;
    }
    uniform int32 final = @llvm.ispc.atomicrmw.sub.seq_cst(ptr, red);
    return elt & final;
}
EXT varying int64 __atomic_and_int64_global(uniform int8 *uniform ptr, varying int64 val, UIntMaskType mask) {
    varying int64 elt = -1;
    varying int64 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int64 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int64 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red & ei;
    }
    uniform int64 final = @llvm.ispc.atomicrmw.sub.seq_cst(ptr, red);
    return elt & final;
}
EXT varying int32 __atomic_or_int32_global(uniform int8 *uniform ptr, varying int32 val, UIntMaskType mask) {
    varying int32 elt = 0;
    varying int32 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int32 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int32 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red | ei;
    }
    uniform int32 final = @llvm.ispc.atomicrmw.or.seq_cst(ptr, red);
    return elt | final;
}
EXT varying int64 __atomic_or_int64_global(uniform int8 *uniform ptr, varying int64 val, UIntMaskType mask) {
    varying int64 elt = 0;
    varying int64 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int64 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int64 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red | ei;
    }
    uniform int64 final = @llvm.ispc.atomicrmw.or.seq_cst(ptr, red);
    return elt | final;
}
EXT varying int32 __atomic_xor_int32_global(uniform int8 *uniform ptr, varying int32 val, UIntMaskType mask) {
    varying int32 elt = 0;
    varying int32 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int32 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int32 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red ^ ei;
    }
    uniform int32 final = @llvm.ispc.atomicrmw.xor.seq_cst(ptr, red);
    return elt ^ final;
}
EXT varying int64 __atomic_xor_int64_global(uniform int8 *uniform ptr, varying int64 val, UIntMaskType mask) {
    varying int64 elt = 0;
    varying int64 v = @llvm.ispc.select((bool)mask, val, elt);
    uniform int64 red = __extract(v, 0);
    for (uniform int32 i = 1; i < TARGET_WIDTH; i++) {
        uniform int64 ei = __extract(v, i);
        elt = __insert(elt, i, red);
        red = red ^ ei;
    }
    uniform int64 final = @llvm.ispc.atomicrmw.xor.seq_cst(ptr, red);
    return elt ^ final;
}

unmasked uniform int32 __cmpxchg(uniform int8 *uniform ptr, uniform int32 cmp, uniform int32 val) {
    return @llvm.ispc.cmpxchg.seq_cst.seq_cst(ptr, cmp, val);
}
unmasked uniform int64 __cmpxchg(uniform int8 *uniform ptr, uniform int64 cmp, uniform int64 val) {
    return @llvm.ispc.cmpxchg.seq_cst.seq_cst(ptr, cmp, val);
}

template <typename F, typename I>
unmasked uniform F __atomic_cmpxchg_u(uniform int8 *uniform ptr, uniform F cmp, uniform F val) {
    uniform I cmp_i64 = __bitcast<uniform I, uniform F>(cmp);
    uniform I val_i64 = __bitcast<uniform I, uniform F>(val);
    uniform I res = __cmpxchg(ptr, cmp_i64, val_i64);
    return __bitcast<uniform F, uniform I>(res);
}

EXT uniform double __atomic_compare_exchange_uniform_double_global(uniform int8 *uniform ptr, uniform double cmp,
                                                                   uniform double val) {
    return __atomic_cmpxchg_u<double, int64>(ptr, cmp, val);
}
EXT uniform float __atomic_compare_exchange_uniform_float_global(uniform int8 *uniform ptr, uniform float cmp,
                                                                 uniform float val) {
    return __atomic_cmpxchg_u<float, int32>(ptr, cmp, val);
}
EXT uniform int32 __atomic_compare_exchange_uniform_int32_global(uniform int8 *uniform ptr, uniform int32 cmp,
                                                                 uniform int32 val) {
    return __cmpxchg(ptr, cmp, val);
}
EXT uniform int64 __atomic_compare_exchange_uniform_int64_global(uniform int8 *uniform ptr, uniform int64 cmp,
                                                                 uniform int64 val) {
    return __cmpxchg(ptr, cmp, val);
}

template <typename F, typename I>
unmasked varying F __atomic_cmpxchg_v(uniform int8 *uniform ptr, varying F cmp, varying F val) {
    varying F res;
    foreach_active(i) {
        uniform F r = __atomic_cmpxchg_u<F, I>(ptr, __extract(cmp, i), __extract(val, i));
        res = __insert(res, i, r);
    }
    return res;
}

UNMANGLED CDECL varying double __atomic_compare_exchange_double_global(uniform int8 *uniform ptr, varying double cmp,
                                                                       varying double val) {
    return __atomic_cmpxchg_v<double, int64>(ptr, cmp, val);
}
UNMANGLED CDECL varying float __atomic_compare_exchange_float_global(uniform int8 *uniform ptr, varying float cmp,
                                                                     varying float val) {
    return __atomic_cmpxchg_v<float, int32>(ptr, cmp, val);
}
UNMANGLED CDECL varying int32 __atomic_compare_exchange_int32_global(uniform int8 *uniform ptr, varying int32 cmp,
                                                                     varying int32 val) {
    return __atomic_cmpxchg_v<int32, int32>(ptr, cmp, val);
}
UNMANGLED CDECL varying int64 __atomic_compare_exchange_int64_global(uniform int8 *uniform ptr, varying int64 cmp,
                                                                     varying int64 val) {
    return __atomic_cmpxchg_v<int64, int64>(ptr, cmp, val);
}

#undef NOESCAPE
#undef ADDRSPACE
#undef READONLY
#undef READNONE
#undef NOALIAS
#undef EXT
