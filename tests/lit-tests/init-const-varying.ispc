// RUN: %{ispc} --target=host --nowrap --nostdlib --debug-phase=first:first %s --emit-llvm-text -o - | FileCheck %s

// CHECK: @programCount = internal constant i32 [[WIDTH:[0-9]+]]

// CHECK: @glb_goo_bool = global <[[WIDTH]] x i8> <i8 -1, i8 0
// CHECK: @glb_goo_int8 = global <[[WIDTH]] x i8> <i8 1, i8 0
// CHECK: @glb_goo_uint8 = global <[[WIDTH]] x i8> <i8 1, i8 0
// CHECK: @glb_goo_int16 = global <[[WIDTH]] x i16> <i16 1, i16 0
// CHECK: @glb_goo_uint16 = global <[[WIDTH]] x i16> <i16 1, i16 0
// CHECK: @glb_goo_int32 = global <[[WIDTH]] x i32> <i32 1, i32 0
// CHECK: @glb_goo_uint32 = global <[[WIDTH]] x i32> <i32 1, i32 0
// CHECK: @glb_goo_int64 = global <[[WIDTH]] x i64> <i64 1, i64 0
// CHECK: @glb_goo_uint64 = global <[[WIDTH]] x i64> <i64 1, i64 0
// CHECK: @glb_goo_float = global <[[WIDTH]] x float> <float 1.000000e+00, float 0.000000e+00
// CHECK: @glb_goo_double = global <[[WIDTH]] x double> <double 1.000000e+00, double 0.000000e+00

// CHECK: @static.{{[0-9]+}}.v_foo_bool = internal constant <[[WIDTH]] x i8> <i8 -1, i8 0
// CHECK: @static.{{[0-9]+}}.v_foo_int8 = internal constant <[[WIDTH]] x i8> <i8 1, i8 0
// CHECK: @static.{{[0-9]+}}.v_foo_uint8 = internal constant <[[WIDTH]] x i8> <i8 1, i8 0
// CHECK: @static.{{[0-9]+}}.v_foo_int16 = internal constant <[[WIDTH]] x i16> <i16 1, i16 0
// CHECK: @static.{{[0-9]+}}.v_foo_uint16 = internal constant <[[WIDTH]] x i16> <i16 1, i16 0
// CHECK: @static.{{[0-9]+}}.v_foo_int32 = internal constant <[[WIDTH]] x i32> <i32 1, i32 0
// CHECK: @static.{{[0-9]+}}.v_foo_uint32 = internal constant <[[WIDTH]] x i32> <i32 1, i32 0
// CHECK: @static.{{[0-9]+}}.v_foo_int64 = internal constant <[[WIDTH]] x i64> <i64 1, i64 0
// CHECK: @static.{{[0-9]+}}.v_foo_uint64 = internal constant <[[WIDTH]] x i64> <i64 1, i64 0
// CHECK: @static.{{[0-9]+}}.v_foo_float = internal constant <[[WIDTH]] x float> <float 1.000000e+00, float 0.000000e+00
// CHECK: @static.{{[0-9]+}}.v_foo_double = internal constant <[[WIDTH]] x double> <double 1.000000e+00, double 0.000000e+00

// CHECK: %cst_boo_bool = alloca <[[WIDTH]] x i8>
// CHECK: %cst_boo_int8 = alloca <[[WIDTH]] x i8>
// CHECK: %cst_boo_uint8 = alloca <[[WIDTH]] x i8>
// CHECK: %cst_boo_int16 = alloca <[[WIDTH]] x i16>
// CHECK: %cst_boo_uint16 = alloca <[[WIDTH]] x i16>
// CHECK: %cst_boo_int32 = alloca <[[WIDTH]] x i32>
// CHECK: %cst_boo_uint32 = alloca <[[WIDTH]] x i32>
// CHECK: %cst_boo_int64 = alloca <[[WIDTH]] x i64>
// CHECK: %cst_boo_uint64 = alloca <[[WIDTH]] x i64>
// CHECK: %cst_boo_float = alloca <[[WIDTH]] x float>
// CHECK: %cst_boo_double = alloca <[[WIDTH]] x double>

#if   TARGET_WIDTH == 2
#define INITIALIZER 1,0,
#elif TARGET_WIDTH == 4
#define INITIALIZER 1,0,1,0,
#elif TARGET_WIDTH == 8
#define INITIALIZER 1,0,1,0,1,0,1,0,
#elif TARGET_WIDTH == 16
#define INITIALIZER 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
#elif TARGET_WIDTH == 32
#define INITIALIZER 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
#elif TARGET_WIDTH == 64
#define INITIALIZER 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
#endif

#define FOO(TYPE)                                                                     \
void foo_##TYPE(uniform float dst[], const uniform uint32 b, const varying float src) \
{                                                                                     \
    static const varying TYPE v_foo_##TYPE = { INITIALIZER };                         \
    if (v_foo_##TYPE != 0) { dst[b+programIndex] = src; }                             \
}

FOO(bool)
FOO(int8)
FOO(uint8)
FOO(int16)
FOO(uint16)
FOO(int32)
FOO(uint32)
FOO(int64)
FOO(uint64)
FOO(float)
FOO(double)

#define BOO(TYPE)                                                                     \
void boo_##TYPE(uniform float dst[], const uniform uint32 b, const varying float src) \
{                                                                                     \
    const varying TYPE cst_boo_##TYPE = { INITIALIZER };                              \
    if (cst_boo_##TYPE != 0) { dst[b+programIndex] = src; }                           \
}

BOO(bool)
BOO(int8)
BOO(uint8)
BOO(int16)
BOO(uint16)
BOO(int32)
BOO(uint32)
BOO(int64)
BOO(uint64)
BOO(float)
BOO(double)

#define GOO(TYPE)                                                                     \
varying TYPE glb_goo_##TYPE = { INITIALIZER };                                        \
void goo_##TYPE(uniform float dst[], const uniform uint32 b, const varying float src) \
{                                                                                     \
    if (glb_goo_##TYPE != 0) { dst[b+programIndex] = src; }                           \
}

GOO(bool)
GOO(int8)
GOO(uint8)
GOO(int16)
GOO(uint16)
GOO(int32)
GOO(uint32)
GOO(int64)
GOO(uint64)
GOO(float)
GOO(double)
