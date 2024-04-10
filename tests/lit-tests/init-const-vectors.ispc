// RUN: %{ispc} --target=host --nowrap --nostdlib --debug-phase=first:first --emit-llvm-text %s -o - | FileCheck %s

// CHECK: @glb_goo_bool = global <4 x i8> <i8 -1, i8 0, i8 -1, i8 0>
// CHECK: @glb_goo_int8 = global <4 x i8> <i8 1, i8 0
// CHECK: @glb_goo_uint8 = global <4 x i8> <i8 1, i8 0
// CHECK: @glb_goo_int16 = global <4 x i16> <i16 1, i16 0
// CHECK: @glb_goo_uint16 = global <4 x i16> <i16 1, i16 0
// CHECK: @glb_goo_int32 = global <4 x i32> <i32 1, i32 0
// CHECK: @glb_goo_uint32 = global <4 x i32> <i32 1, i32 0
// CHECK: @glb_goo_int64 = global <4 x i64> <i64 1, i64 0
// CHECK: @glb_goo_uint64 = global <4 x i64> <i64 1, i64 0
// CHECK: @glb_goo_float = global <4 x float> <float 1.000000e+00, float 0.000000e+00
// CHECK: @glb_goo_double = global <4 x double> <double 1.000000e+00, double 0.000000e+00

// CHECK: @static.{{[0-9]+}}.v_foo_bool = internal constant <4 x i8> <i8 -1, i8 0, i8 -1, i8 0>
// CHECK: @static.{{[0-9]+}}.v_foo_int8 = internal constant <4 x i8> <i8 1, i8 0
// CHECK: @static.{{[0-9]+}}.v_foo_uint8 = internal constant <4 x i8> <i8 1, i8 0
// CHECK: @static.{{[0-9]+}}.v_foo_int16 = internal constant <4 x i16> <i16 1, i16 0
// CHECK: @static.{{[0-9]+}}.v_foo_uint16 = internal constant <4 x i16> <i16 1, i16 0
// CHECK: @static.{{[0-9]+}}.v_foo_int32 = internal constant <4 x i32> <i32 1, i32 0
// CHECK: @static.{{[0-9]+}}.v_foo_uint32 = internal constant <4 x i32> <i32 1, i32 0
// CHECK: @static.{{[0-9]+}}.v_foo_int64 = internal constant <4 x i64> <i64 1, i64 0
// CHECK: @static.{{[0-9]+}}.v_foo_uint64 = internal constant <4 x i64> <i64 1, i64 0
// CHECK: @static.{{[0-9]+}}.v_foo_float = internal constant <4 x float> <float 1.000000e+00, float 0.000000e+00
// CHECK: @static.{{[0-9]+}}.v_foo_double = internal constant <4 x double> <double 1.000000e+00, double 0.000000e+00

// CHECK: @const_initializer{{.*}} = internal constant <4 x i8> <i8 -1, i8 0, i8 -1, i8 0>
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x i8> <i8 1, i8 0
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x i8> <i8 1, i8 0
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x i16> <i16 1, i16 0
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x i16> <i16 1, i16 0
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x i32> <i32 1, i32 0
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x i32> <i32 1, i32 0
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x i64> <i64 1, i64 0
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x i64> <i64 1, i64 0
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x float> <float 1.000000e+00, float 0.000000e+00
// CHECK: @const_initializer.{{[0-9]+}} = internal constant <4 x double> <double 1.000000e+00, double 0.000000e+00

#define FOO(TYPE)                                                                     \
void foo_##TYPE(uniform float dst[], const uniform uint32 b, const varying float src) \
{                                                                                     \
    static const uniform TYPE<4> v_foo_##TYPE = { 1,0,1,0 };                          \
    if (v_foo_##TYPE[0] != 0) { dst[b+programIndex] = src; }                          \
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
    const uniform TYPE<4> cst_boo_##TYPE = { 1,0,1,0 };                               \
    if (cst_boo_##TYPE[0] != 0) { dst[b+programIndex] = src; }                        \
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
uniform TYPE<4> glb_goo_##TYPE = { 1,0,1,0 };                                         \
void goo_##TYPE(uniform float dst[], const uniform uint32 b, const varying float src) \
{                                                                                     \
    if (glb_goo_##TYPE[0] != 0) { dst[b+programIndex] = src; }                        \
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
