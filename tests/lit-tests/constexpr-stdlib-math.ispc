// RUN: %{ispc} %s --target=host --nowrap --werror --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=host --nowrap -DTEST_NAN --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --check-prefix=NAN-WARN --implicit-check-not=Error:

constexpr uniform float newton = _cbrt_newton_iter((uniform float)2, (uniform float)20);
constexpr uniform double halley = _cbrt_halley_iter((uniform double)1, (uniform double)4);
constexpr varying uint32 lane_quotient = _fast_div3((varying uint32)programIndex * 3u);

// CHECK-DAG: @newton = {{.*}} constant float 3.000000e+00
// CHECK-DAG: @halley = {{.*}} constant double 1.500000e+00

// CHECK-LABEL: @lane_quotient_error
// CHECK: ret <{{[1-9][0-9]*}} x i32> zeroinitializer
varying uint32 lane_quotient_error() { return lane_quotient - (varying uint32)programIndex; }

template <uint32 Quotient>
uniform uint32 return_quotient() { return Quotient; }

template <uint32 N>
uniform uint32 divide() { return return_quotient<_fast_div3(N)>(); }

// CHECK-LABEL: @test_divide
// CHECK: ret i32 1000000000
uniform uint32 test_divide() { return divide<3000000000u>(); }

// These existing public helpers share their implementation with runtime calls.
static_assert(abs((uniform int8)-127) == 127);
static_assert(abs((uniform int16)-32767) == 32767);
static_assert(abs((uniform int32)-2147483647) == 2147483647);
static_assert(abs((uniform int64)-9223372036854775807ll) == 9223372036854775807ll);
static_assert(!isnan((uniform float16)-0.0f16));
static_assert(!isnan((uniform float)-0.0f));
static_assert(!isnan((uniform double)-0.0d));
static_assert(!isnan(0x1p128f));
#ifdef TEST_NAN
// Producing NaN through exceptional arithmetic has an existing diagnostic;
// it must still compare unordered during constexpr evaluation.
// NAN-WARN: Warning: Binary expression with type "const uniform float16" can't represent value.
static_assert(isnan(0.0f16 / 0.0f16));
// NAN-WARN: Warning: Binary expression with type "const uniform float" can't represent value.
static_assert(isnan(0.0f / 0.0f));
// NAN-WARN: Warning: Binary expression with type "const uniform double" can't represent value.
static_assert(isnan(0.0d / 0.0d));
#endif

constexpr varying int lane_abs = abs(-(programIndex + 1));
// CHECK-LABEL: @lane_abs_error
// CHECK: ret <{{[1-9][0-9]*}} x i32> zeroinitializer
varying int lane_abs_error() { return lane_abs - programIndex - 1; }

template <int N> uniform int magnitude() { return return_quotient<abs(N)>(); }
// CHECK-LABEL: @test_magnitude
// CHECK: ret i32 5
uniform int test_magnitude() { return magnitude<-5>(); }
