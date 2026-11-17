// RUN: %{ispc} %s --target=host --nowrap --emit-llvm-text -o - | FileCheck %s

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
