// This test checks generated code for varying integer division.

// RUN: %{ispc} -O2 --target=host %s --emit-llvm-text -o - | FileCheck %s

// CHECK-LABEL: @foo_i___vyivyi
// CHECK-DAG:  br i1 %{{.*}}, label %direct_vector_div, label %call_vector_div
// CHECK-DAG: direct_vector_div:
// CHECK-NEXT:  %div_a_load_b_load_direct_div = sdiv <[[WIDTH:.*]] x i32> %a, %b
// CHECK-NEXT:  br label %post_div
// CHECK-DAG: call_vector_div:
// CHECK-DAG: %{{.*}} = sdiv i32 %{{.*}}, %{{.*}}
// CHECK-DAG: post_div:
// CHECK-NEXT:  %div_a_load_b_load_result = phi <[[WIDTH]] x i32> [ %div_a_load_b_load_direct_div, %direct_vector_div ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:  ret <[[WIDTH]] x i32> %div_a_load_b_load_result
int foo_i(int a, int b) { return a / b; }

// CHECK-LABEL: @foo_u___vyUvyU
// CHECK-DAG:  br i1 %{{.*}}, label %direct_vector_div, label %call_vector_div
// CHECK-DAG: direct_vector_div:
// CHECK-NEXT:  %div_a_load_b_load_direct_div = udiv <[[WIDTH:.*]] x i64> %a, %b
// CHECK-NEXT:  br label %post_div
// CHECK-DAG: call_vector_div:
// CHECK-DAG: %{{.*}} = udiv i64 %{{.*}}, %{{.*}}
// CHECK-DAG: post_div:
// CHECK-NEXT:  %div_a_load_b_load_result = phi <[[WIDTH]] x i64> [ %div_a_load_b_load_direct_div, %direct_vector_div ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:  ret <[[WIDTH]] x i64> %div_a_load_b_load_result
uint64 foo_u(uint64 a, uint64 b) { return a / b; }

// CHECK-LABEL: @bar_i___vysvys
// CHECK-DAG:  br i1 %{{.*}}, label %direct_vector_div, label %call_vector_div
// CHECK-DAG: direct_vector_div:
// CHECK-NEXT:  %mod_a_load_b_load_direct_div = srem <[[WIDTH:.*]] x i16> %a, %b
// CHECK-NEXT:  br label %post_div
// CHECK-DAG: call_vector_div:
// CHECK-DAG: %{{.*}} = srem i16 %{{.*}}, %{{.*}}
// CHECK-DAG: post_div:
// CHECK-NEXT:  %mod_a_load_b_load_result = phi <[[WIDTH]] x i16> [ %mod_a_load_b_load_direct_div, %direct_vector_div ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:  ret <[[WIDTH]] x i16> %mod_a_load_b_load_result
int16 bar_i(int16 a, int16 b) { return a % b; }

// CHECK-LABEL: @bar_u___vyTvyT
// CHECK-DAG:  br i1 %{{.*}}, label %direct_vector_div, label %call_vector_div
// CHECK-DAG: direct_vector_div:
// CHECK-NEXT:  %mod_a_load_b_load_direct_div = urem <[[WIDTH:.*]] x i8> %a, %b
// CHECK-NEXT:  br label %post_div
// CHECK-DAG: call_vector_div:
// CHECK-DAG: %{{.*}} = urem i8 %{{.*}}, %{{.*}}
// CHECK-DAG: post_div:
// CHECK-NEXT:  %mod_a_load_b_load_result = phi <[[WIDTH]] x i8> [ %mod_a_load_b_load_direct_div, %direct_vector_div ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:  ret <[[WIDTH]] x i8> %mod_a_load_b_load_result
uint8 bar_u(uint8 a, uint8 b) { return a % b; }

// uniform vectors should be divised regardless of the mask

// CHECK-LABEL: @func_div___uni_3C_4_3E_uni_3C_4_3E_
// CHECK-NEXT: allocas:
// CHECK-NEXT:   %div_a_load_b_load = sdiv <4 x i32> %a, %b
// CHECK-NEXT:   ret <4 x i32> %div_a_load_b_load
uniform int<4> func_div(uniform int<4> a, uniform int<4> b) { return a / b; }

// CHECK-LABEL: @func_mod___uni_3C_4_3E_uni_3C_4_3E_
// CHECK-NEXT: allocas:
// CHECK-NEXT:   %mod_a_load_b_load = srem <4 x i32> %a, %b
// CHECK-NEXT:   ret <4 x i32> %mod_a_load_b_load
uniform int<4> func_mod(uniform int<4> a, uniform int<4> b) { return a % b; }
