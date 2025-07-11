// This test checks that the compiler does not crash when calling templated functions
// with uniform bool<N> parameters using arguments of different types. This was previously
// causing an LLVM assertion failure due to type signature mismatch between
// function parameters and arguments.

// RUN: %{ispc} --target=host --emit-llvm-text --nowrap %s -o - | FileCheck %s
// RUN: %{ispc} --target=host --nowrap %s -o %t.o

// CHECK-NOT: FATAL ERROR

// Test different condition types converting to bool<N>
// CHECK-LABEL: @test_int_cond___uni_3C_4_3E_uni_3C_4_3E_uni_3C_4_3E_(
// CHECK: icmp ne <4 x i32> %cond, zeroinitializer

// CHECK-LABEL: @test_float_cond___unf_3C_4_3E_uni_3C_4_3E_uni_3C_4_3E_(
// CHECK: fcmp one <4 x float> %cond, zeroinitializer

// CHECK-LABEL: @test_int8_cond___unt_3C_4_3E_uni_3C_4_3E_uni_3C_4_3E_(
// CHECK: icmp ne <4 x i8> %cond, zeroinitializer

// Test different return types
// CHECK-LABEL: @test_bool_return___uni_3C_4_3E_uni_3C_4_3E_uni_3C_4_3E_(
// CHECK: icmp ne <4 x i32> %cond, zeroinitializer

#define N 4

// Template function that expects uniform bool<N> condition
template <typename RetT>
inline uniform RetT<N> bool_select(uniform bool<N> cond, uniform int<N> a, uniform int<N> b) {
    uniform RetT<N> result;
    foreach (i = 0 ... N) {
        result[i] = cond[i] ? a[i] : b[i];
    }
    return result;
}

// Test functions with different condition types (should convert to bool<N>)
uniform int<N> test_int_cond(uniform int<N> cond, uniform int<N> a, uniform int<N> b) {
    return bool_select<int>(cond, a, b);
}

uniform int<N> test_float_cond(uniform float<N> cond, uniform int<N> a, uniform int<N> b) {
    return bool_select<int>(cond, a, b);
}

uniform int<N> test_int8_cond(uniform int8<N> cond, uniform int<N> a, uniform int<N> b) {
    return bool_select<int>(cond, a, b);
}

// Test functions with different return types
uniform int<N> test_bool_return(uniform int<N> cond, uniform int<N> a, uniform int<N> b) {
    return bool_select<bool>(cond, a, b);
}
