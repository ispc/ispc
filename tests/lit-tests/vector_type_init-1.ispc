// This test checks that uniform vectors conversion during initialization is perfomed using vector instructions.
// RUN: %{ispc} %s --target=host -O0 --emit-llvm-text -o - | FileCheck %s

// Vector initialization
// CHECK-LABEL: define <4 x double> @test1
// CHECK: fpext <4 x float>
// CHECK-NOT: fpext float
uniform double<4> test1() {
  const uniform float<4> f = {1.f, 2.f, 3.f, 4.f};
  const uniform double<4> d = {f[1], f[3], f.y, 0.0};
  return d;
}

// Vector initialization
// CHECK-LABEL: define <4 x i64> @test2
// CHECK: sext <4 x i32>
// CHECK-NOT: sext i32
uniform int64<4> test2(uniform int k) {
  const uniform int<4> i = {k, k*2, 3, k*3};
  const uniform int64<4> d = {i[1], i[3], i.y, 0};
  return d;
}

// Vector initialization
// CHECK-LABEL: define <4 x i64> @test3
// CHECK: sext <4 x i32>
// CHECK-NOT: sext i32
uniform int64<4> test3(uniform int k) {
  const uniform int<4> i = {k, k*2, 3, k*3};
  const uniform int64<4> d = {i.y, i.w, i.y};
  return d;
}

// CHECK-LABEL: define <4 x double> @test4
// CHECK: sitofp i32
// CHECK: fpext <4 x float>
// CHECK-NOT: fpext float
uniform double<4> test4(uniform int k) {
  const uniform float<4> f = {1.f, 2.f, 3.f, 4.f};
  const uniform double<4> d = {k, f[3], f.y, 0.0};
  return d;
}

// CHECK-LABEL: define <4 x double> @test5
// CHECK: fpext <4 x float>
// CHECK-NOT: fpext float
uniform double<3> test5() {
  const uniform float<3> f = {1.f, 2.f, 3.f};
  const uniform double<3> d = {2.f, f.z, f.y};
  return d;
}

// CHECK-LABEL: define <8 x double> @test6
// CHECK: sitofp <8 x i32>
// CHECK: fpext <8 x float>
// CHECK-NOT: fpext float
uniform double<5> test6(uniform int k, uniform float p) {
  const uniform float<3> f = {p + 1.f, p + 2.f, p + 3.f};
  const uniform int<3> i = {k, k+1, k+2};
  const uniform double<5> d = {f.x, i.x, f.y, i.y, 0.d};
  return d;
}
