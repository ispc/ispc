// Check that type conversion optimization for template-instantiated short vectors 
// behaves the same as for basic short vectors.
// See vector_type_init and vector_type_convert tests for more details.

// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -O0 -o - | FileCheck %s
// CHECK-LABEL: <4 x double> @convert1___vyfvydCuni3
// CHECK: fpext <4 x float>
// CHECK-NOT: fpext float
// CHECK-LABEL: void @convert2___vyfvydCuni3
// CHECK: fpext <4 x float>
// CHECK-NOT: fpext float
template <typename T, typename M, int N>
noinline uniform M<N> convert1() {
  const uniform T<N> f = {1.f, 2.f, 3.f};
  const uniform M<N> d = {2.f, f.z, f.y};
  return d;
}

template <typename T, typename M, int N>
noinline void convert2(uniform M RET[], uniform T b) {
    uniform T<N> x = {1.0f, 2.0f, 3.0f};
    uniform M<N> z = x + b;
    RET[programIndex] = z[programIndex];
}

double<3> test1() {
    return convert1<float, double, 3>();
}

void test2(uniform double RET[], uniform int c) {
    convert2<float, double, 3>(RET, c);
}