// The test checks that template non-type parameters can be used to set an array dimension.
// RUN: %{ispc} %s --ast-dump --target=host --nostdlib | FileCheck %s -check-prefix=CHECK-AST
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

// CHECK-AST: FunctionTemplate {{.*}} [ void(varying float * uniform arr)] "func"
// CHECK-AST: Variable O (uniform float[C])
// CHECK-AST: (instantiation <64>) {{.*}}  [ void(varying float * uniform arr)] "func"
// CHECK-AST: Variable O (uniform float[64])

template <int C> noinline void func(float arr[C]) {
    uniform float O[C] = {0};
    arr[programIndex] = O[programIndex];
}

void test(float x[]) {
    func<64>(x);
}
