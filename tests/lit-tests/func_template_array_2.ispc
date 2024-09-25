// The test checks that template type and non-type parameters can be used to declare an array.
// RUN: %{ispc} %s --ast-dump --target=host --nostdlib | FileCheck %s -check-prefix=CHECK-AST
// RUN: %{ispc} %s --target=host --nostdlib -o %t.o

// CHECK-AST: FunctionTemplate {{.*}} [ void(uniform T * uniform arr)] "func"
// CHECK-AST: Variable O (uniform T[C])
// CHECK-AST: (instantiation <varying float, 64>) {{.*}}  [ void(uniform float * uniform arr)] "func"
// CHECK-AST: Variable O (uniform float[64])

template <typename T, int C> noinline void func(uniform T arr[C]) {
    uniform T O[C] = {0};
    arr[programIndex] = O[programIndex];
}

void test(uniform float x[]) {
    func<float, 64>(x);
}
