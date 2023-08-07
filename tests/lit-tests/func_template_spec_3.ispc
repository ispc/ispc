// Check that error is produced when template function specialization is defined after it was already created through instantiation.
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap  2>&1 | FileCheck %s

// CHECK-NOT: Please file a bug report

// CHECK: Template function specialization was already defined.

template <typename T, typename C> noinline int goo(T argGooOne, C argGooTwo) {
    return argGooOne + argGooTwo;
}

float foo(int argFoo0, float argFoo1) {
    float a = goo<int, float>(argFoo0, argFoo1);
    int b = goo<int, int>(argFoo0, argFoo0);
    return a + b;
}

template <> noinline int goo<int, float>(int argGooOne, float argGooTwo) {
  return argGooOne * argGooTwo;
}
