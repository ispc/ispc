// Check that error is produced when template function specialization is defined several times.
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap  2>&1 | FileCheck %s

// CHECK-NOT: Please file a bug report

template <typename T> noinline int goo(T argGooOne, T argGooTwo) {
    return argGooOne + argGooTwo;
}

template <> noinline int goo<int>(int argGooOne, int argGooTwo) {
  return argGooOne * argGooTwo;
}

// CHECK: Template function specialization was already defined.
template <> noinline int goo<int>(int argGooOne, int argGooTwo) {
  return argGooOne * argGooTwo;
}

// CHECK: No matching function template found for specialization.
template <> noinline int goo<float, int>(float argGooOne, float argGooTwo) {
  return argGooOne + argGooTwo;
}

// CHECK: Template function specialization was declared but never defined.
template <> noinline int goo<double>(double argGooOne, double argGooTwo);

float foo(int argFoo0, float argFoo1) {
    float a = goo<int>(argFoo0, (int)argFoo1);
    int b = goo<float>((float)argFoo0, argFoo1);
    double c = goo<double>((double)argFoo0, (double)argFoo1);
    return a + b;
}
