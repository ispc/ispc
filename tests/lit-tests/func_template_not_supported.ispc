// RUN: not %{ispc} %s --target=host --nostdlib --nowrap  2>&1 | FileCheck %s

// CHECK-NOT: Please file a bug report

// CHECK: Default values for template type parameters are not yet supported.
// CHECK: Default values for template non-type parameters are not yet supported.
template <typename T = int, int I> T foo(T t) {
  T tt = t + 1;
  return tt;
}

template <typename T, int I=10> T foo(T t) {
  T tt = t + 1;
  return tt;
}

// CHECK: Template arguments deduction is not yet supported in template function specialization.
template <> int foo(int t);

// CHECK: Template arguments deduction is not yet supported in template function specialization.
template <> int foo(int t) { return 1; }
