// RUN: not %{ispc} %s --target=host --nostdlib --nowrap  2>&1 | FileCheck %s

// CHECK-NOT: Please file a bug report

// CHECK: Default values for template type parameters are not yet supported.
// CHECK: Non-type template paramters are not yet supported.
template <typename T = int, int I> T foo(T t) {
  T tt = t + 1;
  return tt;
}

// CHECK: Template function specialization not yet supported.
template <> int foo(int t);

// CHECK: Template function specialization not yet supported.
template <> int foo(int t) { return 1; }
