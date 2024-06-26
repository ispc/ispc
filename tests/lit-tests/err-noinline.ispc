// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "noinline" qualifier is illegal outside of function declarations.
struct Example {
    noinline int x;
};
// CHECK-COUNT-8: Error: "noinline" qualifier illegal in variable declaration.
noinline int x;
noinline uniform int w = 0;
void foo(noinline int y) {
  noinline float a, c;
  noinline varying float b = 0;
}
noinline int boo(), f, *p;
noinline void func() {
  return;
}
noinline uniform int64 * uniform p64();
