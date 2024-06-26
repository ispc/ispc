// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "unmasked" qualifier is illegal outside of function declarations.
struct Example {
    unmasked int x;
};
// CHECK-COUNT-8: Error: "unmasked" qualifier illegal in variable declaration.
unmasked int x;
unmasked uniform int w = 0;
void foo(unmasked int y) {
  unmasked float a, c;
  unmasked varying float b = 0;
}
unmasked int boo(), f, *p;
unmasked void func() {
  unmasked { func(); }
  return;
}
unmasked uniform int64 * uniform p64();
