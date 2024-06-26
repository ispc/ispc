// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "export" qualifier is illegal outside of function declarations.
struct Example {
    export int x;
};
// CHECK-COUNT-8: Error: "export" qualifier illegal in variable declaration.
export int x;
export uniform int w = 0;
void foo(export int y) {
  export float a, c;
  export varying float b = 0;
}
export uniform int boo(), f, *p;
export void func() {
  return;
}
export uniform int64 * uniform p64();
