// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "inline" qualifier is illegal outside of function declarations.
struct Example {
    inline int x;
};
// CHECK-COUNT-8: Error: "inline" qualifier illegal in variable declaration.
inline int x;
inline uniform int w = 0;
void foo(inline int y) {
  inline float a, c;
  inline varying float b = 0;
}
inline int boo(), f, *p;
inline void func() {
  return;
}
inline uniform int64 * uniform p64();
