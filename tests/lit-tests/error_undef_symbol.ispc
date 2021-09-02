//; RUN: not %{ispc} %s -o %t.o --target=host --nowrap --nostdlib 2>&1 | FileCheck %s -check-prefix=CHECK

// The code should not crash, but should emit a proper error.

// CHECK: Error: Undeclared symbol "bar"
uniform bool foo(uniform float v) {
  return ((1.f - bar(v)) < 0.1f);
}
