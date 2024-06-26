// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "task" qualifier is illegal outside of function declarations.
struct Example {
    task int x;
};
// CHECK-COUNT-8: Error: "task" qualifier illegal in variable declaration.
task int x;
task uniform int w = 0;
void foo(task int y) {
  task float a, c;
  task varying float b = 0;
}
task int f, *p;
task void func() {
  return;
}
