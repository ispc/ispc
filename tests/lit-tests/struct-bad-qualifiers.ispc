// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "task" qualifier is illegal outside of function declarations

struct Foo {
    task float x;
};
