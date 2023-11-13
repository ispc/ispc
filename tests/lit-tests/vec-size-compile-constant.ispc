// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: syntax error, unexpected identifier, expecting int

void foo(uniform int i) {
    float<i> a;
}
