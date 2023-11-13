// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Vector-typed parameter "x" is illegal 

export void foo(uniform float<3> x) {
}
