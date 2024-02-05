// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to provide "export" qualifier for functions with the same name but different types

export void foo() { }

export void foo(uniform int x) { }

// CHECK: Error: Vector-typed parameter "x" is illegal 

export void bar(uniform float<3> x) {
}
