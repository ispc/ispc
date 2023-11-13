// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal "typedef" provided with function definition

typedef float foo(float a, float b) { }

