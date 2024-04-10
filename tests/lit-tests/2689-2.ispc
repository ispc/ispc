// RUN: %{ispc} --pic -O2 --target=host %s -o %t.o
// RUN: %{cc} %t.o -o %t.bin
// RUN: %t.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

uniform uint8<3> foo(uniform uint8<3> a, uniform uint8<3> b) { return a < b; }

// CHECK: 0 1 0

extern "C" uniform int main() {
    uniform uint8<3> a = { 5, 3, 9 };
    uniform uint8<3> b = { 1, 8, 2 };

    uniform uint8<3> r = foo(a, b);
    print("% % %\n", r[0], r[1], r[2]);
    return 0;
}
