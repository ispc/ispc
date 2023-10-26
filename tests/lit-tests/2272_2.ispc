// RUN: %{ispc} --pic -O2 --target=host %s -o %t.o
// RUN: %{cc} %t.o -o %t.bin
// RUN: %t.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

struct Test { bool b; };

// CHECK: true

extern "C" uniform int main() {
    uniform struct Test t = { 1 < 2 };
    print("%\n", t.b);
    return 0;
}
