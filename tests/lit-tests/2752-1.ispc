// RUN: %{ispc} -DISPC --pic --target=host -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s
// RUN: %{cc} -x c++ -c %s -o %t.cpp.o --include %t.h
// RUN: %{cxx} %t.o %t.cpp.o -o %t.cpp.bin
// RUN: %t.cpp.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: b=1, b==true is 1

#ifdef ISPC
export void set(uniform bool &result) {
    result = true;
}
#else
#if defined(__cplusplus)
#include <iostream>
using namespace ispc;
int main(int argc, char **argv) {
    bool b = false;
    set(b);
    std::cout << "b=" << b << ", b==true is " <<  (b == true) << "\n";
    return 0;
}
#else
#include <stdio.h>
int main(int argc, char **argv) {
    bool b = 0;
    set(&b);
    printf("b=%i, b==true is %i\n", b, b == !0);
    return 0;
}
#endif
#endif

