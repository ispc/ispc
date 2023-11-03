// This test checks whether ISPC generates correct C and C++ header regarding used bool function parameter.
// Check here that this test is compilable with ISPC/C/CPP compilers and runnable producing the sane output.

// RUN: %{ispc} -DISPC --pic --target=host -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s
// RUN: %{cc} -x c++ -c %s -o %t.cpp.o --include %t.h
// RUN: %{cc} %t.o %t.cpp.o -o %t.cpp.bin
// RUN: %t.cpp.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK-LABEL: bool x=false
// CHECK-COUNT-2: bool x=true
// CHECK: bool x=false


#ifdef ISPC
export void use(uniform bool x) {
    print("bool x=%\n", x);
}
#else
#if defined(__cplusplus)
using namespace ispc;
int main(int argc, char **argv) {
    use(false);
    use(true);
    use(5);
    use(!true);
    return 0;
}
#else
int main(int argc, char **argv) {
    use(0);
    use(1);
    use(5);
    use(!1);
    return 0;
}
#endif
#endif
