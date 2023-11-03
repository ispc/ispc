// This test checks whether ISPC generates correct C and C++ header regarding reference as parameter.
// Check here that this test is compilable with ISPC/C/CPP compilers and runnable producing the sane output.

// RUN: %{ispc} -DISPC --pic --target=host -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s
// RUN: %{cc} -x c++ -c %s -o %t.cpp.o --include %t.h
// RUN: %{cc} %t.o %t.cpp.o -o %t.cpp.bin
// RUN: %t.cpp.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK-LABEL: uint x=10
// CHECK: uint x=4294967295

#ifdef ISPC
export void use(uniform uint &x) {
    print("uint x=%\n", x);
}
#else
uint32_t x = 10;
uint32_t max = 4294967295;
#if defined(__cplusplus)
using namespace ispc;
int main(int argc, char **argv) {
    use(x);
    use(max);
    return 0;
}
#else
int main(int argc, char **argv) {
    use(&x);
    use(&max);
    return 0;
}
#endif
#endif
