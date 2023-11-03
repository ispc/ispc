// RUN: %{ispc} -DISPC --pic --target=host -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s
// RUN: %{cc} -x c++ -c %s -o %t.cpp.o --include %t.h
// RUN: %{cc} %t.o %t.cpp.o -o %t.cpp.bin
// RUN: %t.cpp.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: my_data = 10

#ifdef ISPC
struct Test {
  uint &my_data;
};

export void use(uniform Test &test) {
    print("my_data = %\n", test.my_data);
}
#else
#if defined(__cplusplus)
using namespace ispc;
int main(int argc, char **argv) {
    uint32_t x = 10;
    struct Test t = { x };
    use(t);
    return 0;
}
#else
int main(int argc, char **argv) {
    uint32_t x = 10;
    struct Test t = { &x };
    use(&t);
    return 0;
}
#endif
#endif
