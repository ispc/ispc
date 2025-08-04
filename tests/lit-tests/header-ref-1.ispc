// This test checks whether ISPC generates correct C and C++ header regarding reference as parameter.
// Check here that this test is compilable with ISPC/C/CPP compilers and runnable producing the sane output.

// RUN: %{cc_ispc_run} host test | FileCheck %s
// RUN: %{cxx_ispc_run} host test | FileCheck %s

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
