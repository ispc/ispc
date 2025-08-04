// RUN: %{cc_ispc_run} host test | FileCheck %s
// RUN: %{cxx_ispc_run} host test | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: bool = false

#ifdef ISPC
struct Test { bool b; };

export void use(uniform Test &test) {
    print("bool = %\n", test.b);
}
#else
#if defined(__cplusplus)
using namespace ispc;
int main(int argc, char **argv) {
    struct Test t = { 10 > 20};
    use(t);
    return 0;
}
#else
int main(int argc, char **argv) {
    struct Test t = { 10 > 20 };
    use(&t);
    return 0;
}
#endif
#endif

