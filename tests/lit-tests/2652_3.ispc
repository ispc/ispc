// RUN: %{cc_ispc_run} host test | FileCheck %s
// RUN: %{cxx_ispc_run} host test | FileCheck %s

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
