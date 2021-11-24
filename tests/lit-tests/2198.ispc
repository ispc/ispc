// The test checks compiler behavior when launch is used with function pointers.
// When uniform function pointer is used, it should compile successfully.
// When varying function pointer is used, code should not crash, but should emit a proper error.
// RUN: not %{ispc} %s -o %t.o --target=host --nowrap -DCALL_VARYING 2>&1 | FileCheck %s -check-prefix=CHECK-CALL_VARYING
// RUN: %{ispc} %s -o %t.o --target=host --nowrap -DCALL_UNIFORM

// CHECK-CALL_VARYING: Error: Must provide function name or uniform function pointer to "task"-qualified function for "launch" expression
typedef task void (*TaskFn)();

struct Test {
    TaskFn fn;
};

task void example() {
    print("%\n", taskIndex0);
}
#ifdef CALL_VARYING
export void callit_varying() {
    Test test;
    test.fn = example;
    launch[1] test.fn();
}
#endif
#ifdef CALL_UNIFORM
export void callit_uniform() {
    uniform Test test;
    test.fn = example;
    launch[1] test.fn();
}
#endif