// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// XFAIL: ASAN_RUN

// CHECK: Error: syntax error
// CHECK-NOT: detected memory leaks
// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process

// TODO: check no memoryleak reported with "CHECK-NOT: detected memory leaks"
int foo(int i) { return 1;}

int bar() {
    int d = foo(2) // missing semicolon
    return d;
}
