// This test checks there is no error produced when calling void function in 
// return statements of void functions and that the call is correctly evaluated.
// Note that this is valid for a function to call itself in the return statement.

// RUN: %{ispc} --target=host --emit-llvm-text --nowrap --nostdlib %s -O3 -o - 2>&1 | FileCheck %s

// CHECK-COUNT-3: tail call void @unoptimized_call()
// CHECK: br label %tailrecurse

extern "C" void unoptimized_call();

void bar_v1() {
    unoptimized_call();
}

void bar_v2() {
    return unoptimized_call();
}

void foo() {
    return bar_v1();
}

void loop_fun() {
    return loop_fun();
}
