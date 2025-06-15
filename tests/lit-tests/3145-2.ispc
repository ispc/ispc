// This test checks an error is produced when returning a non-void value from 
// a function returning a void type.

// RUN: not %{ispc} --target=host --emit-llvm-text --nowrap --nostdlib %s -O3 -o - 2>&1 | FileCheck %s

// CHECK-COUNT-2: Error: Can't return an expression of type

extern "C" int unoptimized_call();

void bar() {
    return 3;
}

void loop_fun() {
    return unoptimized_call();
}
