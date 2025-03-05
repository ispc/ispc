// Check incomplete struct type handling in template functions parameters and return types

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: variable 'x' has incomplete struct type 'varying struct Si' and cannot be defined
// CHECK: Error: return type is an incomplete type: varying struct Si
// CHECK: Error: parameter 'a' is an incomplete type: varying struct Si
// CHECK: Error: Attempt to allocate memory for incomplete type "varying struct Si".
// CHECK: Error: Malformed return value in function with non-void return type.
template <typename T> T bar(T a) { return a; }

int foo() {
    struct Si x;
    bar(x);
    return 0;
}
