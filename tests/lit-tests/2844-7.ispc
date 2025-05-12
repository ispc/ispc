// Check incomplete struct type handling in template functions parameters and return types

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: variable 'x' has incomplete struct type 'varying struct Si' and cannot be defined
// CHECK: Error: Template parameter 'a' with type '/*unbound*/ T' can't be instantiated with incomplete type 'varying struct Si'
// CHECK: Error: Failed to get a candidate for template instantiation
// CHECK: Error: Unable to find any matching overload for call to function "bar".
template <typename T> T bar(T a) { return a; }

int foo() {
    struct Si x;
    bar(x);
    return 0;
}
