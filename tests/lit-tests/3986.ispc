// Check that an undeclared symbol passed to a call inside a function template
// is reported as an error and does not crash the compiler on instantiation.

// RUN: not %{ispc} --target=host --nowrap --nostdlib -o %t.o %s 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR
// CHECK: Error: Undeclared symbol "undeclared_symbol".
// CHECK: Error: Undeclared symbol "undeclared_symbol2".
// CHECK: Error: Undeclared symbol "undeclared_symbol3".
// CHECK: Error: Undeclared symbol "undeclared_symbol4".
// CHECK: Error: Undeclared symbol "undeclared_symbol5".
// CHECK-NOT: FATAL ERROR

float twice(float x) { return x + x; }
template <typename T> T callee(T x) { return x; }

template <typename T> T outer(T y) { return twice(undeclared_symbol); }
float use(float v) { return outer<float>(v); }

template <typename T> T outer2(T y) { return callee<T>(undeclared_symbol2); }
float use2(float v) { return outer2(v); }

template <typename T> T outer3(T y) { return y + twice(undeclared_symbol3); }
float use3(float v) { return outer3<float>(v); }

template <typename T> T outer4(T y) {
    T arr[2] = {y, undeclared_symbol4};
    return arr[0];
}
float use4(float v) { return outer4<float>(v); }

template <typename T> void outer5(T y) { delete undeclared_symbol5; }
void use5(float v) { outer5<float>(v); }
