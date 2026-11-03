// Regression test for https://github.com/ispc/ispc/issues/3986
// Check that an invalid argument (an undeclared symbol or an expression with an
// error type) is reported as an error and does not crash the compiler, in
// particular when a function template that contains it is instantiated.

// RUN: not %{ispc} --target=host --nowrap --nostdlib -o %t.o %s 2>&1 | FileCheck %s --implicit-check-not=FATAL
// RUN: not %{ispc} --target=host --nowrap --nostdlib --enable-llvm-intrinsics -DINTRINSICS -o %t.o %s 2>&1 | FileCheck %s --check-prefix=CHECK-INTRINSICS --implicit-check-not=FATAL
// RUN: not %{ispc} --target=host --nowrap --nostdlib -DINTRINSICS -o %t.o %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FLAG --implicit-check-not=FATAL

// CHECK: Error: Undeclared symbol "undeclared_symbol".
// CHECK: Error: Undeclared symbol "undeclared_symbol2".
// CHECK: Error: Undeclared symbol "undeclared_symbol3".
// CHECK: Error: Undeclared symbol "undeclared_symbol4".
// CHECK: Error: Undeclared symbol "undeclared_symbol5".

// CHECK-INTRINSICS: Error: Undeclared symbol "undeclared_symbol6".
// CHECK-INTRINSICS: Error: Undeclared symbol "undeclared_symbol7".
// CHECK-INTRINSICS: Error: Undeclared symbol "undeclared_symbol8".
// CHECK-INTRINSICS: Error: Trying to index into non-array, vector, or pointer type "uniform int32".
// CHECK-INTRINSICS: Error: Trying to index into non-array, vector, or pointer type "uniform int32".

// CHECK-NO-FLAG: Error: Undeclared symbol "undeclared_symbol6".
// CHECK-NO-FLAG: Error: Calling LLVM intrinsics from ISPC source code is an experimental feature

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

#ifdef INTRINSICS
// The intrinsic declaration is deferred until instantiation.
template <typename T> T outer6(T y) { return @llvm.fshl(undeclared_symbol6, y, y); }
uniform int32 use6(uniform int32 v) { return outer6<uniform int32>(v); }

template <typename T> T outer7(T y) { return @llvm.fshl(y, undeclared_symbol7, y); }
uniform int32 use7(uniform int32 v) { return outer7<uniform int32>(v); }

uniform int32 use8(uniform int32 v) { return @llvm.fshl(undeclared_symbol8, v, v); }

// The argument is not null, but its type is.
template <typename T> T outer9(T y) { return @llvm.fshl(y[0], y, y); }
uniform int32 use9(uniform int32 v) { return outer9<uniform int32>(v); }

uniform int32 use10(uniform int32 v) { return @llvm.fshl(v[0], v, v); }
#endif
