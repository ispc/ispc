// This test ensures that enum fields are correctly included in generated headers.
// Regression test for issue where enum fields were missing from generated headers

// RUN: %{ispc} %s --target=host --nostdlib -h %t.h
// RUN: FileCheck --input-file=%t.h %s

// CHECK: enum E {
// CHECK-NEXT: A = 0 
// CHECK-NEXT: };
enum E { A };
export void foo(const uniform E x) {}
