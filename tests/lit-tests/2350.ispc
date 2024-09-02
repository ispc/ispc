// RUN: %{ispc} --target=host --emit-llvm-text -o - %s -h %t.h 2>&1 | FileCheck %s
// RUN: cat %t.h | FileCheck --check-prefix=CHECK-HEADER %s

// CHECK: %{{.*}}uniform_TestStruct = type { float }
// CHECK: ret %{{.*}}uniform_TestStruct { float 1.000000e+00 }

// CHECK-HEADER: struct TestStruct {
// CHECK-HEADER: extern struct TestStruct test();

typedef struct {
    float t;
} TestStruct;

export uniform TestStruct test() {
    uniform TestStruct s;
    s.t = 1.0;
    return s;
}
