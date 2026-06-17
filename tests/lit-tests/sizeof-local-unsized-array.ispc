// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -o - | FileCheck %s

export uniform int local_const_array_size() {
    const uniform double x[] = {1.0d, 2.0d, 3.0d};
    const uniform int count = sizeof(x) / sizeof(x[0]);
    return count;
}

// CHECK-LABEL: define {{.*}} @local_const_array_size
// CHECK: ret i32 3
