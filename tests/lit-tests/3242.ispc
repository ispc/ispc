// This test checks that ISPC correctly handles varying and uniform indices
// signed and unsigned integer types when indexing.

// RUN: %{ispc} -O2 --target=host --nostdlib --nowrap --emit-llvm-text %s -o - 2>&1 | FileCheck %s

// CHECK-LABEL: fetch_with_uint__{{.*}}
// CHECK-DAG: zext
varying int fetch_with_uint(uniform int* data, varying uint offset_uint) {
    return data[offset_uint];
}

// CHECK-LABEL: fetch_with_sint__{{.*}}
// CHECK-DAG: sext
varying int fetch_with_sint(uniform int* data, varying int offset_sint) {
    return data[offset_sint];
}

// CHECK-LABEL: uniform_fetch_uint__{{.*}}
// CHECK-DAG: zext
uniform int uniform_fetch_uint(uniform int* uniform data, uniform uint offset_uint) {
    return data[offset_uint];
}

// CHECK-LABEL: uniform_fetch_sint__{{.*}}
// CHECK-DAG: sext
uniform int uniform_fetch_sint(uniform int* uniform data, uniform int offset_sint) {
    return data[offset_sint];
}
