// This test checks that ISPC correcltly promote signed and unsigned int types in pointer arithmetics

// RUN: %{ispc} -O2 --target=host --nostdlib --nowrap --emit-llvm-text %s -o - 2>&1 | FileCheck %s

// CHECK-LABEL: @unsigned_uniform_{{.*}}
// CHECK-DAG: zext
uniform int* uniform unsigned_uniform(uniform int* uniform data, uniform uint offset_uint) {
    return data + offset_uint;
}

// CHECK-LABEL: @signed_uniform_{{.*}}
// CHECK-DAG: sext
uniform int* uniform signed_uniform(uniform int* uniform data, uniform int offset_sint) {
    return data + offset_sint;
}

// CHECK-LABEL: @unsigned_varying_{{.*}}
// CHECK-DAG: zext
uniform int* unsigned_varying(uniform int* data, varying uint offset_uint) {
    return data + offset_uint;
}

// CHECK-LABEL: @signed_varying_{{.*}}
// CHECK-DAG: sext
uniform int* signed_varying(uniform int* data, varying int offset_sint) {
    return data + offset_sint;
}
