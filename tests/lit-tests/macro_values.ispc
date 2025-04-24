// RUN: %{ispc} --target=host --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// CHECK: @ispc = {{.*}} constant i32 1
// CHECK: @ispc_uint_is_defined = {{.*}} constant i32 1
// CHECK: @ispc_attribute_supported = {{.*}} constant i32 1
const uniform int32 ispc = ISPC;
const uniform int32 ispc_uint_is_defined = ISPC_UINT_IS_DEFINED;
const uniform int32 ispc_attribute_supported = ISPC_ATTRIBUTE_SUPPORTED;

// CHECK-NOT: Error: syntax error, unexpected ')'.
void foo() {
    print("%", ISPC);
    print("%", ISPC_UINT_IS_DEFINED);
    print("%", ISPC_ATTRIBUTE_SUPPORTED);
}
