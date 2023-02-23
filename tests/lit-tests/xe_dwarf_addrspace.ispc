// This test checks that dwarfAddressSpace attribute is present in DIDerivedType metadata for Xe targets.
// RUN: %{ispc} %s --target=gen9-x8 -g --emit-llvm-text -o - | FileCheck %s

// REQUIRES: XE_ENABLED

// CHECK: !DIDerivedType(tag: DW_TAG_pointer_type, {{.*}} dwarfAddressSpace: {{[0-9]}})
// CHECK: !DIDerivedType(tag: DW_TAG_reference_type, {{.*}} dwarfAddressSpace: {{[0-9]}})
void test_ptr(void *uniform p) {
}
void test_ref(const uniform int &ref) {
}
