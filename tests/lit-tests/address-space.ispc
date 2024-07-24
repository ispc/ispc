// RUN: %{ispc} --target=host --nowrap --nostdlib %s --emit-llvm-text -o - | FileCheck %s

#define A(N) __attribute__((address_space(N)))

// check also that this macro is defined
#ifdef ISPC_ATTRIBUTE_SUPPORTED

// CHECK: define void @func1___un_3C_uni_3E_({{.*}} addrspace(4){{.*}}, {{.*}}
void func1(A(4) uniform int *uniform ptr) { return; }

// CHECK: define {{.*}} @func2___un_3C_unt_3E_({{.*}} addrspace(2){{.*}}, {{.*}}
A(2) uniform int8 *uniform func2(A(2) uniform int8 *uniform x) { return x; }
#endif // ISPC_ATTRIBUTE_SUPPORTED
