// RUN: %{ispc} --target=host --nowrap --nostdlib --debug-phase=2:2 %s --no-discard-value-names -o t.o | FileCheck %s

#define A(N) __attribute__((address_space(N)))

// CHECK: declare void @foo___un_3C_uni_3E_({{.*}} addrspace(4){{.*}}, {{.*}}
void foo(A(4) uniform int *uniform ptr);

// CHECK: declare {{.*}} @boo___un_3C_unt_3E_({{.*}} addrspace(2){{.*}}, {{.*}}
uniform int8 *uniform boo(A(2) uniform int8 *uniform x);

// CHECK: declare void @struct____{{.*}}({{.*}} addrspace(3){{.*}}, {{.*}}
typedef struct {
    int x;
} my_struct;
void struct_(A(3) uniform my_struct *uniform y);
