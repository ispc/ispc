// RUN: %{ispc} %s --nostdlib --target=host --ast-dump -o %t.o 2>&1 | FileCheck %s

__attribute__((address_space(3))) uniform int * uniform b;

void foo(__attribute__((noescape)) uniform int * uniform a) { }

// CHECK: Function <{{.*}}ast-dump-attrs.ispc:9.70 - 9.72> [ void(uniform int32 addrspace(2) * uniform c)] "func"
// CHECK: (param 0) [uniform int32 addrspace(2) * uniform] c
void func(__attribute__((address_space(2))) uniform int * uniform c) {}

uniform int func2() {
    // CHECK: Variable d (uniform int32 addrspace(1) * uniform)
    // CHECK: SymbolExpr <{{.*}}ast-dump-attrs.ispc:15.13 - 15.14> [uniform int32 addrspace(1) * uniform] symbol name: d
    __attribute__((address_space(1))) uniform int * uniform d;
    return *d;
}
