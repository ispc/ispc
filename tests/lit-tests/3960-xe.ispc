// Regression test for https://github.com/ispc/ispc/issues/3960
// On Xe targets too, pointers to forward declared structs refer to the
// complete struct, and structs declared in function bodies are described in
// the scope of the function.

// RUN: %{ispc} %s --target=xehpg-x8 --nostdlib --nowrap -g --emit-llvm-text -o - | FileCheck %s

// REQUIRES: XE_ENABLED

struct A;
uniform A *uniform global;
struct A {
    int val;
};

uniform int local(void *uniform v) {
    struct L;
    uniform L *uniform p = (uniform L * uniform)v;
    struct L {
        int l;
    };
    return global->val + p->l;
}

// CHECK-DAG: !DIGlobalVariable(name: "global", {{.*}}type: ![[GLOBAL_PTR:[0-9]+]]
// CHECK-DAG: ![[GLOBAL_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[A:[0-9]+]]
// CHECK-DAG: ![[A]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct A", {{.*}}elements:
// CHECK-DAG: ![[LOCAL_SP:[0-9]+]] = distinct !DISubprogram(name: "local",
// CHECK-DAG: !DILocalVariable(name: "p", {{.*}}type: ![[P_PTR:[0-9]+]]
// CHECK-DAG: ![[P_PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L:[0-9]+]]
// CHECK-DAG: ![[L]] = !DICompositeType(tag: DW_TAG_structure_type, name: "uniform struct L", scope: ![[LOCAL_SP]], {{.*}}elements:
