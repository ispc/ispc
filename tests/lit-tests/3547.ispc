// This test checks that ISPC doesn't crash on non-type checked NullPointerExpr
// inside TypeCastExpr

// RUN: %{ispc} --target=host --emit-llvm-text --nostdlib --nowrap %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: Assertion failed: "node->IsTypeChecked() && "Node must be type-checked before optimization""

struct E;
struct S {
    const uniform E *uniform f;
};
void foo(varying S *uniform x) {
    x->f = 0;
}
