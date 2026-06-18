// RUN: %{ispc} %s --target=host --nostdlib --emit-llvm-text -o - | FileCheck %s

typedef int (*FuncType)();

int f0() { return 1; }
int f1() { return 2; }

constexpr uniform FuncType F0 = f0;
constexpr uniform FuncType F1 = f1;

constexpr uniform int has_fn(uniform FuncType f) {
    if (f) {
        return 1;
    }
    return 0;
}

constexpr uniform int Flag = has_fn(F1);

uniform int foo() { return Flag; }

// CHECK-DAG: @F0 = {{.*}} ptr @f0
// CHECK-DAG: @F1 = {{.*}} ptr @f1
// CHECK-DAG: @Flag = {{.*}} i32 1
