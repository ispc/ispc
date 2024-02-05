// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Initializer list for array "varying int32[2][4]" must have no more than 2 elements (has 3)

int a[2][4] = { { 1, 2, 3 }, { 1, 2, 3, 4 }, 1 };

// CHECK: Error: Illegal to declare a global variable with unsized array dimensions that aren't set with an initializer expression

int a6[][];

// CHECK: Error: Inconsistent initializer expression list lengths make it impossible

int a5[2][] = { { 1, 2, 3 }, { 1, 2, 3, 4 } };

// CHECK: Error: Inconsistent initializer expression list lengths make it impossible to size unsized array dimensions

void foo() {
    int b[2][] = { { 1, 2, 3 }, { 1, 2, 3, 4 } };
}

// CHECK: Error: Illegal to declare an unsized array variable without providing an initializer expression to set its size

void func1() { int a[][]; }

// CHECK: Error: Illegal to declare an unsized array variable without providing an initializer expression to set its size

int func2() {
    int a[];
}

// CHECK: Error: Type conversion from "const uniform int32" to "uniform int32 * varying" for initializer is not possible

int voo() {
    int * varying foo = 1l;
}