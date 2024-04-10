// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to delete non-pointer type

void func(int a) {
    delete a;
}

// CHECK: Error: syntax error, unexpected 'const'.

int * func2(int a) {
    return const new int[a];
}

// CHECK: Error: syntax error, unexpected '('

int * func3(int a) {
    return new int[a](10);
}

// CHECK: Error: Type conversion from "varying struct P" to "varying unsigned int32" for item count is not possible

struct P { int x; };

int * func4(P p) {
    return new int[p];
}

// CHECK: Error: Illegal to provide "varying" allocation count with "uniform new" expression

int * func5(int x) {
    return uniform new int[x];
}

// CHECK: Error: Can't convert from type "uniform int32 * varying" to type "uniform int32 * uniform" for return

int * uniform func6(int x) {
    return new int[x];
}

// CHECK: Error: Can't convert from type "varying float" to type "uniform float" for initializer

struct Point {
    uniform float x, y, z;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Point * uniform buf = uniform new uniform Point(a, b, 1234.);
    RET[programIndex] = buf->y;
    delete buf;
}