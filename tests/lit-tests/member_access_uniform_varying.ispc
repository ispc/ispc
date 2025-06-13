// Test struct member access on function calls with uniform and varying contexts

// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: Error

struct Point {
    float x, y;
};

struct Color {
    float r, g, b, a;
};

// Uniform function returning struct
uniform Point makeUniformPoint(uniform float x, uniform float y) {
    uniform Point p;
    p.x = x;
    p.y = y;
    return p;
}

// Varying function returning struct
Point makeVaryingPoint(float x, float y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

uniform Color makeUniformColor(uniform float r, uniform float g, uniform float b, uniform float a) {
    uniform Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    c.a = a;
    return c;
}

Color makeVaryingColor(float r, float g, float b, float a) {
    Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    c.a = a;
    return c;
}

void test_uniform_member_access() {
    // Test member access on uniform function call returns
    uniform float ux = makeUniformPoint(1.0, 2.0).x;
    uniform float uy = makeUniformPoint(3.0, 4.0).y;

    uniform float ur = makeUniformColor(1.0, 0.5, 0.25, 1.0).r;
    uniform float ug = makeUniformColor(0.8, 0.6, 0.4, 0.9).g;
    uniform float ub = makeUniformColor(0.7, 0.3, 0.1, 0.8).b;
    uniform float ua = makeUniformColor(0.9, 0.8, 0.7, 0.6).a;
}

void test_varying_member_access() {
    // Test member access on varying function call returns
    float vx = makeVaryingPoint(programIndex, programIndex + 1).x;
    float vy = makeVaryingPoint(programIndex + 2, programIndex + 3).y;

    float vr = makeVaryingColor(programIndex * 0.1, 0.5, 0.25, 1.0).r;
    float vg = makeVaryingColor(0.8, programIndex * 0.1, 0.4, 0.9).g;
    float vb = makeVaryingColor(0.7, 0.3, programIndex * 0.1, 0.8).b;
    float va = makeVaryingColor(0.9, 0.8, 0.7, programIndex * 0.1).a;
}

void test_mixed_uniform_varying() {
    // Test mixing uniform and varying member access
    uniform float uniform_x = makeUniformPoint(5.0, 6.0).x;
    float varying_y = makeVaryingPoint(uniform_x, programIndex).y;

    // Test using uniform result in varying context
    float mixed_result = makeUniformColor(1.0, 0.5, 0.0, 1.0).r + programIndex * 0.1;

    // Test using varying result in expressions
    uniform float base = makeUniformPoint(10.0, 20.0).x;
    float computed = base + makeVaryingPoint(programIndex, programIndex * 2).y;
}

void test_conditional_member_access() {
    // Test member access in conditional contexts
    if (programIndex % 2 == 0) {
        float even_x = makeVaryingPoint(programIndex, 0).x;
        uniform float uniform_y = makeUniformPoint(100.0, 200.0).y;
    } else {
        float odd_y = makeVaryingPoint(0, programIndex).y;
        uniform float uniform_x = makeUniformPoint(300.0, 400.0).x;
    }

    // Test in varying conditional with uniform access
    float result = (programIndex > 4) ? makeUniformPoint(1.0, 2.0).x : makeUniformPoint(3.0, 4.0).y;
}