// Comprehensive test for struct member access on function call returns
// This tests checks different access patterns, no assertion is expected.
// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: Error

struct Point {
    float x, y;
};

struct Rectangle {
    Point topLeft;
    Point bottomRight;
};

struct Matrix3x3 {
    float data[9];
};

struct Vertex {
    float position[3];
    float normal[3];
};

// Function that returns struct by value
Point makePoint(float x, float y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

Rectangle makeRect(float x1, float y1, float x2, float y2) {
    Rectangle r;
    r.topLeft.x = x1;
    r.topLeft.y = y1;
    r.bottomRight.x = x2;
    r.bottomRight.y = y2;
    return r;
}

Matrix3x3 makeIdentityMatrix() {
    Matrix3x3 m;
    for (uniform int i = 0; i < 9; i++) {
        m.data[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
    return m;
}

Vertex makeVertex(float px, float py, float pz, float nx, float ny, float nz) {
    Vertex v;
    v.position[0] = px;
    v.position[1] = py;
    v.position[2] = pz;
    v.normal[0] = nx;
    v.normal[1] = ny;
    v.normal[2] = nz;
    return v;
}

uniform Point makeUniformPoint(uniform float x, uniform float y) {
    uniform Point p;
    p.x = x;
    p.y = y;
    return p;
}

void test_comprehensive_member_access() {
    // ORIGINAL REGRESSION CASE: This caused assertion failure before the fix
    float x_val = makePoint(1.0, 2.0).x;
    float y_val = makePoint(3.0, 4.0).y;

    // NESTED STRUCTS: Test nested member access
    float tl_x = makeRect(0.0, 0.0, 10.0, 10.0).topLeft.x;
    float br_y = makeRect(1.0, 2.0, 11.0, 12.0).bottomRight.y;

    // ARRAY MEMBERS: Test array member access
    float m00 = makeIdentityMatrix().data[0];
    float m11 = makeIdentityMatrix().data[4];
    float m22 = makeIdentityMatrix().data[8];

    // NESTED ARRAY ACCESS: Test accessing arrays in nested structs
    float pos_x = makeVertex(1.0, 2.0, 3.0, 0.0, 1.0, 0.0).position[0];
    float norm_z = makeVertex(4.0, 5.0, 6.0, 0.0, 0.0, 1.0).normal[2];

    // EXPRESSIONS: Test member access in expressions
    float sum = makePoint(1.0, 2.0).x + makePoint(3.0, 4.0).y;
    float product = makeRect(0.0, 0.0, 5.0, 3.0).bottomRight.x * makeRect(2.0, 1.0, 7.0, 4.0).bottomRight.y;

    // UNIFORM/VARYING: Test uniform member access
    uniform float ux = makeUniformPoint(100.0, 200.0).x;
    uniform float uy = makeUniformPoint(300.0, 400.0).y;

    // VARYING CONTEXT: Test varying member access
    float vx = makePoint(programIndex, programIndex + 1).x;
    float vy = makePoint(programIndex * 2, programIndex + 2).y;

    // COMPLEX EXPRESSIONS: Test in complex scenarios
    float complex_result = makePoint(makeIdentityMatrix().data[0], makeIdentityMatrix().data[4]).x +
                          makeRect(makePoint(0.0, 0.0).x, makePoint(0.0, 0.0).y,
                                  makePoint(10.0, 10.0).x, makePoint(10.0, 10.0).y).topLeft.x;
}

// Test that this compiles without assertion failures
void test_no_assertion_failure() {
    // The original bug would cause an assertion failure here
    // Now it should compile successfully
    float result = makePoint(42.0, 3.14).x;
}