// Test basic struct member access on function call returns
// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: Error

struct Point {
    float x, y;
};

struct Vector3 {
    float x, y, z;
};

Point makePoint(float x, float y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

Vector3 makeVector3(float x, float y, float z) {
    Vector3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

void test_basic_member_access() {
    // Basic member access on function call return
    float x_val = makePoint(1.0, 2.0).x;
    float y_val = makePoint(3.0, 4.0).y;
}

void test_multiple_members() {
    // Access multiple members from same function call
    Vector3 v = makeVector3(1.0, 2.0, 3.0);
    float x = v.x;
    float y = v.y;
    float z = v.z;

    // Also test direct member access on function call
    float direct_x = makeVector3(4.0, 5.0, 6.0).x;
    float direct_y = makeVector3(7.0, 8.0, 9.0).y;
    float direct_z = makeVector3(10.0, 11.0, 12.0).z;
}

float test_member_access_in_expressions() {
    // Test member access in arithmetic expressions
    return makePoint(1.0, 2.0).x + makePoint(3.0, 4.0).y +
           makeVector3(2.0, 3.0, 4.0).x * makeVector3(5.0, 6.0, 7.0).y;
}