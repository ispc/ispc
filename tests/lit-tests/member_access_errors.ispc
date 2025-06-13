// Test that member access errors are properly detected.
// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

struct Point {
    float x, y;
};

struct Color {
    float r, g, b;
};

struct Rectangle {
    Point topLeft;
    Point bottomRight;
};

struct PointArray {
    Point points[2];
};

Point makePoint(float x, float y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

Color makeColor(float r, float g, float b) {
    Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    return c;
}

Rectangle makeRectangle(float x1, float y1, float x2, float y2) {
    Rectangle r;
    r.topLeft = makePoint(x1, y1);
    r.bottomRight = makePoint(x2, y2);
    return r;
}

PointArray makePointArray() {
    PointArray pa;
    pa.points[0] = makePoint(1.0, 2.0);
    pa.points[1] = makePoint(3.0, 4.0);
    return pa;
}

int makeInt() {
    return 42;
}

void modifyPoint(Point& p) {
    p.x = 999.0;
    p.y = 999.0;
}

void test_nonexistent_member() {
    // CHECK: has no member named "z"
    float bad_z = makePoint(1.0, 2.0).z;
}

void test_wrong_member_name() {
    // CHECK: has no member named "red"
    float bad_red = makeColor(1.0, 0.5, 0.0).red;
}

void test_member_access_on_non_struct() {
    // CHECK: Member operator "." can't be used with expression of
    float bad_access = makeInt().x;
}

// New tests for lvalue restrictions on function returns

void test_cannot_take_address_of_function_return() {
    // CHECK: Illegal to take address of non-lvalue or function
    Point *ptr1 = &makePoint(1.0, 2.0);
}

void test_cannot_take_address_of_nested_member() {
    // CHECK: Illegal to take address of non-lvalue or function
    float *ptr2 = &makeRectangle(0, 0, 10, 10).topLeft.x;
}

void test_cannot_take_address_of_array_member() {
    // CHECK: Illegal to take address of non-lvalue or function
    float *ptr3 = &makePointArray().points[0].x;
}

void test_cannot_take_address_of_varying_function_return() {
    // CHECK: Illegal to take address of non-lvalue or function
    varying float *vptr = &makePoint(programIndex, programIndex).x;
}

void test_cannot_assign_to_function_return_member() {
    // CHECK: Left hand side of assignment expression can't be assigned to
    makePoint(1.0, 2.0).x = 5.0;
}

void test_cannot_assign_to_nested_member() {
    // CHECK: Left hand side of assignment expression can't be assigned to
    makeRectangle(0, 0, 10, 10).topLeft.x = 42.0;
}

void test_cannot_assign_to_array_member() {
    // CHECK: Left hand side of assignment expression can't be assigned to
    makePointArray().points[0].x = 99.0;
}
