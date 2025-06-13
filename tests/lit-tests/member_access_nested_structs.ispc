// Test nested struct member access on function call returns

// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: Error

struct Point {
    float x, y;
};

struct Rectangle {
    Point topLeft;
    Point bottomRight;
};

struct Transform {
    Point translation;
    float rotation;
    Point scale;
};

Rectangle makeRectangle(float x1, float y1, float x2, float y2) {
    Rectangle rect;
    rect.topLeft.x = x1;
    rect.topLeft.y = y1;
    rect.bottomRight.x = x2;
    rect.bottomRight.y = y2;
    return rect;
}

Transform makeTransform(float tx, float ty, float rot, float sx, float sy) {
    Transform t;
    t.translation.x = tx;
    t.translation.y = ty;
    t.rotation = rot;
    t.scale.x = sx;
    t.scale.y = sy;
    return t;
}

void test_nested_member_access() {
    // Test nested member access on function call returns
    float top_x = makeRectangle(0.0, 0.0, 10.0, 10.0).topLeft.x;
    float top_y = makeRectangle(1.0, 2.0, 11.0, 12.0).topLeft.y;
    float bottom_x = makeRectangle(5.0, 6.0, 15.0, 16.0).bottomRight.x;
    float bottom_y = makeRectangle(7.0, 8.0, 17.0, 18.0).bottomRight.y;
}

void test_deep_nested_access() {
    // Test accessing deeply nested members
    float trans_x = makeTransform(1.0, 2.0, 45.0, 2.0, 2.0).translation.x;
    float trans_y = makeTransform(3.0, 4.0, 90.0, 1.5, 1.5).translation.y;
    float scale_x = makeTransform(5.0, 6.0, 180.0, 3.0, 3.0).scale.x;
    float scale_y = makeTransform(7.0, 8.0, 270.0, 0.5, 0.5).scale.y;
    float rot = makeTransform(9.0, 10.0, 360.0, 1.0, 1.0).rotation;
}

float test_nested_in_expressions() {
    // Test nested member access in arithmetic expressions
    return makeRectangle(0.0, 0.0, 10.0, 5.0).bottomRight.x - makeRectangle(0.0, 0.0, 10.0, 5.0).topLeft.x +
           makeRectangle(2.0, 3.0, 12.0, 8.0).bottomRight.y - makeRectangle(2.0, 3.0, 12.0, 8.0).topLeft.y;
}