// Regression test for member access on function call returns

// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: Error

struct Point {
    float x, y;
};

struct Complex {
    float real, imaginary;
};

struct Pair {
    int first, second;
};

Point makePoint(float x, float y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

Complex makeComplex(float r, float i) {
    Complex c;
    c.real = r;
    c.imaginary = i;
    return c;
}

Pair makePair(int a, int b) {
    Pair p;
    p.first = a;
    p.second = b;
    return p;
}

void test_original_regression_case() {
    float x_val = makePoint(1.0, 2.0).x;
    float y_val = makePoint(3.0, 4.0).y;
}

void test_multiple_struct_types() {
    // Test with different struct types to ensure the fix is general
    float real_part = makeComplex(3.14, 2.718).real;
    float imag_part = makeComplex(1.414, 1.732).imaginary;

    int first_value = makePair(42, 17).first;
    int second_value = makePair(99, 88).second;
}

void test_chained_function_calls() {
    // Test more complex scenarios
    float sum_x = makePoint(1.0, 2.0).x + makePoint(3.0, 4.0).x;
    float sum_y = makePoint(5.0, 6.0).y + makePoint(7.0, 8.0).y;

    // Test with function calls as arguments
    float distance_sq = makePoint(makeComplex(1.0, 0.0).real, makeComplex(0.0, 1.0).imaginary).x *
                       makePoint(makeComplex(1.0, 0.0).real, makeComplex(0.0, 1.0).imaginary).x +
                       makePoint(makeComplex(1.0, 0.0).real, makeComplex(0.0, 1.0).imaginary).y *
                       makePoint(makeComplex(1.0, 0.0).real, makeComplex(0.0, 1.0).imaginary).y;
}

void test_member_access_with_varying() {
    float varying_x = makePoint(programIndex, programIndex + 1).x;
    float varying_y = makePoint(programIndex * 2, programIndex * 3).y;

    Point vp = makePoint(10.0, 20.0);
    float mixed = vp.x + programIndex;
}