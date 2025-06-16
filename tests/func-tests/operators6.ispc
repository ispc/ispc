#include "test_static.isph"

struct S {
    int a;
};

// Unary minus (-x)
struct S operator-(struct S rs) {
    struct S result;
    result.a = -rs.a - 2;
    return result;
}

// Unary minus for reference (-x)
struct S operator-(struct S &rs) {
    struct S result;
    result.a = -rs.a; // Different behavior for reference to distinguish
    return result;
}

// Logical NOT (!x)
bool operator!(struct S rs) { return rs.a == 0; }

// Logical NOT for reference (!x)
bool operator!(struct S &rs) { return rs.a <= 0; } // Different behavior for reference

// Bitwise NOT (~x)
struct S operator~(struct S rs) {
    struct S result;
    result.a = ~(rs.a) + 1;
    return result;
}

// Bitwise NOT for reference (~x)
struct S operator~(struct S &rs) {
    struct S result;
    result.a = ~(rs.a); // Different behavior for reference to distinguish
    return result;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    struct S a;
    struct S b;

    a.a = 0;
    b.a = aFOO[programIndex];

    if (!a) {
        RET[programIndex] = -b.a; // -(b.a), not using overloaded operator
        RET[programIndex] += (-b).a; // Using overloaded - operator
        RET[programIndex] += (~b).a; // Using overloaded ~ operator
    } else {
        RET[programIndex] = 0;
    }
}

task void result(uniform float RET[4]) { RET[programIndex] = -5 - (3 * programIndex); }
