#include "test_static.isph"

struct S {
    int a;
};

// Unary minus (-x)
struct S operator-(struct S rs) {
    struct S result;
    result.a = -rs.a;
    return result;
}

// Unary minus for reference (-x)
struct S operator-(struct S &rs) {
    struct S result;
    result.a = -rs.a - 10; // Different behavior for reference to distinguish
    return result;
}

// Logical NOT (!x)
bool operator!(struct S rs) { return rs.a == 0; }

// Logical NOT for reference (!x)
bool operator!(struct S &rs) { return rs.a <= 0; } // Different behavior for reference

// Bitwise NOT (~x)
struct S operator~(struct S rs) {
    struct S result;
    result.a = ~(rs.a);
    return result;
}

// Bitwise NOT for reference (~x)
struct S operator~(struct S &rs) {
    struct S result;
    result.a = ~(rs.a) + 5; // Different behavior for reference to distinguish
    return result;
}

// New task to test operators with references
task void f_f(uniform float RET[], uniform float aFOO[]) {
    struct S a;
    struct S b;

    a.a = -1;
    b.a = aFOO[programIndex];

    // Create references
    struct S &refA = a;
    struct S &refB = b;

    if (!refA) {       // Using overloaded ! operator for reference
        S neg = -refB; // Using overloaded - operator for reference
        RET[programIndex] = neg.a;
        S bitNot = ~refB; // Using overloaded ~ operator for reference
        RET[programIndex] += bitNot.a;
    } else {
        RET[programIndex] = 0;
    }
}

task void result(uniform float RET[4]) { RET[programIndex] = -8 - 2 * programIndex; }
