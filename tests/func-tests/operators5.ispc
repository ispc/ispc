#include "test_static.isph"

struct S {
    int a;
};

// Prefix increment (++x)
struct S& operator++(struct S &rs) {
    rs.a = rs.a + 1;
    return rs;
}

// Prefix decrement (--x)
struct S& operator--(struct S &rs) {
    rs.a = rs.a - 1;
    return rs;
}

// Postfix increment (x++)
struct S operator++(struct S &rs, int) {
    struct S temp;
    temp.a = rs.a;
    rs.a = rs.a + 1;
    return temp;
}

// Postfix decrement (x--)
struct S operator--(struct S &rs, int) {
    struct S temp;
    temp.a = rs.a;
    rs.a = rs.a - 1;
    return temp;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    struct S a;
    struct S c;

    a.a = aFOO[programIndex];
    c.a = aFOO[programIndex];

    // Test prefix increment
    ++a;
    RET[programIndex] = a.a;

    // Test prefix decrement
    --a;
    RET[programIndex] += a.a;

    // Test postfix increment
    c = a++;
    RET[programIndex] += a.a - c.a; // 1

    // Test postfix decrement
    c = a--;
    RET[programIndex] += c.a - a.a; // 1
}

task void result(uniform float RET[16]) { RET[programIndex] = programIndex * 2 + 5; }
