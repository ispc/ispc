#include "../test_static.isph"
struct S {
    float a;
};

// References "struct&" were put in random order to test them.
struct S operator*(struct S& rr, struct S rv) {
    struct S c;
    c.a = rr.a + rv.a + 2;
    return c;
}

struct S operator/(struct S& rr, struct S& rv) {
    struct S c;
    c.a = rr.a - rr.a + 2;
    return c;
}

struct S operator%(struct S rr, struct S& rv) {
    struct S c;
    c.a = rr.a + rv.a + 2;
    return c;
}

struct S operator+(struct S rr, struct S rv) {
    struct S c;
    c.a = rr.a / rv.a + 3;
    return c;
}

struct S operator-(struct S rr, struct S& rv) {
    struct S c;
    c.a = rr.a + rv.a + 2;
    return c;
}

struct S operator>>(struct S& rr, struct S rv) {
    struct S c;
    c.a = rr.a / rv.a + 3;
    return c;
}

struct S operator<<(struct S& rr, struct S& rv) {
    struct S c;
    c.a = rr.a + rv.a + 2;
    return c;
}

struct S a, a1;
struct S b, b1;
struct S d1, d2, d3, d4, d5, d6, d7;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    a.a = aFOO[programIndex];
    b.a = -aFOO[programIndex];
    d1 = a * b;
    d2 = a / b;
    d3 = a % b;
    d4 = a + b;
    d5 = a - b;
    d6 = a >> b;
    d7 = a << b;
    RET[programIndex] = d1.a + d2.a + d3.a + d4.a + d5.a + d6.a + d7.a;
}

task void result(uniform float RET[4]) {
    RET[programIndex] = 14;
}
