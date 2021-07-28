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

struct S a;
struct S b;
struct S d;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    a.a = 5;
    b.a = -5;
    d = a * b + b / a - a << (b - b) - a;
    RET[programIndex] = d.a;
}

task void result(uniform float RET[4]) {
    RET[programIndex] = 12;
}
