#include "test_static.isph"

struct S {
    uint a;
};

struct S operator&(struct S rr, struct S rv) {
    struct S c;
    c.a = rr.a & rv.a;
    return c;
}

struct S operator|(struct S rr, struct S rv) {
    struct S c;
    c.a = rr.a | rv.a;
    return c;
}

struct S operator&&(struct S rr, struct S rv) {
    struct S c;
    c.a = rr.a && rv.a;
    return c;
}

struct S operator||(struct S rr, struct S rv) {
    struct S c;
    c.a = rr.a || rv.a;
    return c;
}

struct S operator^(struct S rr, struct S rv) {
    struct S c;
    c.a = rr.a ^ rv.a;
    return c;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    struct S a;
    struct S b;
    struct S c;

    a.a = aFOO[programIndex];
    b.a = aFOO[programIndex] + 1;
    c.a = aFOO[programIndex] + 2;
    struct S d;
    if (programIndex < 3)
        d = a & b | c;
    else
        d.a = 0;

    RET[programIndex] = reduce_add(d.a);
}

task void result(uniform float RET[16]) {
    RET[programIndex] = 14;
}