#include "test_static.isph"

struct S {
    uint a;
};

uint operator==(struct S rr, struct S rv) {
    return rr.a == rv.a;
}

uint operator!=(struct S rr, struct S rv) {
    return rr.a != rv.a;
}

uint operator>(struct S rr, struct S rv) {
    return rr.a > rv.a;
}

uint operator<(struct S rr, struct S rv) {
    return rr.a < rv.a;
}

uint operator>=(struct S rr, struct S rv) {
    return rr.a >= rv.a;
}

uint operator<=(struct S rr, struct S rv) {
    return rr.a <= rv.a;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    struct S a;
    struct S b;

    a.a = aFOO[programIndex];
    b.a = aFOO[programIndex] + 5;

    RET[programIndex] = (a == b) + (a != b) + (a > b) + (a < b) + (a >= b) + (a <= b);
}

task void result(uniform float RET[16]) {
    RET[programIndex] = 3;
}