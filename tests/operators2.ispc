int off;


struct S {
    float a;
};

struct S operator+(struct S rr, struct S rv) {
    struct S c;
    c.a = rr.a / rv.a + 3;
    if (off == 1)
        c.a = 22;
    return c;
}

struct S operator/(struct S rr, struct S rv) {
    struct S c;
    c.a = rr.a + rv.a + 10;
    if (off == 1)
        c.a = 33;
    return c;
}


export void f_f(uniform float RET[], uniform float aFOO[]) {
    struct S a;
    struct S b;
    struct S d;
    int T = programIndex;
    a.a = aFOO[programIndex];
    b.a = -aFOO[programIndex];
    if (programIndex == 3)
        off = 1;
    else
        off = 0;
    #pragma ignore warning(perf)
    if (T % 2)
        d = a + b;
    else
        d = a / b;

    RET[programIndex] = d.a;
}

export void result(uniform float RET[4]) {
    if (programIndex % 2)
        RET[programIndex] = 2;
    else
        RET[programIndex] = 10;
    RET[3] = 22;
}
