#ifdef __NVPTX__
uniform int _off[programCount];
#define off _off[programIndex]
#else  /* global varying data types are not yet supported with "nvptx" target */
int off;
#endif

export uniform int width() { return programCount; }

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

#ifdef __NVPTX__
uniform struct S _a[programCount];
uniform struct S _b[programCount];
uniform struct S _d[programCount];
#define global_a _a[programIndex]
#define global_b _b[programIndex]
#define global_d _d[programIndex]
#else
struct S global_a;
struct S global_b;
struct S d;
#endif

export void f_f(uniform float RET[], uniform float aFOO[]) {
    int T = programIndex;
    global_a.a = aFOO[programIndex];
    global_b.a = -aFOO[programIndex];
    if (programIndex == 3)
        off = 1;
    else
        off = 0;
    if (T % 2)
        global_d = global_a + global_b;
    else
        global_d = global_a / global_b;

    RET[programIndex] = global_d.a;
}

export void result(uniform float RET[4]) {
    if (programIndex % 2)
        RET[programIndex] = 2;
    else
        RET[programIndex] = 10;
    RET[3] = 22;
}
