

export uniform int width() { return programCount; }

struct Foo {
    float f;
    uniform int a;
    float y;
};

extern void aa(Foo * uniform f);
extern void bb(Foo f[]);

void set(Foo f[], int offset, Foo val) {
    f[offset] = val;
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    Foo foo[35];
    for (uniform int i = 0; i < 35; ++i) {
        foo[i].f = a;
        foo[i].a = i;
        foo[i].y = 2*a;
    }

    Foo x = { 3*a, -1, 4*a };
    set(foo, a-1, x);

    RET[programIndex] = foo[programIndex].f; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3+3*programIndex;
}

