struct Foo {
    float f[6];
};

struct Bar {
    uniform Foo foo[6];
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Bar bar;
    for (uniform int i = 0; i < 6; ++i)
        for (uniform int j = 0; j < 6; ++j)
            bar.foo[i].f[j] = 2.+b-5;
    RET[programIndex] = bar.foo[b].f[b];
}

export void result(uniform float RET[]) { RET[programIndex] = 2; }
