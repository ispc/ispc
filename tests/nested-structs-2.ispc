
export uniform int width() { return programCount; }



struct Foo {
    float f[18];
};

struct Bar {
    uniform Foo foo[6];
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Bar bar = 2.;
    bar.foo[5].f[a] = a;
    RET[programIndex] = bar.foo[b].f[a];
}

export void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }

