
export uniform int width() { return programCount; }



struct Bar {
    uniform float x;
};

struct Foo {
    uniform float x[2];
    uniform Bar bar;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Foo f[3] = { { { b, b }, {b+10} }, { { 2*b, 2*b }, {b+20} }, { { 3*b, 3*b }, {b+30} } };
    varying Foo g = f[0];
    RET[programIndex] = g.x[0];
}

export void result(uniform float RET[]) { RET[programIndex] = 5; }
