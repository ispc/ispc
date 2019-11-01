struct Foo {
    float x, y;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo f[3] = { { b, b }, { 2*b, 2*b }, { 3*b, 3*b } };
    int index = (a <= 2) ? 1 : 2;
    #pragma ignore warning(perf)
    varying Foo g = f[index];
    RET[programIndex] = g.x;
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 15;
    RET[0] = RET[1] = 10;
}
