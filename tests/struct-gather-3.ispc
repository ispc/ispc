struct Foo {
    float f;
};

float func(Foo foo[], int offset) {
    #pragma ignore warning(perf)
    return foo[offset].f;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Foo foo[programCount+1];
    uniform int i;
    for (i = 0; i < programCount+1; ++i)
        foo[i].f = i*a;
    RET[programIndex] = func(foo, (int)a);
}

export void result(uniform float RET[]) { RET[programIndex] = (1+programIndex)*(1+programIndex); }
