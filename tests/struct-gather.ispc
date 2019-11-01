struct Foo {
    float f;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Foo foo[programCount+1];
    uniform int i;
    for (i = 0; i < programCount+1; ++i)
        foo[i].f = i*a;
    #pragma ignore warning(perf)
    RET[programIndex] = foo[(int)a].f;
}

export void result(uniform float RET[]) { RET[programIndex] = (programIndex+1)*(programIndex+1); }
