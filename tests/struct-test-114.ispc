struct Foo {
    uniform float x;
    uniform float f;
};

export void f_fi(uniform float RET[], uniform float aFOO[], uniform int bFOO[]) {
    float a = aFOO[programIndex];
    int b = bFOO[programIndex];
    varying Foo myFoo[128];
    uniform int i;
    for (i = 0; i < 128; ++i) {
        myFoo[i].x = i;
        myFoo[i].f = 2*i;
    }
    assert(b/2 < 128);
    #pragma ignore warning(perf)
    RET[programIndex] = myFoo[b/2].f;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
