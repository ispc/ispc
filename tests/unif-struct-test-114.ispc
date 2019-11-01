struct Foo {
    uniform float x;
    uniform float f;
};
export void f_fi(uniform float RET[], uniform float a[], uniform int bFOO[]) {
    int b = bFOO[programIndex];
    uniform struct Foo myFoo[256];
    uniform int i;
    for (i = 0; i < 256; ++i) {
        myFoo[i].x = i;
        myFoo[i].f = 17+2*i;
    }
    assert(b/2 < 256);
    #pragma ignore warning(perf)
    RET[programIndex] = myFoo[b/2].f;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 19 + 2 * programIndex;
}
