
export uniform int width() { return programCount; }


struct Foo {
    uniform float x;
    uniform float f;
};
export void f_fi(uniform float RET[], uniform float a[], uniform int bFOO[]) {
    int b = bFOO[programIndex];
    uniform struct Foo myFoo[17];
    uniform int i;
    for (i = 0; i < 17; ++i) {
        myFoo[i].x = i;
        myFoo[i].f = 17+2*i;
    }
    RET[programIndex] = myFoo[b/2].f;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 19 + 2 * programIndex;
}
