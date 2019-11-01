struct Foo {
    float x;
    float f;
};

export void f_fi(uniform float RET[], uniform float aFOO[], uniform int bFOO[]) {
    float a = aFOO[programIndex];
    int b = bFOO[programIndex];
    varying Foo myFoo = { a, b };
    RET[programIndex] = myFoo.x + myFoo.f;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 3 + 3 * programIndex;
}
