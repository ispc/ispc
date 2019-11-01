struct Foo {
    varying int i;
    varying float f;
};
export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform struct Foo myFoo;
    myFoo.f = a;
    RET[programIndex] = myFoo.f + a;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
