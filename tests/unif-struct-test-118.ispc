struct Foo {
    float x;
    float f;
};
export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    struct Foo myFoo[3] = { { a,a}, {a,a}, {a,a} };
    struct Foo barFoo;
    barFoo = myFoo[1];
    RET[programIndex] = barFoo.x;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
