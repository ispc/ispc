struct Foo {
    float x;
    float f;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    varying Foo myFoo = { a, a };
    RET[programIndex] = myFoo.x;
}


export void result(uniform float RET[4]) {
    RET[programIndex] = 1+programIndex;
}
