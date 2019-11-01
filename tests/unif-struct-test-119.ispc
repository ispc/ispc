struct Foo {
    float x;
    float f;
    int i[3];
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform struct Foo myFoo[3] = { { b, b, {b, b, b} }, 
                                    { b, b, {b, b, b} }, 
                                    { b, b, {b, b, b} } };
    uniform struct Foo barFoo;
    barFoo = myFoo[0];
    RET[programIndex] = barFoo.x + myFoo[1].i[2];
}


export void result(uniform float RET[4]) {
    RET[programIndex] = 10;
}
