
export uniform int width() { return programCount; }


struct Foo {
    float x;
    float f;
};
export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo myFoo[3] = a;
    int i = 1;
    varying Foo barFoo = myFoo[i];
    RET[programIndex] = barFoo.x;
}


export void result(uniform float RET[4]) {
    RET[programIndex] = 1+programIndex;
}
