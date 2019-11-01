struct Foo {
    uniform float x;
    uniform float f;
};

export void f_fi(uniform float RET[], uniform float a[], uniform int b[]) {
    uniform struct Foo myFoo = {2,2};
    RET[programIndex] = myFoo.x + myFoo.f;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
