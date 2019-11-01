struct Foo {
    uniform float x;
    uniform float f;
};
export void f_fi(uniform float RET[], uniform float a[], uniform int b[]) {
    uniform struct Foo myFoo = { 1, 2 };
    RET[programIndex] = myFoo.x;
}


export void result(uniform float RET[4]) {
    RET[programIndex] = 1;
}
