struct Foo {
    uniform int16 a;
    uniform unsigned int16 c;
};

export void f_f(uniform float RET[], uniform float aFOO[]) {
    Foo a[10];
    RET[programIndex] = sizeof(a);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 40;
}
