struct Foo {
    uniform float a;
    uniform int8 b;
    uniform float c;
};

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = sizeof(Foo); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 12;
}
