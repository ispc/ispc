struct Foo {
    float a;
    int8 b;
};

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = sizeof(Foo); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 8*programCount;
}
