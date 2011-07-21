
export uniform int width() { return programCount; }

export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    uniform int foo[10];
    for (uniform int i = 0; i < 10; ++i)
        foo[i] = 10+i;
    int bb = b;
    foo[bb] = 0;
    ret[programIndex] = foo[4] + foo[5];
}

export void result(uniform float ret[]) {
    ret[programIndex] = 14;
}
