
export uniform int width() { return programCount; }

void inc(reference float v) { ++v; }

export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    uniform float foo[32];
    for (uniform int i = 0; i < 32; ++i)
        foo[i] = 10;
    int a = (int)aa[programIndex];
    inc(foo[a]);
    ret[programIndex] = foo[programIndex];
}

export void result(uniform float ret[]) {
    ret[programIndex] = 11;
    ret[0] = 10;
}
