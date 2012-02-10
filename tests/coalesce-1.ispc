
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform buf = uniform new uniform float[32*32];
    for (uniform int i = 0; i < 32*32; ++i)
        buf[i] = i;

    RET[programIndex] = buf[64-programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 64 - programIndex;
}
