
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform buf = uniform new uniform float[32*32];
    for (uniform int i = 0; i < 32*32; ++i)
        buf[i] = i;

    int index = (programIndex < 4) ? (programIndex & 1) :
        (programIndex / 4);
    float a = buf[index];

    RET[programIndex] = a;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programIndex < 4) ? (programIndex & 1) :
        (programIndex / 4);
}
