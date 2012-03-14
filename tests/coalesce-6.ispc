
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform buf = uniform new uniform float[32*32];
    for (uniform int i = 0; i < 32*32; ++i)
        buf[i] = i;

    float a = buf[4*programIndex];
    float b = buf[4*programIndex+1];
    buf[4*programIndex+2] = 0;
    float c = buf[4*programIndex+2];
    float d = buf[4*programIndex+3];

    RET[programIndex] = a+b+c+d;;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4 * programIndex + 4 * programIndex + 1 + 
        4 * programIndex + 3;
}
