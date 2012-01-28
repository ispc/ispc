
export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform float * uniform buf = uniform new uniform float[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        buf[i] = i;
    RET[programIndex] = buf[a-1];
    delete buf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
