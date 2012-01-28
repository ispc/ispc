
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    float * uniform buf = uniform new float[programCount+1];
    for (uniform int i = 0; i < programCount+1; ++i) {
        buf[i] = i+a;
    }
    RET[programIndex] = buf[a];
    delete buf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
