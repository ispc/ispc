
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    unsigned int16 x = programIndex;
    uniform unsigned int16 a[programCount];
    streaming_store(a, x);
    RET[programIndex] = (a[programIndex] == x) ? 1. : 0.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
