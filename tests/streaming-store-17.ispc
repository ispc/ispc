
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform unsigned int8 x = 5;
    uniform unsigned int8 a[1];
    streaming_store(a, x);
    RET[programIndex] = (a[0] == x) ? 1. : 0.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
