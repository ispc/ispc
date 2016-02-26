
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    int x = programIndex;
    uniform int a[programCount];
    streaming_store(a, x);
    RET[programIndex] = (a[programIndex] == x) ? 1. : 0.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
