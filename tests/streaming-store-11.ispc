
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float x = 5.;
    uniform float a[1];
    uniform float d, result;
    streaming_store(a, x);
    result = a[0];
    d = (result - x);
    d = (d < 0) ? -d : d;
    RET[programIndex] = (d < 1e-7) ? 1. : 0.;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1.;
}
