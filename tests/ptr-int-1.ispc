
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a = 1;
    uniform int * uniform b = &a;
    int64 pi = (int64)b;
    RET[programIndex] = *((uniform int * varying)pi);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
