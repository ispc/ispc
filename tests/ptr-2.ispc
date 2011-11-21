
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    int a = aFOO[programIndex];
    int * uniform b = &a;
    RET[programIndex] = *b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
