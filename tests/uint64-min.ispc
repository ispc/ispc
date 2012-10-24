
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    unsigned int64 a = aFOO[programIndex];
    unsigned int64 b = 2. * a;
    RET[programIndex] = min(a,b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
