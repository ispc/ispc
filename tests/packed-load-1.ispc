
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a[programCount];
    a[programIndex] = aFOO[programIndex];
    int aa;
    packed_load_active(a, 0, aa);
    RET[programIndex] = aa;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
