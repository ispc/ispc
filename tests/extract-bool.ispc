export void f_f(uniform float RET[], uniform float aFOO[]) {
    bool a = (((programIndex + 1) % programCount) == 0);
    for (uniform int i = 0; i < programCount; i++) {
        uniform bool val = extract(a, i);
        RET[i] = val ? -1 : -2;
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = -2;
    if (((programIndex + 1) % programCount) == 0)
        RET[programIndex] = -1;
}
