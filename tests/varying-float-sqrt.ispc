export void f_f(uniform float RET[], uniform float aFOO[]) {
    varying float f = programIndex;
    varying float calc = sqrt(f) * sqrt(f) - programIndex;
    if (calc < 0.) calc = -calc;
    if (calc < 1e-5)
    {
        RET[programIndex] = 1;
    }
    else
    {
        RET[programIndex] = 0;
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
