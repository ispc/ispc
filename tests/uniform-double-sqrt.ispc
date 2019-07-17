export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform double f = aFOO[programCount/2];
    uniform double calc = sqrt(f) * sqrt(f) - aFOO[programCount/2];
    if (calc < 0.) calc = -calc;
    RET[programIndex] = calc; //(calc < 1e-24) ? 1 : 0;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
