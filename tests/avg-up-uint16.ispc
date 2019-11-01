export void f_fu(uniform float RET[], uniform float aFOO[], uniform float bf) {
    #pragma ignore warning(perf)
    unsigned int16 a = aFOO[programIndex];
    unsigned int16 b = bf;
    #pragma ignore warning(perf)
    RET[programIndex] = avg_up(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = ((int)programIndex + 1 + 5 + 1) / 2;
}
