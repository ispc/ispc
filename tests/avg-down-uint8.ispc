export void f_fu(uniform float RET[], uniform float aFOO[], uniform float bf) {
    #pragma ignore warning(perf)
    unsigned int8 a = aFOO[programIndex];
    unsigned int8 b = bf;
    #pragma ignore warning(perf)
    RET[programIndex] = avg_down(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = ((int)programIndex + 1 + 5) / 2;
}
