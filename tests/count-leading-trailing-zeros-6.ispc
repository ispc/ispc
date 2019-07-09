export void f_f(uniform float RET[], uniform float aFOO[]) {
    int64 i = (1ull << ((programIndex * 4) % 63));
    RET[programIndex] = count_leading_zeros(i);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 63 - ((programIndex * 4) % 63);
}
