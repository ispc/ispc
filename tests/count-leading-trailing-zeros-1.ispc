export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = count_trailing_zeros(0xf0ul);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
