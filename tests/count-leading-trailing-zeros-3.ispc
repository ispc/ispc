export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = count_leading_zeros((unsigned int64)0xf0);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 56;
}
