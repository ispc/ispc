export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = count_leading_zeros((int32)0xf0);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 24;
}
