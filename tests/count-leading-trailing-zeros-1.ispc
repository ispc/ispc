
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = count_trailing_zeros(0xf0);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
