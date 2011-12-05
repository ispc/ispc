
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    unsigned int64 i = ((unsigned int64)1 << (40+programIndex));
    RET[programIndex] = count_trailing_zeros(i);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 40+programIndex;
}
