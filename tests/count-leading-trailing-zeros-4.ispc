
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 i = (1 << programIndex);
    RET[programIndex] = count_leading_zeros(i);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 31-programIndex;
}
