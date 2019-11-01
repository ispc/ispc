export void f_fu(uniform float RET[], uniform float aFOO[], uniform float bf) {
    int16 a = aFOO[programIndex];
    int16 b = bf;
    RET[programIndex] = avg_down(a, b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = ((int)programIndex + 1 + 5) / 2;
}
