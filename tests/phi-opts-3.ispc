export void f_f(uniform float RET[], uniform float aFOO[]) {
    float sum = 0;
    for (int i = 0; i < 16; ++i)
        sum += aFOO[i];
    RET[programIndex] = extract(sum, 0);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 136;
}
