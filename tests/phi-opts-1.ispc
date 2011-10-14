
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float sum = 0;
    for (int i = 0; i < 16; i += programCount)
        sum += aFOO[i+programIndex];
    RET[programIndex] = reduce_add(sum); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 136;
}
