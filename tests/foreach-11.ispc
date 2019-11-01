export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float val[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        val[i] = 0;

    foreach_tiled (i = 0 ... programCount)
        val[i] += aFOO[i] - 1;

    uniform float sum = 0;
    for (uniform int i = 0; i < programCount; ++i)
        sum += val[i];

    RET[programIndex] = sum; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = reduce_add(programIndex);
}
