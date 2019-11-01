export void f_f(uniform float RET[], uniform float aFOO[]) {
    float sum = 0;
    foreach_tiled (i = 0 ... 6)
        sum += aFOO[i];
    RET[programIndex] = reduce_add(sum);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 21;
}
