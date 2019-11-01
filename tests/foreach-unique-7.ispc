export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = aFOO[programIndex]; 

    foreach_unique (p in programIndex & 1)
        RET[programIndex] = p;
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex & 1;
}
