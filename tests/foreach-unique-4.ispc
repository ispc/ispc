export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int count = 0;
    int a = aFOO[programIndex];
    foreach_unique (ua in a & 1)
        ++count;
    RET[programIndex] = count; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2;
}
