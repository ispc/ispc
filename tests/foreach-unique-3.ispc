export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int count = 0;
    float a = aFOO[programIndex];
    uniform float sum = 0;
    foreach_unique (ua in a)
        sum += ua;
    RET[programIndex] = sum; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = reduce_add(programIndex+1);
}
