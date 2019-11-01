export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform int count = 0;
    if (programIndex & 1)
        foreach_active (i)
            ++count;
    RET[programIndex] = count; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = programCount / 2;
}
