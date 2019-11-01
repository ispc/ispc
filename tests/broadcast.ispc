export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = (programCount == 1) ? 4 : broadcast(a, 2);
    RET[programIndex] = b;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
