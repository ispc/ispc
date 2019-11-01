export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = 0.; b = a; 
    RET[programIndex] = 0; 
 encore:
    ++RET[programIndex];
    if (any(a != 0)) {
        a = max(a-1, 0.f);
        goto encore;
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = programCount+1; 
}
