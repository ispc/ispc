export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    if (a == 2) {
        unmasked {
            a = 0;
        }
    }
    RET[programIndex] = a; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
