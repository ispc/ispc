export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;
    if (programIndex >= 1) {
        RET[programIndex] = packmask(programIndex<3); 
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 6;
    RET[0] = 0;
}

