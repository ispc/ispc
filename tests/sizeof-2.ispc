export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = sizeof(uniform double); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 8;
}
