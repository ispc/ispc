export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = sizeof(uniform int32); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
