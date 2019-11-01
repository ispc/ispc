export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int * varying ptr = NULL;
    RET[programIndex] = sizeof(ptr) - sizeof(uniform int * varying); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
