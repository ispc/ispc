export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    a *= 1k;
    RET[programIndex] = a; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1024*(programIndex+1);
}
