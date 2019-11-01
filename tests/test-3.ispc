float foo(float x) {
    return x + x; 
}


export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b; 
    b = foo(a); 
    RET[programIndex] = b; 
}


export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
