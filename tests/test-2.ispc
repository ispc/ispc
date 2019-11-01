float foo(uniform float xFOO[]) { 
    float x = xFOO[programIndex];
    return x + x; 
}

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = foo(aFOO); 
}


export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
