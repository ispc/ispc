int b = { 2. }; 

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float aa = { a };
    RET[programIndex] = aa+b; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3 + 1*programIndex;
}
