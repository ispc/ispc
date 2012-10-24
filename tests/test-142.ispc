
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = round(a+.499999); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
