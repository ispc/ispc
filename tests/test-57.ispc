float bar(float x) { return 2. * x; }
export void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = aFOO[programIndex]; 
    RET[programIndex] = extract(bar(x)  * -1., 0); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = -2.000000;
}
