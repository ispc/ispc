// rule: skip on arch=genx32
// rule: skip on arch=genx64

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    varying float * uniform buf = uniform new varying float[programCount+1];
    for (uniform int i = 0; i < programCount+1; ++i) {
        buf[i] = i+a;
    }
    #pragma ignore warning(perf)
    RET[programIndex] = buf[a];
    delete buf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
