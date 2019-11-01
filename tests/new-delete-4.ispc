// rule: skip on arch=genx32
// rule: skip on arch=genx64

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    float * uniform buf = uniform new float(2*b);
    RET[programIndex] = buf[0];
    delete[] buf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
