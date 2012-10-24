
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    float * uniform buf = uniform new float(2*b);
    RET[programIndex] = buf[0];
    delete[] buf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
