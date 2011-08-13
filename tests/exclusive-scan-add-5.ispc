
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    float a = aFOO[programIndex];
    if (programIndex & 1) {
        RET[programIndex] = exclusive_scan_add(a);
    }
}


export void result(uniform float RET[]) {
    uniform int result[16] = { 0, 0, 0, 2, 0, 6, 0, 12, 
                               0, 20, 0, 30, 0, 42, 0, 56 };
    if (programIndex & 1)
        RET[programIndex] = result[programIndex]; 
    else
        RET[programIndex] = -1;
}
