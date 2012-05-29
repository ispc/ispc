
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int32 a = (1 << (min(programIndex, 30)));
    RET[programIndex] = exclusive_scan_or(a);
}


export void result(uniform float RET[]) {
    RET[programIndex] = (1 << (min(programIndex, 31))) - 1;
}
