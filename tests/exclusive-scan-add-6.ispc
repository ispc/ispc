
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = exclusive_scan_add((float)programIndex); 
}

export void result(uniform float RET[]) {
    uniform int result[] = { 0, 0, 1, 3, 6, 10, 15, 21, 28,
                             36, 45, 55, 66, 78, 91, 105, 120 };
    RET[programIndex] = result[programIndex]; 
}
