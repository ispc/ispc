export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int64 a = aFOO[programIndex];
    if (a <= 2)
        RET[programIndex] = exclusive_scan_add(a);
}

export void result(uniform float RET[]) {
    uniform int result[] = { 0, 1, 3, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0 };
    RET[programIndex] = -1;
    if (programIndex <= 1)
        RET[programIndex] = result[programIndex]; 
}
