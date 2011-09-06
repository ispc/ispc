
export uniform int width() { return programCount; }



export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    RET[programIndex] = popcnt((int)a);
}

export void result(uniform float RET[]) { 
    uniform int pc[16] = { 1, 1, 2, 1,
                           2, 2, 3, 1,
                           2, 2, 3, 2, 
                           3, 3, 4, 1 };
    RET[programIndex] = pc[programIndex];
}

