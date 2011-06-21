
export uniform int width() { return programCount; }


export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex&3]; 
    RET[programIndex] = log(a); 
}

export void result(uniform float RET[]) {
    RET[0] = RET[4] = RET[8] = RET[12] = 0x0.0p+0;
    RET[1] = RET[5] = RET[9] = RET[13] = 0x1.62e43p-1;
    RET[2] = RET[6] = RET[10] = RET[14] =  0x1.193ea8p+0;
    RET[3] = RET[7] = RET[11] = RET[15] = 0x1.62e43p+0;
}
