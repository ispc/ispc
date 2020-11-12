export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    if (a < 3) {
        if (all(a < 256)) 
            RET[programIndex] = 1;
        else {
#ifndef ISPC_TARGET_GENX
            print("FALSE %\n", a);
            print("programCount %\n", programCount);
#endif
        }
    }
    else RET[programIndex] = 0;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = RET[1] = 1;
}
