

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    if (programIndex % 2 == 0) {
        varying saturated int32 tmp1 = -0x80000000; // min signed int32
        RET[programIndex] = --tmp1;
    }
    else {
        varying saturated int32 tmp2 = 0x7FFFFFFF; // max signed int32
        RET[programIndex] = --tmp2;
    }  
}

export void result(uniform float RET[]) {
    if (programIndex % 2 == 0) {
        RET[programIndex] = (varying saturated int32) -0x80000000; // min signed int32
    }
    else {
        RET[programIndex] = (varying saturated int32) 0x7FFFFFFE; // max signed int32 - 1
    }
}
