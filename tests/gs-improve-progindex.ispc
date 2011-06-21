
export uniform int width() { return programCount; }



export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform float udx[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int x = programIndex;
    RET[programIndex] = udx[x];
}

export void result(uniform float RET[]) { 
    RET[programIndex] = programIndex + 1;
}
