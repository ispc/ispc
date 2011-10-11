

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    int index = programIndex + 5;
    RET[index-b] = a; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
