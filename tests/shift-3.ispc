export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int rot = shift(a, 1);
    RET[programIndex] = rot;
}

export void result(uniform float RET[]) {
    varying int val = 2 + programIndex;
    if (val > programCount) val = 0;	 
    RET[programIndex] = val;	 
}   
