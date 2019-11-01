export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    uniform int delta = b - 6; // -1
    int rot = rotate(a, delta);
    RET[programIndex] = rot;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + (programIndex + programCount - 1) % programCount;
}
