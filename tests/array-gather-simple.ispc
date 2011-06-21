
export uniform int width() { return programCount; }

struct Foo { float f; };

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float foo[5] = { 0, a, 0, 2 * a, 3 };
    uniform int offset[4] = { 1,2,3,4 };
    int x = offset[programIndex & 0x3];
    RET[programIndex] = foo[x];
}

export void result(uniform float RET[]) { 
    RET[0] = 1; RET[4] = 5; RET[8] = 9; RET[12] = 13;
    RET[1] = RET[5] = RET[9] = RET[13] = 0;
    RET[2] = 6; RET[6] = 14; RET[10] = 22; RET[14] = 30;
    RET[3] = RET[7] = RET[11] = RET[15] = 3;
}
