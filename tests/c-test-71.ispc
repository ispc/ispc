export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float i, r = a;
    if (a >= 4)
        { RET[programIndex] = 0; return; }
    else {
        for (i = 0; i < a; ++i) {
            if (r == b) 
                r += 10;
            ++r;
            if (r == 2) break;
        }
    }
    RET[programIndex] = r;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0]= 2;
    RET[1] = 4;
    RET[2] = 16;
}
