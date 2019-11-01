export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform float<5> x = {1,2,3,10,20}, y = {3,1,0,0,5};
    uniform bool<5> c = x < y;

    RET[programIndex] = 0;
    if (programIndex < 5) {
        uniform float<5> z = c ? x : y;
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        RET[programIndex] = z[programIndex];
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 1;
    RET[1] = 1;
    RET[2] = 0;
    RET[3] = 0;
    if (programCount > 4)
        RET[4] = 5;
}
