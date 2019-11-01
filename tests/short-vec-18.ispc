void inc(uniform float<4> * uniform v) { *v=2; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float<4> x[2] = { {b,b+1,b+2,b+3}, {1,2,3,4} };

    inc(&x[0]);
    RET[programIndex] = 0;
    if (programIndex < 4) {
        // TODO: we need to optimize ispc to not throw out-of-bound warning here.
        #pragma ignore warning
        RET[programIndex] = x[1][programIndex];
    }
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 1;
    RET[1] = 2;
    RET[2] = 3;
    RET[3] = 4;
}
