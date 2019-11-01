export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 10., t = 2., c;
    if (a <= t) {
        if (a == 1.) c = 7.;
        else c = a;
    }
    else {
        c = b;
        if (a == 3.) c = 12;
    }
    RET[programIndex] = c;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 10;
    RET[0] = 7;
    RET[1] = 2;
    RET[2] = 12;
}
