export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 10., t = 2., c;
    cif (a <= t) {
        if (a == 1.) { RET[programIndex] = 7.; return; }
        else { RET[programIndex] = a; return; }
    }
    else {
        c = b;
        if (a == 3.) { RET[programIndex] = 12; return; }
    }
    RET[programIndex] = c;
}


export void result(uniform float RET[]) {
    RET[programIndex] = 10;
    RET[0] = 7;
    RET[1] = 2;
    RET[2] = 12;
}
