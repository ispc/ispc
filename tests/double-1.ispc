// rule: skip on cpu=TGLLP

export void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    RET[programIndex] = a / b;
}

export void result(uniform float RET[]) { RET[programIndex] = (1.+programIndex)/5.;}
