// rule: skip on cpu=TGLLP

export void f_fu(uniform float ret[], uniform float v[], uniform float u) {
    double d = v[programIndex];
    double r = u;
    if (d <= 2.)
        r = d;
    ret[programIndex] = r;
}

export void result(uniform float ret[]) {
    ret[programIndex] = 5;
    ret[0] = 1;
    ret[1] = 2;
}
