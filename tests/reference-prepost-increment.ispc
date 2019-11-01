void foo(float & x) {
    if ((x) <= 2) {
        ++(x);
        x += 1;
    }
}

export void f_fu(uniform float ret[], uniform float a[], uniform float b) {
    float aa = a[programIndex];
    foo(aa);
    ret[programIndex] = aa;
}

export void result(uniform float r[]) {
    r[programIndex] = 1+programIndex;
    r[0] = 3;
    r[1] = 4;
}
