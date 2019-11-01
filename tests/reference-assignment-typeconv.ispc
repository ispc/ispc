void foo(float &x, int y) {
    x = y;
}

export void f_fu(uniform float ret[], uniform float a[], uniform float b) {
    float aa = a[programIndex];
    int bb = (int)b;
    foo(aa, bb);
    ret[programIndex] = aa;
}

export void result(uniform float r[]) {
    r[programIndex] = 5;
}
