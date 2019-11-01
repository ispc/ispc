void foo(varying float * uniform x, float y) {
    *x = y;
}

export void f_fu(uniform float ret[], uniform float a[], uniform float b) {
    float aa = a[programIndex], bb = b;
    foo(&aa, bb);
    ret[programIndex] = aa;
}

export void result(uniform float r[]) {
    r[programIndex] = 5;
}
