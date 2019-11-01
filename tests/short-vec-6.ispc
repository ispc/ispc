typedef int<4> int4;


void inc(varying int4 * uniform v) {
    int4 delta = { 1, 1, 1, 1 };
    (*v) += delta;
}

export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    int4 v0 = { b, 2*b, 3*b, 2 };
    inc(&v0);
    if (programIndex & 1) inc(&v0);
    ret[programIndex] = v0.z;
}

export void result(uniform float ret[]) {
    ret[programIndex] = 16;
    if (programIndex & 1) ++ret[programIndex];
}
