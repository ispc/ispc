typedef int<4> int4;


export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    int4 v0 = { b, 2*b, 3*b, 2 };
    int4 v1 = { a, 2*a, 3*a, 1 };
    #pragma ignore warning(perf)
    v0.y /= 2.;
    if (programIndex & 1)
        v1[0] += 1;
    ret[programIndex] = (v0+v1).y + (v0+v1).x;
}

export void result(uniform float ret[]) {
    ret[programIndex] = 5 + 2 * (programIndex+1) + 5 + (programIndex+1);
    if (programIndex & 1) ++ret[programIndex];
}
