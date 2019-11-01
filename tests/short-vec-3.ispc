typedef int<3> int3;


export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    int3 v0 = { b, 2*b, 3*b };
    int3 v1 = { a, 2*a, 3*a };
    ret[programIndex] = (v0+v1).y;
}

export void result(uniform float ret[]) {
    ret[programIndex] = 10 + 2 * (programIndex+1);
}
