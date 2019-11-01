typedef int<3> int3;


export void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    int3 v0 = { b, 2*b, 3*b };
    int3 v1 = { a, 2*a, 3*a };
    int3 v = v0+v1;
    ret[programIndex] = v.z; // (v0+v1)[2];
}

export void result(uniform float ret[]) {
    ret[programIndex] = 15 + 3 * (programIndex+1);
}
