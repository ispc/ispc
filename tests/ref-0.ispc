float foo(varying float * uniform a) {
    *a = 0;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    foo(&a);
    RET[programIndex] = a;
}

export void result(uniform float RET[]) { RET[programIndex] = 0; }
