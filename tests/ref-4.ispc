void foo(float & a) {
    a = 0;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    if (a >= 3)
        foo(a);
    RET[programIndex] = a;
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 0;
    RET[0] = 1;
    RET[1] = 2;
}
