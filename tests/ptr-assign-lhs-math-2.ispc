export void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float a[programCount];
    a[programIndex] = aFOO[programIndex];

    uniform float * varying ptr = a;
    *(ptr+programIndex) = 0;
    RET[programIndex] = a[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
