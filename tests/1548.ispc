export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 0;
    uniform float f = sqrt(-1.0f);
    if (f < 0.0f)
        RET[programIndex] = 1;

    if (f > 0.0f)
        RET[programIndex] = 1;

    if (f <= 0.0f)
        RET[programIndex] = 1;

    if (f >= 0.0f)
        RET[programIndex] = 1;

    if (f == f)
        RET[programIndex] = 1;

    if (!(f != f))
        RET[programIndex] = 1;
}

export void result(uniform float RET[]) { RET[programIndex] = 0; }
