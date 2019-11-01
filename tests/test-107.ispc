export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float o = a / (sqrt(a) * sqrt(a));
    RET[programIndex] = ((o > .9999) && (o < 1.00001));
}


export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
