export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    int i, j=0;
    for (i=a; i < 10; ++i) {
        j += sqrt((a / 3.f) * (1.f / (i+2)));
        cif (i >= 5) continue;
        j += sqrt(((a+2) / 3.f) * (1.f / (i+3)));
    }
    RET[programIndex] = (float)j; 
}


export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
