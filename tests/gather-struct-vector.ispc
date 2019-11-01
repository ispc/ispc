struct Ray {
    float<3> v;
};



export void f_f(uniform float RET[], uniform float aFOO[]) {
    Ray r[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        r[i].v.x = 100*i + programIndex;
        r[i].v.y = 200*i + 2*programIndex;
        r[i].v.z = 300*i + 3*programIndex;
    }

    varying Ray *rp = &r[programIndex/2];
    #pragma ignore warning(perf)
    RET[programIndex] = rp->v.z; 
}

export void result(uniform float RET[]) {
    uniform int d0 = 0;
    uniform int d1 = 0;
    for (uniform int i = 0; i < programCount; i += 2) {
        RET[i] = d0+d1;
        d1 += 3;
        RET[i+1] = d0+d1;
        d0 += 300;
        d1 += 3;
    }
}
