struct Point { float x, y, z; };


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[20];

    foreach (i = b-5 ... 160) {
        #pragma ignore warning(perf)
        pts[i].x = b*i;
        #pragma ignore warning(perf)
        pts[i].y = 2*b*i;
        #pragma ignore warning(perf)
        pts[i].z = 3*b*i;
    }

    uniform Point up = pts[4];

    assert(2*programCount < 160);
    #pragma ignore warning(perf)
    RET[programIndex] = pts[2*programIndex].x;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 10 * programIndex;
}
