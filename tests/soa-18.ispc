struct Point { float x, y, z; };


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[10];
    foreach (i = 0 ... 80) {
        pts[i].x = b*i;
        pts[i].y = 2*b*i;
        pts[i].z = 3*b*i;
    }
    
    soa<8> Point * ptr = &pts[programIndex];
    ++ptr;
    #pragma ignore warning(perf)
    ptr->y = -programIndex;

    assert(1+programCount < 80);
    #pragma ignore warning(perf)
    RET[programIndex] = pts[1+programIndex].y;
}

export void result(uniform float RET[]) {
    RET[programIndex] = -programIndex;
}
