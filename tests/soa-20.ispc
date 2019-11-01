struct Point { float x, y, z; };


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[10];
    foreach (i = 0 ... 80) {
        pts[i].x = b*i;
        pts[i].y = 2*b*i;
        pts[i].z = 3*b*i;
    }

    assert(6+programIndex < 80);
    soa<8> Point * ptr = &pts[6+programIndex];
    RET[programIndex] = ptr - pts;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 6 + programIndex;
}
