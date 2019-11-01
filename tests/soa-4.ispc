struct Point { float x, y, z; };


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Point pts[10];
    foreach (i = 0 ... 80) {
        pts[i].x = 0;
        pts[i].y = 0;
        pts[i].z = 0;
    }

    Point pv = { a, b, -a };
    assert(8+programCount < 80);
    pts[8+programIndex] = pv;

    RET[programIndex] = pts[8+programIndex].z;
}

export void result(uniform float RET[]) {
    RET[programIndex] = -(1 + programIndex);
}
