
struct Point { float x, y[3], z; };

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

//CO    soa<8> Point pts[10];
    uniform Point pts[80];
    foreach (i = 0 ... 80) {
        pts[i].x = b*i;
        pts[i].y[0] = 2*b*i;
        pts[i].y[1] = 2*b*i+1;
        pts[i].y[2] = 2*b*i+2;
        pts[i].z = 3*b*i;
    }

    a *= -1;
    Point vp = { a, { 2*a, 3*a, 4*a }, {5*a} };
    pts[2+programIndex] = vp;

    RET[programIndex] = pts[programIndex].y[2];
}

export void result(uniform float RET[]) {
    RET[programIndex] = -4 * (programIndex-1);
    RET[0] = 2;
    RET[1] = 12;
}
