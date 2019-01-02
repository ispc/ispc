
struct Point { double x; double y[3], z; };

export uniform int width() { return programCount; }

static void p(uniform float *uniform ptr) {
    for (uniform int s = 0; s < 4; ++s) {
        for (uniform int i = 0; i < 5; ++i) {
            for (uniform int j = 0; j < 4; ++j, ++ptr)
                print("% ", *ptr);
            print("\n");
        }
        print("\n");
    }
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
#ifdef __NVPTX__  /* soa is converted to shared memory story for now, use smaller amount to check the test */
    soa<4> Point pts[10];
    #pragma ignore warning(perf)
    for (uniform int i = 0; i < 40; ++i) {
        pts[i].x = b*i;
        pts[i].y[0] = 2*b*i;
        pts[i].y[1] = 2*b*i+1;
        pts[i].y[2] = 2*b*i+2;
        pts[i].z = 3*b*i;
    }
#else
    #pragma ignore warning(perf)
    soa<4> Point pts[30];
    for (uniform int i = 0; i < 120; ++i) {
        pts[i].x = b*i;
        pts[i].y[0] = 2*b*i;
        pts[i].y[1] = 2*b*i+1;
        pts[i].y[2] = 2*b*i+2;
        pts[i].z = 3*b*i;
    }
#endif

    float a = aFOO[programIndex]; 
    a *= -1;
    Point vp = { a, { 2*a, 3*a, 4*a }, {5*a} };
    assert(programCount + 2 < 120);
    #pragma ignore warning(perf)
    pts[2+programIndex] = vp;

    #pragma ignore warning(perf)
    RET[programIndex] = pts[programIndex].y[2];
}

export void result(uniform float RET[]) {
    RET[programIndex] = -4 * (programIndex-1);
    RET[0] = 2;
    RET[1] = 12;
}
