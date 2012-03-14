
struct Point { float x, y, z; };

struct Foo {
    int x;
    Point *pts[3];
    int8 z;
};

export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Foo * uniform pts = uniform new soa<8> Foo[4];
//CO    uniform Foo pts[32];
    foreach (i = 0 ... 32) {
        pts[i].x = b*i;
        pts[i].z = -b*i;
        for (uniform int j = 0; j < 3; ++j) {
            pts[i].pts[j] = new uniform Point[4];
            for (uniform int k = 0; k < 4; ++k) {
                pts[i].pts[j][k].x = 100*i+10*j+k;
                pts[i].pts[j][k].y = -1234;
                pts[i].pts[j][k].z = -(100*i+10*j+k);
            }
        }
    }
    
    RET[programIndex] = pts[programIndex].pts[programIndex % 3][programIndex % 4].z;
}

export void result(uniform float RET[]) {
    RET[programIndex] = -(100*programIndex+10*(programIndex % 3)+(programIndex % 4));
}
