// rule: skip on arch=genx32
// rule: skip on arch=genx64
struct Point { float x, y, z; };

struct Foo {
    float<3> vec;
    int8 z;
};


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 

    soa<8> Foo * uniform pts = uniform new soa<8> Foo[4];
    for (uniform int i = 0; i < 32; ++i) {
        pts[i].vec.x = b*i;
        pts[i].vec.y = -b*i;
        pts[i].vec.z = 2*b*i;
        pts[i].z = i;
    }

    RET[programIndex] = pts[9].vec.y;
}

export void result(uniform float RET[]) {
    RET[programIndex] = -45;
}
