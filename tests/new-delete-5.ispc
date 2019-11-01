// rule: skip on arch=genx32
// rule: skip on arch=genx64

struct Point {
    float x, y, z;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    varying Point * uniform buf = uniform new varying Point(a, b, 1234.);
    RET[programIndex] = buf->y;
    delete buf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 5;
}
