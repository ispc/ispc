// rule: skip on arch=genx32
// rule: skip on arch=genx64

struct Point {
    float x, y, z;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    varying Point * buf = new varying Point(0., b, a);
    #pragma ignore warning(perf)
    RET[programIndex] = buf->z;
    delete buf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
