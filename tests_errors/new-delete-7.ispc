// Can't convert from type "varying float" to type "uniform float" for initializer
// rule: skip on arch=genx32
// rule: skip on arch=genx64

struct Point {
    uniform float x, y, z;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Point * uniform buf = uniform new uniform Point(a, b, 1234.);
    RET[programIndex] = buf->y;
    delete buf;
}
