// Can't convert from varying type "float" to uniform type "uniform float" for initializer

struct Point {
    uniform float x, y, z;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform Point * uniform buf = uniform new uniform Point(a, b, 1234.);
    RET[programIndex] = buf->y;
    delete buf;
}
