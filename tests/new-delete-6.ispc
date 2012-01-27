
export uniform int width() { return programCount; }

struct Point {
    float x, y, z;
};

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    Point * varying buf = new Point(0., b, a);
    RET[programIndex] = buf->z;
    delete buf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
