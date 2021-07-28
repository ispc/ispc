#include "../test_static.isph"
struct Vector { float x, y, z; };

struct Ray { Vector o; float t; Vector d; };


void init(varying Ray rays[], uniform int count, float v) {
    for (uniform int i = 0; i < count; ++i) {
        rays[i].o.x = programIndex;
        rays[i].o.y = v;
        rays[i].o.z = -1234;
        rays[i].t = 42;
        rays[i].d.x = 2*v;
        rays[i].d.y = 3*v;
        rays[i].d.z = 4*v;
    }
}

void zero_dx(Ray &r) {
    r.d.x = 0;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    Ray rays[10];
    init(rays, 10, v);
    zero_dx(rays[b]);
    RET[programIndex] = rays[5].d.x + rays[5].d.y; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3 + 3*programIndex;
}
