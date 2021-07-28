#include "../test_static.isph"
struct Vector { float x, y, z; };

struct Ray { Vector o; float t; Vector d; };


void init(uniform Ray rays[], uniform int count, float v) {
    for (uniform int i = 0; i < count; ++i) {
        rays[i].o.x = i;
        rays[i].o.y = 2*i;
        rays[i].o.z = 3*i;
        rays[i].t = 4*i;
        rays[i].d.x = 5*i;
        rays[i].d.y = 6*i;
        rays[i].d.z = 7*i;
    }
}

void zero_dx(Ray &r) {
    r.d.x = 0;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform Ray rays[programCount+1];
    init(rays, programCount+1, v);
    #pragma ignore warning(perf)
    Ray rg = rays[v];
    RET[programIndex] = rg.o.z;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3 + 3*programIndex;
}
