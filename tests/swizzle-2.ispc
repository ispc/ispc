#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    float<3> a = {1,10,100};
    float<3> b = a.zxy;

    RET[programIndex] = b.x + 2*b.y + 3*b.z;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 132;
}
