#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    float<3> a = {1,2,3};
    float<2> b = a.xy - a.uv;

    RET[programIndex] = b.x + 2*b.y;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
