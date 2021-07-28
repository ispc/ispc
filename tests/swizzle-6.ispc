#include "../test_static.isph"
task void f_v(uniform float RET[]) {
    float<3> a = {1,2,3};
    float<3> b1 = a.yzx;
    float<3> b2 = b1.yzx;
    float<3> b3 = b2.yzx;
    float<3> b = b3 - a;

    RET[programIndex] = b.x + 2*b.y + 3*b.z;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
