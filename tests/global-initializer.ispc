#include "../test_static.isph"
/*COextern "C" float f;*/
uniform float f = 2.;

task void f_v(uniform float RET[]) {
    RET[programIndex] = f;
}

task void result(uniform float RET[]) { RET[programIndex] = 2.; }
