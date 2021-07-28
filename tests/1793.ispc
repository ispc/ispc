#include "../test_static.isph"
uniform bool b = true;

task void f_v(uniform float RET[]) {
    uniform int i = 0;
    if (((varying int8) b) < 0) {
        // This branch may not be triggered, regardless of value of b;
        i = 1;
    }
    RET[programIndex] = i;
    return;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
