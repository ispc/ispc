#include "../test_static.isph"
// rule: skip on arch=*
// rule: run on arch=xe64

task void f_t(uniform float RET[]) {
    uniform unsigned int a = taskCount/16;
    uniform unsigned int end = a * taskIndex+1;
    foreach (k = 0 ... end) {
        RET[k] = 0;
    }
}

task void result_t(uniform float RET[], uniform int groupSpaceWidth, uniform int groupSpaceHeight) {
    foreach(i = 0 ... taskCount) {
        RET[i] = 0;
    }
}
