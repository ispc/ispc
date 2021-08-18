#include "../test_static.isph"
// rule: skip on arch=*
// rule: run on arch=xe32
// rule: run on arch=xe64

task void f_t(uniform float RET[]) {
    foreach(i = 0 ... taskCount) {
        RET[i] = 0;
    }
    RET[0] = taskCount;
    RET[1] = taskCount0;
    RET[2] = taskCount1;
    RET[3] = taskCount2;
}

task void result_t(uniform float RET[], uniform int groupSpaceWidth, uniform int groupSpaceHeight) {
    foreach(i = 0 ... taskCount) {
        RET[i] = 0;
    }
    RET[0] = groupSpaceWidth*groupSpaceHeight;
    RET[1] = groupSpaceWidth;
    RET[2] = groupSpaceHeight;
    RET[3] = 1;
}
