#include "../test_static.isph"
// rule: skip on arch=*
// rule: run on arch=xe64
// TODO: the test should be reworked when atomic operation on Xe are available.
// It is hard to check taskIndex now.

task void f_t(uniform float RET[]) {
    RET[taskIndex] = taskIndex;
}

task void result_t(uniform float RET[], uniform int groupSpaceWidth, uniform int groupSpaceHeight) {
    foreach(i = 0 ... taskCount) {
        RET[i] = i;
    }
}
