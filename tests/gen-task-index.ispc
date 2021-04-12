// rule: skip on arch=*
// rule: run on arch=genx32
// rule: run on arch=genx64
// TODO: the test should be reworked when atomic operation on gen are available.
// It is hard to check taskIndex now.

export void f_t(uniform float RET[]) {
    RET[taskIndex] = taskIndex;
}

export void result_t(uniform float RET[], uniform int groupSpaceWidth, uniform int groupSpaceHeight) {
    foreach(i = 0 ... taskCount) {
        RET[i] = i;
    }
}
