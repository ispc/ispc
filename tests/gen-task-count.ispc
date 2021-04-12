// rule: skip on arch=*
// rule: run on arch=genx32
// rule: run on arch=genx64

export void f_t(uniform float RET[]) {
    foreach(i = 0 ... taskCount) {
        RET[i] = 0;
    }
    RET[0] = taskCount;
    RET[1] = taskCount0;
    RET[2] = taskCount1;
    RET[3] = taskCount2;
}

export void result_t(uniform float RET[], uniform int groupSpaceWidth, uniform int groupSpaceHeight) {
    foreach(i = 0 ... taskCount) {
        RET[i] = 0;
    }
    RET[0] = groupSpaceWidth*groupSpaceHeight;
    RET[1] = groupSpaceWidth;
    RET[2] = groupSpaceHeight;
    RET[3] = 1;
}
