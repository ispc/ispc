// rule: skip on arch=*
// rule: run on arch=genx32
// rule: run on arch=genx64

export void f_t(uniform float RET[]) {
    uniform unsigned int a = taskCount/16;
    uniform unsigned int end = a * taskIndex+1;
    foreach (k = 0 ... end) {
        RET[k] = 0;
    }
}

export void result_t(uniform float RET[], uniform int groupSpaceWidth, uniform int groupSpaceHeight) {
    foreach(i = 0 ... taskCount) {
        RET[i] = 0;
    }
}
