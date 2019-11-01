uniform bool b = true;

export void f_v(uniform float RET[]) {
    uniform int i = 0;
    if (((varying int8) b) < 0) {
        // This branch may not be triggered, regardless of value of b;
        i = 1;
    }
    RET[programIndex] = i;
    return;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
