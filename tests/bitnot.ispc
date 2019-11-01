export void f_v(uniform float RET[]) {
    unsigned int x = 0xffff0f0f;
    #pragma ignore warning(perf)
    RET[programIndex] = ~x;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0xf0f0;
}
