export void f_v(uniform float RET[]) {
    int i=0, j=0;
    do {
        ++j;
        if (i >= 5) continue;
        ++j;
        } while (++i < 10);
    RET[programIndex] = (float)j;
}


export void result(uniform float RET[]) { RET[programIndex] = 15; }
