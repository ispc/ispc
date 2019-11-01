export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = b + 2M;
    RET[programIndex] = a;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 2ul*1024ul*1024ul + 5;
}
