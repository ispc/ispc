int switchit(int a, uniform int b) {
    int r = 0;
    switch (a) {
    case 3:
        return 1;
    default:
        return 0;
    }
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int x = switchit(a, b);
    RET[programIndex] = x; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programIndex == 2) ? 1 : 0;
}
