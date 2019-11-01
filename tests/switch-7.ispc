int switchit(int a, uniform int b) {
    switch (a) {
    case 3:
        return 1;
    case 7:
    case 6:
    case 4:
    case 5:
        if (a & 1)
            break;
        return 2;
    default:
        return 0;
    }
    return 3;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int x = switchit(a, b);
    RET[programIndex] = x; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[2] = 1;
    RET[6] = RET[4] = 3;
    RET[5] = RET[3] = 2;
}
