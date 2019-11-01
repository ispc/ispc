int switchit(int a, uniform int b) {
    switch (b) {
    case 5:
        if (a & 1)
            break;
        return 2;
    default:
        return 42;
    }
    return 3;
}

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int a = aFOO[programIndex]; 
    int x = switchit(a, b);
    RET[programIndex] = x; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 2 : 3;
}
