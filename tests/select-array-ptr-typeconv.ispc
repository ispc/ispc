uniform float a[1234];

float * uniform func(uniform bool x) {
    return x ? a : NULL;
}


export void f_f(uniform float RET[], uniform float aFOO[]) {
    a[programIndex] = aFOO[programIndex];
    float * uniform ptr = func(aFOO[0] == 1);
    RET[programIndex] = ptr[programIndex]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
