enum {
    ZERO,
    ONE,
    TWO
};


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RET[programIndex] = ONE; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
