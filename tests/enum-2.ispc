enum Foo {
    ZERO,
    ONE,
    TWO,
    TEN = 10,
    ELEVEN
};


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RET[programIndex] = TEN; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
