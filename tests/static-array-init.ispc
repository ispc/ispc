static uniform int foo[4] = { 1, 2, 3, 4 };



export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RET[programIndex] = foo[b-3]; 
}

export void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
