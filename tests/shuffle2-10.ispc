/* failing due to llvm bug http://llvm.org/bugs/show_bug.cgi?id=10421 */


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int8 aa = aFOO[programIndex]; 
    int8 bb = aa + programCount;
    int8 shuf = shuffle(aa, bb, 2*programIndex+(int)b-5);
//CO    print("%\n%\n%\n%\n", aa, bb, 2*programIndex+(int)b-5, shuf);
    RET[programIndex] = shuf;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + 2*programIndex;
}
