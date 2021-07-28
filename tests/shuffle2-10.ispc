#include "../test_static.isph"
/* failing due to llvm bug http://llvm.org/bugs/show_bug.cgi?id=10421 */


task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int8 aa = aFOO[programIndex]; 
    int8 bb = aa + programCount;
    int8 shuf = shuffle(aa, bb, 2*programIndex+(int)b-5);
//CO    print("%\n%\n%\n%\n", aa, bb, 2*programIndex+(int)b-5, shuf);
    RET[programIndex] = shuf;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + 2*programIndex;
}
