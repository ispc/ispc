#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    int tmp1 = shuffle(programIndex, 0, programIndex);

    RET[programIndex] = 10;
    if (programIndex < 1) {
        uniform int foo = extract(tmp1, 0);
        RET[programIndex] = aFOO[foo + programIndex];
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10;
    RET[0] = 1;
}
