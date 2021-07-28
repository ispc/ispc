#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    RET[programIndex] = 0;
    float16 check = 0;
    float16 input = aFOO[programIndex];
    ;
    float16 rot = 0;

    // Case 1
    rot = shift(input, -1);
    check = programIndex;
    if (check < 0)
        check = 0;
    if (check != rot)
        RET[programIndex] = -1;

    // Case 2
    rot = shift(input, 1);
    check = 2 + programIndex;
    if (check > programCount)
        check = 0;
    if (check != rot)
        RET[programIndex] = -1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
