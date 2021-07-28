#include "../test_static.isph"
task void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    uniform int foo[10];
    for (uniform int i = 0; i < 10; ++i)
        foo[i] = 10+i;
    int bb = b;
    // Pragma included for testing running run_test.py for entire suite wothout warnings
    // Remove pragma to view warning
    #pragma ignore warning
    foo[bb] = 0;
    ret[programIndex] = foo[4] + foo[5];
}

task void result(uniform float ret[]) {
    ret[programIndex] = 14;
}
