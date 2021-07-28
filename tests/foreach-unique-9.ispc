#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = aFOO[programIndex];

    foreach_unique (p in (programIndex & 3))
        switch(p) {
        case 0:
            RET[programIndex] = 15;
            break;
        case 1:
        case 2:
            RET[programIndex] = 20;
        }
}

task void result(uniform float RET[]) {
    if ((programIndex & 3) == 0)
        RET[programIndex] = 15;
    else if (((programIndex & 3) == 1) ||
             ((programIndex & 3) == 2))
        RET[programIndex] = 20;
    else
        RET[programIndex] = programIndex + 1;
}
