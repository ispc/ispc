#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_v(uniform float RET[]) {
    int errors = 0;
    for (uniform int i = -2; i <= 2; ++i) {
        double f = log((double)i);
        
        if ((i < 0) && (!isnan(f))) errors ++;
        if ((i >= 0) && isnan(f))   errors ++;
    }

    for (uniform int i = -2; i <= 2; ++i) {
        uniform double f = log((uniform double)i);
        
        if ((i < 0) && (!isnan(f))) errors ++;
        if ((i >= 0) && isnan(f))   errors ++;
    }

    RET[programIndex] = errors;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
