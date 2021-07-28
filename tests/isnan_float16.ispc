#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_v(uniform float RET[]) {
    float errors = 0;
    for (uniform int i = -2; i <= 2; ++i) {
        float16 f = log((float16)i);

        if ((i < 0) && (!isnan(f)))
            errors++;
        if ((i >= 0) && isnan(f))
            errors++;
    }

    for (uniform int i = -2; i <= 2; ++i) {
        uniform float16 f = log((uniform float16)i);

        if ((i < 0) && (!isnan(f)))
            errors++;
        if ((i >= 0) && isnan(f))
            errors++;
    }
    RET[programIndex] = errors;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
