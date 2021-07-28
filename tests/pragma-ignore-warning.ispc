#include "../test_static.isph"
#define foo(x) ((int8)x / (programIndex-1))

inline int fill(uniform float b[], int8 a) {
    for (int i=0; i < programCount; i++) {
        // Line 11 would have resulted in a performance and a compiler warning.
        // The '#pragma ignore warning(all)' directive ensures that BOTH warnings are ignored.
        #pragma ignore warning(all)
        b[i] = i;
    }
    return b[0];
}

task void f_f(uniform float RET[], uniform float fFOO[]) {
    float f = fFOO[programIndex];
    uniform float a[programCount], b[programCount];
    for (int i=0; i < programCount; i++) {
        // Line 23 would have resulted in a performance and a compiler warning.
        // The '#pragma   ignore   warning ( all )' directive ensures that BOTH warnings are ignored.
        #pragma   ignore   warning ( all )
        a[i] = i;
    }
    int x = 2;
    // Line 29 would have resulted in a performance and a compiler warning.
    // The '#pragma ignore warning(perf)' directive ignores ONLY performance warning.
    if(programIndex + 1 < programCount) {
        #pragma ignore warning
        a[programIndex + 1] = fill(b, (int8)x / (programIndex-1));
    }

    // Line 34 would have resulted in a performance warning when the macro 'foo' gets expanded.
    // The '#pragma ignore warning(perf)' directive ignores the performance warning from the expanded macro.
    #pragma ignore warning(perf)
    x = foo(x);

    // Line 39 would have resulted in a compiler warning.
    // The '#pragma ignore warning' directive ignores this warning.
    #pragma ignore warning
    a[programIndex + 1] =  b[programIndex];
    RET[programIndex] = 1;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
