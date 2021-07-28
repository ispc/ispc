#include "../test_static.isph"
void update(uniform float<2> * varying vp) {
    #pragma ignore warning(perf)
    vp->y = 0;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float<2> v[programCount];
    for (uniform int i = 0; i < programCount; ++i) {
        v[i].x = 2*i;
        v[i].y = 2*i+1;
    }
    
    int index = aFOO[programIndex] - 1;
    update(&v[programIndex]);
//CO    for (uniform int i = 0; i < programCount; ++i) 
//CO        print("%: % %\n", i, v[i].x, v[i].y);
    #pragma ignore warning(perf)
    RET[programIndex] = v[programIndex].x + v[programIndex].y;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2*programIndex;
}
