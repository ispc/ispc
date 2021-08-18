#include "../test_static.isph"
// rule: skip on arch=xe32
// rule: skip on arch=xe64
int64 zero = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
 uniform int64 a[programCount];
 for (uniform int i = 0; i < programCount; ++i)
 a[i] = aFOO[i];

 int64 *ptr = &(a[programIndex+zero]);
 prefetch_l1(ptr);
 prefetch_l2(ptr);
 prefetch_l3(ptr);
 prefetch_nt(ptr);
 #pragma ignore warning(perf)
 int g = *ptr;
 RET[programIndex] = g;
}

task void result(uniform float RET[]) {
 RET[programIndex] = 1 + programIndex;
}
