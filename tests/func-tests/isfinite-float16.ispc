#include "test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

template <typename T>
int test_isfinite()
{
    // Constants assumed to be converted to the right type and variability
    const uniform double neg_inf = doublebits(0xFFF0000000000000);
    const uniform double pos_inf = doublebits(0x7FF0000000000000);
    const uniform double nan_val1 = doublebits(0xFFF8000000000000);
    const uniform double nan_val2 = doublebits(0xFFF8000000000001);
    const uniform double nan_val3 = doublebits(0x7FF8000000000000);

    int errors = 0;
    if (isfinite((T)neg_inf)) errors++;
    if (isfinite((T)pos_inf)) errors++;
    if (isfinite((T)nan_val1)) errors++;
    if (isfinite((T)nan_val2)) errors++;
    if (isfinite((T)nan_val3)) errors++;
    if (!isfinite((T)0)) errors++;
    if (!isfinite((T)-0)) errors++;
    if (!isfinite((T)42)) errors++;
    if (!isfinite((T)-42)) errors++;
    return errors;
}

task void f_v(uniform float RET[]) {
    int errors = 0;
    errors += test_isfinite<uniform float16>() + test_isfinite<float16>();
    RET[programIndex] = errors;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
