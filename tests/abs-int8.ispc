#include "../test_static.isph"
task void f_v(uniform float RET[]) {

    RET[programIndex] = 0;
    // abs() for int8 varying -ve value.
    int8 in_var = -4;
    int8 out_var = 4;
    in_var = abs(in_var);
    if (in_var != out_var)
         RET[programIndex] = 1;

    // abs() for int8 varying +ve value.
    in_var = 2;
    out_var = 2;
    in_var = abs(in_var);
    if (in_var != out_var)
         RET[programIndex] = 1;

    // abs() for int8 uniform -ve value.
    uniform int8 in_uni = -3;
    uniform int8 out_uni = 3;
    in_uni = abs(in_uni);
    if (in_uni != out_uni)
         RET[programIndex] = 1;

    // abs() for int8 uniform +ve value.
    in_uni = 5;
    out_uni = 5;
    in_uni = abs(in_uni);
    if (in_uni != out_uni)
         RET[programIndex] = 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
