export void f_v(uniform float RET[]) {

    RET[programIndex] = 0;
    // abs() for int16 varying -ve value.
    int16 in_var = -4;
    int16 out_var = 4;
    in_var = abs(in_var);
    if (in_var != out_var)
         RET[programIndex] = 1;

    // abs() for int16 varying +ve value.
    in_var = 2;
    out_var = 2;
    in_var = abs(in_var);
    if (in_var != out_var)
         RET[programIndex] = 1;

    // abs() for int16 uniform -ve value.
    uniform int16 in_uni = -3;
    uniform int16 out_uni = 3;
    in_uni = abs(in_uni);
    if (in_uni != out_uni)
         RET[programIndex] = 1;

    // abs() for int16 uniform +ve value.
    in_uni = 5;
    out_uni = 5;
    in_uni = abs(in_uni);
    if (in_uni != out_uni)
         RET[programIndex] = 1;
}

export void result(uniform float RET[]) { RET[programIndex] = 0; }
