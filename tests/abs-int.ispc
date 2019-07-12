
export uniform int width() { return programCount; }

export void tests_int8(uniform float RET[]) {

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

export void tests_int16(uniform float RET[]) {

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

export void tests_int32(uniform float RET[]) {

    // abs() for int32 varying -ve value.
    int32 in_var = -4;
    int32 out_var = 4;
    in_var = abs(in_var);
    if (in_var != out_var)
         RET[programIndex] = 1;

    // abs() for int32 varying +ve value.
    in_var = 2;
    out_var = 2;
    in_var = abs(in_var);
    if (in_var != out_var)
         RET[programIndex] = 1;

    // abs() for int32 uniform -ve value.
    uniform int32 in_uni = -3;
    uniform int32 out_uni = 3;
    in_uni = abs(in_uni);
    if (in_uni != out_uni)
         RET[programIndex] = 1;

    // abs() for int32 uniform +ve value.
    in_uni = 5;
    out_uni = 5;
    in_uni = abs(in_uni);
    if (in_uni != out_uni)
         RET[programIndex] = 1;
}

export void tests_int64(uniform float RET[]) {

    // abs() for int64 varying -ve value.
    int64 in_var = -4;
    int64 out_var = 4;
    in_var = abs(in_var);
    if (in_var != out_var)
         RET[programIndex] = 1;

    // abs() for int64 varying +ve value.
    in_var = 2;
    out_var = 2;
    in_var = abs(in_var);
    if (in_var != out_var)
         RET[programIndex] = 1;

    // abs() for int64 uniform -ve value.
    uniform int64 in_uni = -3;
    uniform int64 out_uni = 3;
    in_uni = abs(in_uni);
    if (in_uni != out_uni)
         RET[programIndex] = 1;

    // abs() for int64 uniform +ve value.
    in_uni = 5;
    out_uni = 5;
    in_uni = abs(in_uni);
    if (in_uni != out_uni)
         RET[programIndex] = 1;
}

export void f_v(uniform float RET[]) {

    RET[programIndex] = 0;
    tests_int8(RET);
    tests_int16(RET);
    tests_int32(RET);
    tests_int64(RET);
}

export void result(uniform float RET[]) { RET[programIndex] = 0; }
