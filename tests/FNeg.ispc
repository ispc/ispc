uniform float Neg(uniform float inVal) {
    return -inVal;
}

void test_uni(uniform float RET[]) {
    uniform float val;
    unsigned int signBit;
    uniform float retval;
    RET[programIndex] = 0;

    val = 0.0;
    signBit = signbits(val);
    if (signBit != 0)
        RET[programIndex] = 1;
    retval = Neg(val);
    signBit = signbits(retval);
    if (signBit == 0)
        RET[programIndex] = 1;
 
    val = -0.0;
    signBit = signbits(val);
    if (signBit == 0)
        RET[programIndex] = 1;
    retval = Neg(val);
    signBit = signbits(retval);
    if (signBit != 0)
        RET[programIndex] = 1;

}

float Neg(float inVal) {
    return -inVal;
}


void test_vary(uniform float RET[]) {
    double val;
    unsigned int signBit;
    double retval;
    RET[programIndex] = 0;

    val = 0.0;
    signBit = signbits(val);
    if (signBit != 0)
        RET[programIndex] = 1;
    retval = Neg(val);
    signBit = signbits(retval);
    if (signBit == 0)
        RET[programIndex] = 1;

    val = -0.0;
    signBit = signbits(val);
    if (signBit == 0)
        RET[programIndex] = 1;
    retval = Neg(val);
    signBit = signbits(retval);
    if (signBit != 0)
        RET[programIndex] = 1;

}

export void f_v(uniform float RET[]) {

    test_vary(RET);
    test_uni(RET);

}

export void result(uniform float RET[]) { RET[programIndex] = 0; }

