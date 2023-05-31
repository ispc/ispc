#include "../test_static.isph"

float Neg(float inVal) {
    return -inVal;
}

void test_vary(uniform float RET[]) {
    float val;
    unsigned int signBit;
    float retval;
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

task void f_v(uniform float RET[]) {
    test_vary(RET);
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
