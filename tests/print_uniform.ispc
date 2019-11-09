export void print_uf(uniform float a) {
    uniform int intA = (int)a;

    // uniform max 32 bit values
    uniform          int8   uniCharVal    = -intA;
    uniform unsigned int8   uniUCharVal   =  intA;
    uniform          int16  uniShortVal   = -intA - 1;
    uniform unsigned int16  uniUShortVal  =  intA + 1;
    uniform          int    uniIntVal     = -intA - 2;
    uniform unsigned int    uniUIntVal    =  intA + 2;
    uniform          float  uniFloatVal   =  a + 4.75;
    // uniform 64 bit values
    uniform          int64  uniLongVal    = -(int64)intA - 10000000000;
    uniform unsigned int64  uniULongVal   =  (int64)intA + 10000000000;
    uniform          double uniDoubleVal  =  a + 13.25;

    print("Test uniform: small ones: %, %, %, %, %, %, %; big ones: %, %, %.\n",
        uniCharVal, uniUCharVal, uniShortVal, uniUShortVal, uniIntVal, uniUIntVal,
        uniFloatVal, uniLongVal, uniULongVal, uniDoubleVal);
}

export void print_result() {
    print("Test uniform: small ones: -5, 5, -6, 6, -7, 7, 9.750000; big ones: -10000000005, 10000000005, 18.250000.\n");
}
