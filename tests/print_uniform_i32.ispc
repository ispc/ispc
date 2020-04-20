export void print_uf(uniform float a) {
    uniform int intA = (int)a;

    // uniform max 32 bit values
    uniform          int8   uniCharVal    = -intA;
    uniform unsigned int8   uniUCharVal   =  intA;
    uniform          int16  uniShortVal   = -intA - 1;
    uniform unsigned int16  uniUShortVal  =  intA + 1;
    uniform          int    uniIntVal     = -intA - 2;
    uniform unsigned int    uniUIntVal    =  intA + 2;

    print("Test uniform: small ones: %, %, %, %, %, %\n",
        uniCharVal, uniUCharVal, uniShortVal, uniUShortVal, uniIntVal, uniUIntVal);
}

export void print_result() {
    print("Test uniform: small ones: -5, 5, -6, 6, -7, 7\n");
}
