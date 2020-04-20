export void print_f(uniform float aFOO[]) {
    double a = aFOO[programIndex];
    int intA = a;

    // varying max 32 bit values
             int8   varCharVal    = -intA;
    unsigned int8   varUCharVal   =  intA;
             int16  varShortVal   = -intA - 1;
    unsigned int16  varUShortVal  =  intA + 1;
             int    varIntVal     = -intA - 2;
    unsigned int    varUIntVal    =  intA + 2;
    print("Test varying: small ones: %, %, %, %, %, %\n",
        varCharVal, varUCharVal, varShortVal, varUShortVal,
        varIntVal, varUIntVal);
}

export void print_result() {
    // varCharVal
    print("Test varying: small ones: [-1,-2,-3,-4");
    if(programCount > 4)
        print(",-5,-6,-7,-8");
    if(programCount > 8)
        print(",-9,-10,-11,-12,-13,-14,-15,-16");

    // varUCharVal
    print("], [1,2,3,4");
    if(programCount > 4)
        print(",5,6,7,8");
    if(programCount > 8)
        print(",9,10,11,12,13,14,15,16");

    // varShortVal
    print("], [-2,-3,-4,-5");
    if(programCount > 4)
        print(",-6,-7,-8,-9");
    if(programCount > 8)
        print(",-10,-11,-12,-13,-14,-15,-16,-17");

    // varUShortVal
    print("], [2,3,4,5");
    if(programCount > 4)
        print(",6,7,8,9");
    if(programCount > 8)
        print(",10,11,12,13,14,15,16,17");

    // varShortVal
    print("], [-3,-4,-5,-6");
    if(programCount > 4)
        print(",-7,-8,-9,-10");
    if(programCount > 8)
        print(",-11,-12,-13,-14,-15,-16,-17,-18");

    // varUShortVal
    print("], [3,4,5,6");
    if(programCount > 4)
        print(",7,8,9,10");
    if(programCount > 8)
        print(",11,12,13,14,15,16,17,18");

    print("]\n");
}
