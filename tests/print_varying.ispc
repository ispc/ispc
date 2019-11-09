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
             float  varFloatVal   =  a + 4.75;
    // varying 64 bit values
             int64  varLongVal    = -(int64)intA - 10000000000;
    unsigned int64  varULongVal   =  (int64)intA + 10000000000;
             double varDoubleVal  =  a + 13.25;

    print("Test varying: small ones: %,", varCharVal);
    print(" %, %,", varUCharVal, varShortVal);
    print(" %, %,", varUShortVal, varIntVal);
    print(" %, %;", varUIntVal, varFloatVal);
    print(" big ones: %,", varLongVal);
    print(" %,", varULongVal);
    print(" %.\n", varDoubleVal);
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

    // varFloatVal
    print("], [5.750000,6.750000,7.750000,8.750000");
    if(programCount > 4)
        print(",9.750000,10.750000,11.750000,12.750000");
    if(programCount > 8)
        print(",13.750000,14.750000,15.750000,16.750000,17.750000,18.750000,19.750000,20.750000");

    // varLongVal
    print("]; big ones: [-10000000001,-10000000002,-10000000003,-10000000004");
    if(programCount > 4)
        print(",-10000000005,-10000000006,-10000000007,-10000000008");
    if(programCount > 8)
        print(",-10000000009,-10000000010,-10000000011,-10000000012,-10000000013,-10000000014,-10000000015,-10000000016");

    // varULongVal
    print("], [10000000001,10000000002,10000000003,10000000004");
    if(programCount > 4)
        print(",10000000005,10000000006,10000000007,10000000008");
    if(programCount > 8)
        print(",10000000009,10000000010,10000000011,10000000012,10000000013,10000000014,10000000015,10000000016");

    // varDoubleVar
    print("], [14.250000,15.250000,16.250000,17.250000");
    if(programCount > 4)
        print(",18.250000,19.250000,20.250000,21.250000");
    if(programCount > 8)
        print(",22.250000,23.250000,24.250000,25.250000,26.250000,27.250000,28.250000,29.250000");
    print("].\n");
}
