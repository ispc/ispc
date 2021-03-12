// rule: skip on cpu=TGLLP
// Test is unstable on TGLLP
export void print_fuf(uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform bool uniBoolVal = (bool)b;
    bool varBoolVal = (bool)a;
    if (programIndex % 3 == 0)
        varBoolVal = false;

    print("Test bool: uniform: %, %;", uniBoolVal, !uniBoolVal);
    print(" varying: %, %;", varBoolVal, !varBoolVal);

    if (a > b/2)
        print(" in simd cf: %, %.\n", varBoolVal, !varBoolVal);
}

export void print_result() {
    assert(programCount <= 64);
    print("Test bool: uniform: true, false; varying: [false,true,true,false");
    if (programCount > 4)
        print(",true,true,false,true");
    if (programCount > 8)
        print(",true,false,true,true,false,true,true,false");
    if (programCount > 16)
        print(",true,true,false,true,true,false,true,true"
              ",false,true,true,false,true,true,false,true");
    if (programCount > 32)
        print(",true,false,true,true,false,true,true,false"
              ",true,true,false,true,true,false,true,true"
              ",false,true,true,false,true,true,false,true"
              ",true,false,true,true,false,true,true,false");

    print("], [true,false,false,true");
    if (programCount > 4)
        print(",false,false,true,false");
    if (programCount > 8)
        print(",false,true,false,false,true,false,false,true");
    if (programCount > 16)
        print(",false,false,true,false,false,true,false,false"
              ",true,false,false,true,false,false,true,false");
    if (programCount > 32)
        print(",false,true,false,false,true,false,false,true"
              ",false,false,true,false,false,true,false,false"
              ",true,false,false,true,false,false,true,false"
              ",false,true,false,false,true,false,false,true");
#ifndef ISPC_TARGET_GENX
        print("]; in simd cf: [_________,_________,true,false");
#else
        print("]; in simd cf: [((false)),((true)),true,false");
#endif

    if (programCount > 4)
        print(",true,true,false,true");
    if (programCount > 8)
        print(",true,false,true,true,false,true,true,false");
    if (programCount > 16)
        print(",true,true,false,true,true,false,true,true"
              ",false,true,true,false,true,true,false,true");
    if (programCount > 32)
        print(",true,false,true,true,false,true,true,false"
              ",true,true,false,true,true,false,true,true"
              ",false,true,true,false,true,true,false,true"
              ",true,false,true,true,false,true,true,false");
#ifndef ISPC_TARGET_GENX
    print("], [_________,_________,false,true");
#else
    print("], [((true)),((false)),false,true");
#endif
    if (programCount > 4)
        print(",false,false,true,false");
    if (programCount > 8)
        print(",false,true,false,false,true,false,false,true");
    if (programCount > 16)
        print(",false,false,true,false,false,true,false,false"
              ",true,false,false,true,false,false,true,false");
    if (programCount > 32)
        print(",false,true,false,false,true,false,false,true"
              ",false,false,true,false,false,true,false,false"
              ",true,false,false,true,false,false,true,false"
              ",false,true,false,false,true,false,false,true");
    print("].\n");
}
