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
    print("Test bool: uniform: true, false; varying: [false,true,true,false");
    if (programCount > 4)
        print(",true,true,false,true");
    if (programCount > 8)
        print(",true,false,true,true,false,true,true,false");

    print("], [true,false,false,true");
    if (programCount > 4)
        print(",false,false,true,false");
    if (programCount > 8)
        print(",false,true,false,false,true,false,false,true");

    print("]; in simd cf: [_________,_________,true,false");
    if (programCount > 4)
        print(",true,true,false,true");
    if (programCount > 8)
        print(",true,false,true,true,false,true,true,false");

    print("], [_________,_________,false,true");
    if (programCount > 4)
        print(",false,false,true,false");
    if (programCount > 8)
        print(",false,true,false,false,true,false,false,true");
    print("].\n");
}
