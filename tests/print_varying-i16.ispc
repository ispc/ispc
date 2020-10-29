export void print_f(uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int intA = a;
    int16  varShortVal = -intA - 1;
    unsigned int16 varUShortVal = intA + 1;

    print("Test varying int16: %, %\n", varShortVal, varUShortVal);
}

export void print_result() {
    assert(programCount <= 64);
    // varCharVal
    print("Test varying int16: ");
   
    // varShortVal
    print("[-2,-3,-4,-5");
    if(programCount > 4)
        print(",-6,-7,-8,-9");
    if(programCount > 8)
        print(",-10,-11,-12,-13,-14,-15,-16,-17");
    if(programCount > 16)
        print(",-18,-19,-20,-21,-22,-23,-24,-25"
              ",-26,-27,-28,-29,-30,-31,-32,-33");
    if(programCount > 32)
        print(",-34,-35,-36,-37,-38,-39,-40,-41"
              ",-42,-43,-44,-45,-46,-47,-48,-49"
              ",-50,-51,-52,-53,-54,-55,-56,-57"
              ",-58,-59,-60,-61,-62,-63,-64,-65");

    // varUShortVal
    print("], [2,3,4,5");
    if(programCount > 4)
        print(",6,7,8,9");
    if(programCount > 8)
        print(",10,11,12,13,14,15,16,17");
    if(programCount > 16)
        print(",18,19,20,21,22,23,24,25"
              ",26,27,28,29,30,31,32,33");
    if(programCount > 32)
        print(",34,35,36,37,38,39,40,41"
              ",42,43,44,45,46,47,48,49"
              ",50,51,52,53,54,55,56,57"
              ",58,59,60,61,62,63,64,65");

    print("]\n");
}
