#include "../test_static.isph"
task void print_f(uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int intA = a;
    int varIntVal = -intA - 2;
    unsigned int varUIntVal = intA + 2;
    print("Test varying int32: %, %\n", varIntVal, varUIntVal);
}

task void print_result() {
    assert(programCount <= 64);
    print("Test varying int32: ");
    // varIntVal
    print("[-3,-4,-5,-6");
    if(programCount > 4)
        print(",-7,-8,-9,-10");
    if(programCount > 8)
        print(",-11,-12,-13,-14,-15,-16,-17,-18");
    if(programCount > 16)
        print(",-19,-20,-21,-22,-23,-24,-25,-26"
              ",-27,-28,-29,-30,-31,-32,-33,-34");
    if(programCount > 32)
        print(",-35,-36,-37,-38,-39,-40,-41,-42"
              ",-43,-44,-45,-46,-47,-48,-49,-50"
              ",-51,-52,-53,-54,-55,-56,-57,-58"
              ",-59,-60,-61,-62,-63,-64,-65,-66");

    // varUIntVal
    print("], [3,4,5,6");
    if(programCount > 4)
        print(",7,8,9,10");
    if(programCount > 8)
        print(",11,12,13,14,15,16,17,18");
    if(programCount > 16)
        print(",19,20,21,22,23,24,25,26"
              ",27,28,29,30,31,32,33,34");
    if(programCount > 32)
        print(",35,36,37,38,39,40,41,42"
              ",43,44,45,46,47,48,49,50"
              ",51,52,53,54,55,56,57,58"
              ",59,60,61,62,63,64,65,66");

    print("]\n");
}
