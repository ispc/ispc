#include "../test_static.isph"
task void print_fuf(uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform bool uniBoolVal = (bool)b;
    bool varBoolVal = (bool)a;
    if (programIndex % 3 == 0)
        varBoolVal = false;

    print("Test bool: uniform: %, %;", uniBoolVal, !uniBoolVal);
    print(" varying: %, %\n", varBoolVal, !varBoolVal);
}

task void print_result() {
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
    print("]\n");
}
