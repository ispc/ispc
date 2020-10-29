#define TEST_SIMD

export void print_f(uniform float aFOO[]) {
    float a = aFOO[programIndex];
    int intA = a;

    if (intA % 2 == 0)
      print("Test varying simd: small ones: %\n", intA);
}

export void print_result() {
    assert(programCount <= 64);
    print("Test varying simd: small ones: [((1)),2,((3)),4");
    if(programCount > 4)
        print(",((5)),6,((7)),8");
    if(programCount > 8)
        print(",((9)),10,((11)),12,((13)),14,((15)),16");
    if(programCount > 16)
        print(",((17)),18,((19)),20,((21)),22,((23)),24"
              ",((25)),26,((27)),28,((29)),30,((31)),32");
    if(programCount > 32)
        print(",((33)),34,((35)),36,((37)),38,((39)),40"
              ",((41)),42,((43)),44,((45)),46,((47)),48"
              ",((49)),50,((51)),52,((53)),54,((55)),56"
              ",((57)),58,((59)),60,((61)),62,((63)),64");
    print("]\n");
}
