#include "../test_static.isph"
task void print_uf(uniform float a) {
    uniform int intA = (int)a;
    uniform          int    uniIntVal     = -intA - 2;
    uniform unsigned int    uniUIntVal    =  intA + 2;

    print("Test uniform int32: %, %\n", uniIntVal, uniUIntVal);
}

task void print_result() {
    print("Test uniform int32: -7, 7\n");
}
