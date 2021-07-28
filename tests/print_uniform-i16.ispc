#include "../test_static.isph"
task void print_uf(uniform float a) {
    uniform int intA = (int)a;
    uniform          int16  uniShortVal   = -intA - 1;
    uniform unsigned int16  uniUShortVal  =  intA + 1;

    print("Test uniform int16: %, %\n", uniShortVal, uniUShortVal);
}

task void print_result() {
    print("Test uniform int16: -6, 6\n");
}
