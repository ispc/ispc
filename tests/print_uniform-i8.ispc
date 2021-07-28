#include "../test_static.isph"
task void print_uf(uniform float a) {
    uniform int intA = (int)a;
    uniform          int8   uniCharVal    = -intA;
    uniform unsigned int8   uniUCharVal   =  intA;

    print("Test uniform int8: %, %\n", uniCharVal, uniUCharVal);
}

task void print_result() {
    print("Test uniform int8: -5, 5\n");
}
