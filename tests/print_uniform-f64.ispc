#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void print_uf(uniform float a) {
    uniform int intA = (int)a;
    uniform double uniDoubleVal  =  a + 13.25;

    print("Test uniform double: %\n", uniDoubleVal);
}

task void print_result() {
    print("Test uniform double: 18.250000\n");
}
