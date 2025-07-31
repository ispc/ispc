#include "test_static.isph"
// rule: skip on jit=true
// rule: run on OS=!windows
task void print_uf(uniform float a) {
    uniform int intA = (int)a;
    uniform          int64  uniLongVal    = -(int64)intA - 10000000000;
    uniform unsigned int64  uniULongVal   =  (int64)intA + 10000000000;

    print("Test uniform int64: %, %\n", uniLongVal, uniULongVal);
}

task void print_result() {
    print("Test uniform int64: -10000000005, 10000000005\n");
}
