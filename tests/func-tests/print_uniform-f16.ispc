#include "test_static.isph"
// rule: skip on jit=true
// rule: skip on arch=ppc64le

task void print_uf(uniform float a) {
    uniform float16 uniFloatVal = (float16)a + 4.75f16;
    print("Test uniform float: %\n", uniFloatVal);
}

task void print_result() { print("Test uniform float: 9.750000\n"); }
