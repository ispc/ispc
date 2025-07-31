#include "test_static.isph"
// rule: skip on jit=true
// rule: run on OS=!windows
task void print_f(uniform float aFOO[]) {
    print("Hello World!\n");
}

task void print_result() {
    print("Hello World!\n");
}
