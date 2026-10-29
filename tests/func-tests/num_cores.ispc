#include "test_static.isph"
// num_cores() calls sysconf() in the C builtins. The sysconf name of another C
// library gives -1 or an unrelated value (#3887).
// rule: skip on arch=xe64

task void f_v(uniform float RET[]) {
    uniform int n = num_cores();
    RET[programIndex] = n > 0 ? 1 : 0;
}

task void result(uniform float RET[]) { RET[programIndex] = 1; }
