#include "../test_static.isph"
// rule: skip on arch=xe64
struct InnerVec {
    int i;
};

struct f_sz {
    InnerVec u;
    uniform int tail;
};



export void keep_struct_declared(varying f_sz * uniform s) {
}

task void result(uniform float RET[]) {
    RET[0] = sizeof(f_sz);
}
