#include "../test_static.isph"
enum Enum111{
    One,
    Two
};

// For this to work, an enum->int type-casting need to happen.
int func(int  x) {
    int ret = 0;
    if(x == Two)
        ret = 3;
    return ret;
}


task void f_v(uniform float RET[]) {
    uniform Enum111 enum_ = Two;
    int rval = func(enum_);
    RET[programIndex] = rval;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
